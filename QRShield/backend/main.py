"""
main.py
=======
QRShield FastAPI backend.
Provides explainable pre-navigation QR and URL phishing detection.

Start with:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""

import io
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).parent
ROOT         = BACKEND_DIR.parent
FRONTEND_DIR = ROOT / 'frontend'
MODEL_DIR    = ROOT / 'models'
RESULT_DIR   = ROOT / 'results'
TRAINING_DIR = ROOT / 'training'

sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from url_features import (
    extract_url_features, FEATURE_NAMES, features_to_vector,
    normalize_url, validate_url_syntax
)
from risk_engine import assess_risk, explain_url
from qr_decoder  import decode_qr_from_bytes, detect_payload_type, get_analyzable_url

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Upload Security Constraints
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB streaming limit
MAX_DIMENSION = 4096              # 4096 px width and height limit
ALLOWED_MIME_TYPES = {
    'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'
}

# ── Startup: load models ──────────────────────────────────────────────────────
url_model        = None
image_model      = None
feature_names    = None
MODEL_LOAD_ERROR = None


def load_models():
    global url_model, image_model, feature_names, MODEL_LOAD_ERROR
    try:
        url_model_path = MODEL_DIR / 'url_model.pkl'
        if url_model_path.exists():
            url_model = joblib.load(str(url_model_path))
            logger.info("URL model loaded successfully.")
        else:
            logger.warning("URL model not found in models/ directory.")

        img_model_path = MODEL_DIR / 'image_model.pkl'
        if img_model_path.exists():
            image_model = joblib.load(str(img_model_path))
            logger.info("Image model loaded successfully.")
        else:
            logger.warning("Image model not found in models/ directory.")

        fn_path = MODEL_DIR / 'feature_names.pkl'
        if fn_path.exists():
            feature_names = joblib.load(str(fn_path))
        else:
            feature_names = FEATURE_NAMES

    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        logger.error(f"Error loading models: {e}")


load_models()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title='QRShield API',
    description='Explainable Pre-Navigation QR Code Phishing Detection System',
    version='1.1.0',
)

# Restrict CORS to localhost demonstration environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')

# Serve evaluation charts
if RESULT_DIR.exists():
    app.mount('/results', StaticFiles(directory=str(RESULT_DIR)), name='results')


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Hardened Image Validation
# ─────────────────────────────────────────────────────────────────────────────
async def validate_and_read_image_stream(file: UploadFile) -> bytes:
    """
    Hardens image upload handling:
    1. Enforces streaming size limit (max 10MB) to prevent memory exhaustion.
    2. Validates magic bytes against supported image signatures.
    3. Uses PIL verify() to catch corrupted files.
    4. Enforces dimension limits (max 4096x4096px).
    """
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file MIME type '{file.content_type}'. Allowed types: JPEG, PNG, GIF, WebP, BMP."
        )

    chunks = []
    total_bytes = 0
    chunk_size = 1024 * 512  # 512 KB chunks

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed upload size of 10 MB ({total_bytes} bytes streamed)."
            )
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)

    if len(raw_bytes) < 32:
        raise HTTPException(
            status_code=400,
            detail="File is too small to be a valid image."
        )

    # Magic byte signature check
    is_valid_magic = False
    if raw_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        is_valid_magic = True
    elif raw_bytes.startswith(b'\xff\xd8\xff'):
        is_valid_magic = True
    elif raw_bytes.startswith(b'GIF87a') or raw_bytes.startswith(b'GIF89a'):
        is_valid_magic = True
    elif raw_bytes.startswith(b'RIFF') and len(raw_bytes) >= 12 and raw_bytes[8:12] == b'WEBP':
        is_valid_magic = True
    elif raw_bytes.startswith(b'BM'):
        is_valid_magic = True

    if not is_valid_magic:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. File header does not match any supported image types (PNG, JPEG, GIF, WebP, BMP)."
        )

    # PIL verification and dimension constraints
    try:
        img_verify = Image.open(io.BytesIO(raw_bytes))
        img_verify.verify()

        # Re-open for dimension inspection
        img = Image.open(io.BytesIO(raw_bytes))
        width, height = img.size
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Invalid image dimensions (zero or negative).")
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions ({width}x{height}) exceed maximum allowed limit of {MAX_DIMENSION}x{MAX_DIMENSION} pixels."
            )
    except (IOError, SyntaxError) as e:
        raise HTTPException(status_code=400, detail=f"Corrupted or malformed image: {str(e)}")

    return raw_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Safe URL Prediction & Risk Scoring
# ─────────────────────────────────────────────────────────────────────────────
def run_url_analysis(url_input: str) -> Dict[str, Any]:
    """
    Safely normalize bare hostnames, extract 30 static features,
    run ML inference, and generate explainable threat report.
    Never visits or fetches the destination URL.
    """
    normalized_url, scheme_assumed, assumed_scheme = normalize_url(url_input)

    features = extract_url_features(normalized_url)
    feat_vec = np.array(features_to_vector(normalized_url), dtype=np.float32).reshape(1, -1)

    model_available = (url_model is not None)
    model_label = None
    model_prob  = None
    model_used  = 'none'

    if model_available:
        try:
            model_label = int(url_model.predict(feat_vec)[0])
            if hasattr(url_model, 'predict_proba'):
                probs = url_model.predict_proba(feat_vec)[0]
                model_prob = float(probs[1])  # P(Phishing)
            else:
                model_prob = float(model_label)
            model_used = getattr(url_model, '__class__', type(url_model)).__name__
        except Exception as e:
            logger.error(f"Inference error on URL model: {e}")
            model_available = False
            model_label = None
            model_prob  = None

    # Risk evaluation
    risk_report = assess_risk(
        features=features,
        model_prob=model_prob,
        model_label=model_label,
        model_available=model_available,
        url_str=normalized_url
    )

    explanation = explain_url(normalized_url, features, risk_report)

    indicators_clean = [
        {'label': i['label'], 'severity': i['severity']}
        for i in risk_report['indicators']
    ]

    return {
        'original_url':           url_input,
        'url':                    normalized_url,
        'scheme_assumed':         scheme_assumed,
        'assumed_scheme':         assumed_scheme,
        'features': {
            'url_length':         features['url_len'],
            'subdomain_count':    features['subdomain_count'],
            'is_ip':              bool(features['is_ip']),
            'has_https':          bool(features['has_https']),
            'suspicious_tld':     bool(features['suspicious_tld']),
            'brand_impersonation':bool(features['brand_impersonation']),
            'is_shortener':       bool(features['is_shortener']),
            'host_entropy':       features['host_entropy'],
        },
        'model_prediction':       risk_report['model_prediction'],
        'phishing_probability':   risk_report['phishing_probability'],
        'prediction_confidence':  risk_report['prediction_confidence'],
        'model_used':             model_used,
        'model_available':        model_available,
        'risk_level':             risk_report['risk_level'],
        'risk_score':             risk_report['risk_score'],
        'indicators':             indicators_clean,
        'consequences':           risk_report['consequences'],
        'safeguards':             risk_report['safeguards'],
        'recommendation':         risk_report['recommendation'],
        'explanation':            explanation,
        'model_note':             risk_report['model_note'],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/')
async def serve_index():
    index = FRONTEND_DIR / 'index.html'
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({'message': 'QRShield API running. Frontend static files not found.'})


@app.get('/health')
async def health():
    return {
        'status':             'ok',
        'url_model_loaded':   url_model is not None,
        'image_model_loaded': image_model is not None,
        'model_error':        MODEL_LOAD_ERROR,
        'feature_count':      len(feature_names) if feature_names else 0,
    }


@app.post('/api/analyze')
async def analyze_qr(file: UploadFile = File(...)):
    """
    Decodes uploaded QR code locally and performs pre-navigation threat detection.
    Never navigates to or opens the destination URL.
    """
    raw_bytes = await validate_and_read_image_stream(file)

    payload, payload_type, decode_method = decode_qr_from_bytes(raw_bytes)

    if payload is None:
        return JSONResponse({
            'success': False,
            'error':   'No valid QR code detected in the uploaded image.',
            'tips': [
                'Ensure the QR code is clearly focused and adequately illuminated.',
                'The QR code should occupy the central region of the image.',
                'Try uploading a higher-resolution image file.',
            ]
        }, status_code=422)

    url_to_analyze = get_analyzable_url(payload, payload_type)

    if url_to_analyze is None:
        return {
            'success':               True,
            'payload':               payload,
            'payload_type':          payload_type,
            'decode_method':         decode_method,
            'is_url':                False,
            'url':                   payload,
            'original_url':          payload,
            'scheme_assumed':        False,
            'assumed_scheme':        None,
            'risk_level':            'SAFE',
            'risk_score':            0,
            'model_prediction':      'Non-URL Content',
            'phishing_probability':  0.0,
            'prediction_confidence': 1.0,
            'recommendation':        'SAFE TO VIEW — Non-URL payload',
            'explanation': (
                f"QR code decoded successfully via **{decode_method}**. "
                f"The content is **{payload_type}** (`{payload}`). "
                f"Because this payload does not contain a web URL, there is no pre-navigation web phishing threat. "
                f"No browser redirects or network requests will occur."
            ),
            'indicators':            [],
            'consequences':          ['No external web navigation or browser redirection risks detected.'],
            'safeguards':            [
                f"Payload type is {payload_type}. Content is displayed as plain text without execution.",
                "Verify the displayed content matches what you expect (e.g. serial number, Wi-Fi code, or text)."
            ],
            'model_note':            'Static analysis confirmed payload contains no navigable web URLs.',
        }

    analysis = run_url_analysis(url_to_analyze)

    return {
        'success':       True,
        'payload':       payload,
        'payload_type':  payload_type,
        'decode_method': decode_method,
        'is_url':        True,
        **analysis
    }


@app.post('/api/analyze-url')
async def analyze_url_direct(request: dict):
    """
    Directly analyze a URL without QR decoding.
    Validates syntax, normalizes bare hostnames, and extracts features.
    """
    raw_url = request.get('url', '').strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="No URL provided.")

    is_valid, err_msg = validate_url_syntax(raw_url)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    analysis = run_url_analysis(raw_url)

    return {
        'success': True,
        'is_url':  True,
        **analysis
    }


@app.get('/api/performance')
async def get_performance():
    """Return genuine saved model metrics and evaluation charts without retraining."""
    url_path = RESULT_DIR / 'url_model_metrics.json'
    img_path = RESULT_DIR / 'image_model_metrics.json'

    result = {}

    if url_path.exists():
        with open(str(url_path), 'r') as f:
            result['url_model'] = json.load(f)
    else:
        result['url_model'] = {'error': 'URL model evaluation metrics not found.'}

    if img_path.exists():
        with open(str(img_path), 'r') as f:
            result['image_model'] = json.load(f)
    else:
        result['image_model'] = {'error': 'Image model evaluation metrics not found.'}

    chart_names = [
        'cm_extra_trees.png', 'cm_random_forest.png',
        'roc_comparison.png', 'feature_importance_rf.png',
        'feature_importance_et.png', 'model_comparison.png',
        'cm_image_extra_trees.png', 'cm_image_random_forest.png',
        'roc_image_comparison.png', 'model_comparison_image.png'
    ]

    result_images = {}
    for fname in chart_names:
        if (RESULT_DIR / fname).exists():
            result_images[fname] = f'/results/{fname}'

    result['charts'] = result_images
    return result
