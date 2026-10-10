"""
main.py
=======
QRShield FastAPI backend.
Provides endpoints for QR analysis, model performance, and health checks.

Start with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
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

from url_features import extract_url_features, FEATURE_NAMES, features_to_vector
from risk_engine   import assess_risk, explain_url
from qr_decoder    import decode_qr_from_bytes, detect_payload_type, get_analyzable_url

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Startup: load models ──────────────────────────────────────────────────────
url_model       = None
image_model     = None
feature_names   = None
MODEL_LOAD_ERROR = None

def load_models():
    global url_model, image_model, feature_names, MODEL_LOAD_ERROR
    try:
        url_model_path = MODEL_DIR / 'url_model.pkl'
        if url_model_path.exists():
            url_model = joblib.load(str(url_model_path))
            logger.info("URL model loaded.")
        else:
            logger.warning("URL model not found — run train_url_model.py first.")

        img_model_path = MODEL_DIR / 'image_model.pkl'
        if img_model_path.exists():
            image_model = joblib.load(str(img_model_path))
            logger.info("Image model loaded.")
        else:
            logger.warning("Image model not found.")

        fn_path = MODEL_DIR / 'feature_names.pkl'
        if fn_path.exists():
            feature_names = joblib.load(str(fn_path))
        else:
            feature_names = FEATURE_NAMES

    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        logger.error(f"Model load error: {e}")


load_models()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title='QRShield API',
    description='Explainable Pre-Navigation QR Code Phishing Detection',
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Serve frontend
if FRONTEND_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')

# Serve result images
if RESULT_DIR.exists():
    app.mount('/results', StaticFiles(directory=str(RESULT_DIR)), name='results')


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/')
async def serve_index():
    index = FRONTEND_DIR / 'index.html'
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({'message': 'QRShield API running. Frontend not found.'})


@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'url_model_loaded':   url_model is not None,
        'image_model_loaded': image_model is not None,
        'model_error':        MODEL_LOAD_ERROR,
        'feature_count':      len(feature_names) if feature_names else 0,
    }


@app.post('/api/analyze')
async def analyze_qr(file: UploadFile = File(...)):
    """
    Analyze an uploaded QR code image.
    Decodes the QR, extracts URL features, runs ML prediction, returns threat report.
    Never visits or fetches the URL.
    """
    # ── Validate file ─────────────────────────────────────────────────────────
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'}

    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. "
                                 f"Allowed: jpeg, png, gif, webp, bmp.")

    raw = await file.read()
    if len(raw) > MAX_SIZE:
        raise HTTPException(413, "File too large. Maximum size is 10 MB.")
    if len(raw) < 100:
        raise HTTPException(400, "File too small to be a valid image.")

    # ── Decode QR ─────────────────────────────────────────────────────────────
    payload, payload_type, method = decode_qr_from_bytes(raw)

    if payload is None:
        return JSONResponse({
            'success': False,
            'error': 'No QR code detected in the uploaded image.',
            'tips': [
                'Ensure the image is clear and well-lit.',
                'The QR code should fill most of the image frame.',
                'Try a higher-resolution image.',
            ]
        }, status_code=422)

    # ── Non-URL payloads ──────────────────────────────────────────────────────
    url_to_analyze = get_analyzable_url(payload, payload_type)

    if url_to_analyze is None:
        return {
            'success':      True,
            'payload':      payload,
            'payload_type': payload_type,
            'decode_method': method,
            'is_url':       False,
            'message': (
                f"QR payload is of type '{payload_type}' — not a URL. "
                f"No URL phishing analysis applicable."
            ),
        }

    # ── Feature extraction ────────────────────────────────────────────────────
    features = extract_url_features(url_to_analyze)
    feat_vec = np.array(features_to_vector(url_to_analyze), dtype=np.float32).reshape(1, -1)

    # ── Model prediction ──────────────────────────────────────────────────────
    model_label = 0
    model_prob  = 0.0
    model_used  = 'none'

    if url_model is not None:
        try:
            model_label = int(url_model.predict(feat_vec)[0])
            if hasattr(url_model, 'predict_proba'):
                probs = url_model.predict_proba(feat_vec)[0]
                model_prob = float(probs[1])
            else:
                model_prob = float(model_label)
            model_used = 'url_model'
        except Exception as e:
            logger.warning(f"URL model prediction failed: {e}")

    # ── Risk assessment ───────────────────────────────────────────────────────
    risk_report = assess_risk(features, model_prob, model_label)
    explanation = explain_url(url_to_analyze, features, risk_report)

    # Clean up internal-only indicator field
    indicators_clean = [
        {'label': i['label'], 'severity': i['severity']}
        for i in risk_report['indicators']
    ]

    return {
        'success':       True,
        'payload':       payload,
        'payload_type':  payload_type,
        'decode_method': method,
        'is_url':        True,
        'url':           url_to_analyze,
        'features': {
            'url_length':        features['url_len'],
            'subdomain_count':   features['subdomain_count'],
            'is_ip':             bool(features['is_ip']),
            'has_https':         bool(features['has_https']),
            'suspicious_tld':    bool(features['suspicious_tld']),
            'has_brand_keyword': bool(features['has_brand_keyword']),
            'is_shortener':      bool(features['is_shortener']),
            'host_entropy':      features['host_entropy'],
        },
        'model_prediction':  risk_report['model_prediction'],
        'model_confidence':  risk_report['model_confidence'],
        'model_used':        model_used,
        'risk_level':        risk_report['risk_level'],
        'risk_score':        risk_report['risk_score'],
        'indicators':        indicators_clean,
        'consequences':      risk_report['consequences'],
        'safeguards':        risk_report['safeguards'],
        'recommendation':    risk_report['recommendation'],
        'explanation':       explanation,
        'model_note':        risk_report['model_note'],
    }


@app.get('/api/performance')
async def get_performance():
    """Return saved model evaluation metrics (does NOT retrain)."""
    url_path = RESULT_DIR / 'url_model_metrics.json'
    img_path = RESULT_DIR / 'image_model_metrics.json'

    result = {}

    if url_path.exists():
        with open(str(url_path)) as f:
            result['url_model'] = json.load(f)
    else:
        result['url_model'] = {'error': 'Run training/train_url_model.py first.'}

    if img_path.exists():
        with open(str(img_path)) as f:
            result['image_model'] = json.load(f)
    else:
        result['image_model'] = {'error': 'Run training/train_image_model.py first.'}

    # List available result images
    result_images = {}
    for fname in [
        'cm_extra_trees.png', 'cm_random_forest.png',
        'roc_comparison.png', 'feature_importance_et.png',
        'feature_importance_rf.png', 'model_comparison.png',
        'roc_image_comparison.png', 'model_comparison_image.png',
    ]:
        if (RESULT_DIR / fname).exists():
            result_images[fname] = f'/results/{fname}'

    result['charts'] = result_images
    return result


@app.post('/api/analyze-url')
async def analyze_url_direct(request: dict):
    """
    Analyze a URL string directly (without QR decoding).
    Useful for testing and manual URL checks.
    """
    url = request.get('url', '').strip()
    if not url:
        raise HTTPException(400, "No URL provided.")
    if len(url) > 2048:
        raise HTTPException(400, "URL too long (max 2048 chars).")

    features  = extract_url_features(url)
    feat_vec  = np.array(features_to_vector(url), dtype=np.float32).reshape(1, -1)

    model_label = 0
    model_prob  = 0.0

    if url_model is not None:
        try:
            model_label = int(url_model.predict(feat_vec)[0])
            if hasattr(url_model, 'predict_proba'):
                model_prob = float(url_model.predict_proba(feat_vec)[0][1])
        except Exception as e:
            logger.warning(f"Prediction error: {e}")

    risk_report = assess_risk(features, model_prob, model_label)
    explanation = explain_url(url, features, risk_report)

    return {
        'url':              url,
        'model_prediction': risk_report['model_prediction'],
        'model_confidence': risk_report['model_confidence'],
        'risk_level':       risk_report['risk_level'],
        'risk_score':       risk_report['risk_score'],
        'indicators':       [{'label': i['label'], 'severity': i['severity']}
                             for i in risk_report['indicators']],
        'consequences':     risk_report['consequences'],
        'safeguards':       risk_report['safeguards'],
        'recommendation':   risk_report['recommendation'],
        'explanation':      explanation,
        'features':         features,
    }

