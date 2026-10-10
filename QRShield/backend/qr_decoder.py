"""
qr_decoder.py
=============
Safe QR code decoding module for QRShield.
Decodes QR codes from uploaded image bytes WITHOUT visiting or fetching the URL.
"""

import io
import re
from typing import Tuple, Optional
from PIL import Image

# Try pyzbar first, fall back gracefully
# On Windows, pyzbar requires libzbar-64.dll — handle missing DLL gracefully
PYZBAR_AVAILABLE = False
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except (ImportError, OSError, Exception):
    # libzbar-64.dll not found — use OpenCV fallback
    pyzbar_decode = None

# Try opencv as second option
CV2_AVAILABLE = False
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None


# URL pattern for detecting URL payloads
URL_PATTERN = re.compile(
    r'^(https?|ftp)://[^\s/$.?#].[^\s]*$', re.IGNORECASE
)

# Patterns that indicate non-URL payloads
PAYLOAD_TYPES = {
    'URL':      lambda s: bool(re.match(r'^https?://', s, re.I)),
    'FTP':      lambda s: bool(re.match(r'^ftp://', s, re.I)),
    'EMAIL':    lambda s: bool(re.match(r'^mailto:', s, re.I)),
    'PHONE':    lambda s: bool(re.match(r'^tel:', s, re.I)),
    'SMS':      lambda s: bool(re.match(r'^smsto?:', s, re.I)),
    'WIFI':     lambda s: bool(re.match(r'^WIFI:', s, re.I)),
    'GEO':      lambda s: bool(re.match(r'^geo:', s, re.I)),
    'VCARD':    lambda s: bool(re.match(r'^BEGIN:VCARD', s, re.I)),
    'TEXT':     lambda s: True,  # fallback
}


def detect_payload_type(payload: str) -> str:
    for ptype, check in PAYLOAD_TYPES.items():
        if ptype == 'TEXT':
            continue
        if check(payload):
            return ptype
    return 'TEXT'


def decode_qr_from_bytes(image_bytes: bytes) -> Tuple[Optional[str], str, str]:
    """
    Decode QR code from raw image bytes.

    Returns
    -------
    (payload, payload_type, method_used)
    payload      : decoded string, or None if decoding failed
    payload_type : 'URL', 'EMAIL', 'WIFI', 'TEXT', etc.
    method_used  : 'pyzbar', 'opencv', or 'failed'
    """
    payload = None
    method  = 'failed'

    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Ensure RGB for compatibility
        if img.mode not in ('RGB', 'L', 'RGBA'):
            img = img.convert('RGB')
    except Exception as e:
        return None, 'UNKNOWN', f'failed:image_open:{e}'

    # ── Method 1: pyzbar ─────────────────────────────────────────────────────
    if PYZBAR_AVAILABLE and payload is None:
        try:
            decoded = pyzbar_decode(img)
            if decoded:
                payload = decoded[0].data.decode('utf-8', errors='replace').strip()
                method  = 'pyzbar'
        except Exception:
            pass

    # ── Method 2: OpenCV QRCodeDetector ──────────────────────────────────────
    if CV2_AVAILABLE and payload is None:
        try:
            img_cv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2GRAY)
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(img_cv)
            if data:
                payload = data.strip()
                method  = 'opencv'
        except Exception:
            pass

    # ── Method 3: Try with enhanced contrast ─────────────────────────────────
    if payload is None:
        try:
            from PIL import ImageFilter, ImageEnhance
            enhanced = img.convert('L')
            enhanced = ImageEnhance.Contrast(enhanced).enhance(2.0)
            enhanced = enhanced.filter(ImageFilter.SHARPEN)
            if PYZBAR_AVAILABLE:
                decoded = pyzbar_decode(enhanced)
                if decoded:
                    payload = decoded[0].data.decode('utf-8', errors='replace').strip()
                    method  = 'pyzbar-enhanced'
        except Exception:
            pass

    if payload is None:
        return None, 'UNKNOWN', 'failed'

    payload_type = detect_payload_type(payload)
    return payload, payload_type, method


def is_safe_to_analyze(payload_type: str) -> bool:
    """Return True if we can analyze the payload without safety concerns."""
    return payload_type in ('URL', 'FTP', 'TEXT')


def get_analyzable_url(payload: str, payload_type: str) -> Optional[str]:
    """
    Extract the URL to analyze from the decoded payload.
    Returns None for non-URL payloads that can't be analyzed as URLs.
    NEVER navigates to the URL — only returns the string for analysis.
    """
    if payload_type == 'URL':
        return payload
    if payload_type == 'FTP':
        return payload
    # For text payloads, check if the text itself looks like a URL
    if payload_type == 'TEXT':
        stripped = payload.strip()
        if re.match(r'^https?://', stripped, re.I):
            return stripped
    return None
