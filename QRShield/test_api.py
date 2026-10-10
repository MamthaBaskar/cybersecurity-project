"""
test_api.py
===========
Live integration verification script for QRShield HTTP endpoints.
Tests running server at http://127.0.0.1:8000.
"""

import urllib.request
import urllib.error
import json

BASE_URL = 'http://127.0.0.1:8000'

print("=" * 60)
print("  QRShield Live Server Verification")
print("=" * 60)

# 1. Health check
print("\n[1] Health Check Endpoint (/health):")
with urllib.request.urlopen(f'{BASE_URL}/health', timeout=5) as resp:
    data = json.loads(resp.read().decode())
    print("  Status code:       200 OK")
    print("  Server status:    ", data.get('status'))
    print("  URL model loaded: ", data.get('url_model_loaded'))
    print("  Image model:      ", data.get('image_model_loaded'))
    print("  Feature count:    ", data.get('feature_count'))

# 2. Benchmark URL tests
benchmarks = [
    ("www.google.com", "Legitimate"),
    ("https://www.google.com", "Legitimate"),
    ("google.com", "Legitimate"),
    ("https://github.com", "Legitimate"),
    ("github.com", "Legitimate"),
    ("https://www.wikipedia.org", "Legitimate"),
    ("http://paypal-security-update.xyz/login", "Phishing"),
    ("http://192.168.1.1/admin/login.php", "Phishing"),
    ("http://google.com.phish-account.top/verify", "Phishing")
]

print("\n[2] Direct URL Analysis (/api/analyze-url):")
print(f"{'URL':40s} Expected    Predicted    Confidence  Prob(Phish)  Risk Level  Status")
print("-" * 105)

all_passed = True
for url, expected in benchmarks:
    payload = json.dumps({'url': url}).encode('utf-8')
    req = urllib.request.Request(
        f'{BASE_URL}/api/analyze-url',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode())
        pred = res.get('model_prediction')
        conf = res.get('prediction_confidence')
        prob = res.get('phishing_probability')
        risk = res.get('risk_level')
        conf_str = f"{conf*100:.1f}%" if conf is not None else "N/A"
        prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"
        ok = (pred == expected)
        if not ok:
            all_passed = False
        print(f"{url:40s} {expected:11s} {pred:11s}  {conf_str:10s}  {prob_str:11s}  {risk:10s}  {'PASS' if ok else 'FAIL'}")

# 3. Validation rejection
print("\n[3] URL Syntax Validation Errors:")
for invalid in ["", "https://bad url.com", "notadomain"]:
    payload = json.dumps({'url': invalid}).encode('utf-8')
    req = urllib.request.Request(
        f'{BASE_URL}/api/analyze-url',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"  Unexpected 200 OK on invalid URL: '{invalid}'")
    except urllib.error.HTTPError as e:
        print(f"  Correctly rejected '{invalid}' with HTTP {e.code}: {e.read().decode()[:70]}...")

# 4. Performance endpoint
print("\n[4] Performance Metrics Endpoint (/api/performance):")
with urllib.request.urlopen(f'{BASE_URL}/api/performance', timeout=5) as resp:
    perf = json.loads(resp.read().decode())
    um = perf.get('url_model', {})
    print("  Selected Model:   ", um.get('selected_model'))
    print("  Dataset total:    ", um.get('dataset', {}).get('total_samples'))
    print("  Charts available: ", list(perf.get('charts', {}).keys()))

print("\n" + "=" * 60)
print(f"  Live Server Verification Result: {'ALL TESTS PASSED' if all_passed else 'FAILURES DETECTED'}")
print("=" * 60)
