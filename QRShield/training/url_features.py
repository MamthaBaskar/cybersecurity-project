"""
url_features.py
Static URL feature extractor for QRShield.
All features are derived from the URL string alone — no HTTP requests.
"""

import re
import math
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

# ── known safe TLDs / suspicious TLDs ─────────────────────────────────────────
SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'club', 'work', 'party',
    'loan', 'download', 'bid', 'win', 'review', 'trade', 'date', 'racing',
    'men', 'faith', 'accountant', 'science', 'stream', 'gdn', 'kim',
}

BRAND_KEYWORDS = [
    'paypal', 'amazon', 'apple', 'google', 'microsoft', 'facebook',
    'instagram', 'twitter', 'netflix', 'ebay', 'bank', 'secure', 'login',
    'signin', 'account', 'verify', 'update', 'confirm', 'wallet',
    'blockchain', 'coinbase', 'binance', 'chase', 'wellsfargo', 'citibank',
]

PHISHING_KEYWORDS = [
    'free', 'prize', 'winner', 'click', 'now', 'limited', 'offer', 'deal',
    'urgent', 'suspend', 'alert', 'warning', 'bonus', 'gift', 'reward',
    'crypto', 'nft', 'airdrop', 'claim', 'redeem', 'discount', 'cheap',
]


def _entropy(s: str) -> float:
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _is_ip(host: str) -> int:
    """Return 1 if host looks like an IPv4 address."""
    parts = host.split('.')
    if len(parts) == 4:
        try:
            if all(0 <= int(p) <= 255 for p in parts):
                return 1
        except ValueError:
            pass
    # IPv6 brackets
    if host.startswith('['):
        return 1
    return 0


def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extract 30 static lexical features from a URL string.
    Returns an ordered dict that matches FEATURE_NAMES.
    """
    url = url.strip()

    # ── parse ──────────────────────────────────────────────────────────────────
    try:
        parsed = urlparse(url if '://' in url else 'http://' + url)
    except Exception:
        parsed = urlparse('http://invalid')

    scheme   = (parsed.scheme or '').lower()
    netloc   = (parsed.netloc or '').lower()
    path     = parsed.path or ''
    query    = parsed.query or ''
    fragment = parsed.fragment or ''
    host     = netloc.split(':')[0]  # strip port
    parts    = host.split('.')

    tld = parts[-1] if len(parts) >= 2 else ''
    domain = parts[-2] if len(parts) >= 2 else host
    subdomains = parts[:-2]  # everything before domain.tld

    full_path = path + ('?' + query if query else '') + ('#' + fragment if fragment else '')

    # ── 1. URL length ──────────────────────────────────────────────────────────
    url_len = len(url)

    # ── 2. Hostname length ────────────────────────────────────────────────────
    host_len = len(host)

    # ── 3. Path length ────────────────────────────────────────────────────────
    path_len = len(path)

    # ── 4. Number of dots in URL ──────────────────────────────────────────────
    num_dots = url.count('.')

    # ── 5. Number of hyphens ──────────────────────────────────────────────────
    num_hyphens = url.count('-')

    # ── 6. Number of underscores ─────────────────────────────────────────────
    num_underscores = url.count('_')

    # ── 7. Number of slashes ─────────────────────────────────────────────────
    num_slashes = url.count('/')

    # ── 8. Number of question marks ──────────────────────────────────────────
    num_questions = url.count('?')

    # ── 9. Number of @ symbols ───────────────────────────────────────────────
    num_at = url.count('@')

    # ── 10. Number of equals signs ───────────────────────────────────────────
    num_equals = url.count('=')

    # ── 11. Number of ampersands ─────────────────────────────────────────────
    num_ampersands = url.count('&')

    # ── 12. Number of percent-encoded chars ──────────────────────────────────
    num_percent = url.count('%')

    # ── 13. Number of digits in URL ──────────────────────────────────────────
    num_digits = sum(c.isdigit() for c in url)

    # ── 14. Digit ratio ──────────────────────────────────────────────────────
    digit_ratio = num_digits / max(url_len, 1)

    # ── 15. Subdomain count ───────────────────────────────────────────────────
    subdomain_count = len(subdomains)

    # ── 16. Is IP address ────────────────────────────────────────────────────
    is_ip = _is_ip(host)

    # ── 17. HTTPS ────────────────────────────────────────────────────────────
    has_https = 1 if scheme == 'https' else 0

    # ── 18. Suspicious TLD ───────────────────────────────────────────────────
    suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0

    # ── 19. Domain length ────────────────────────────────────────────────────
    domain_len = len(domain)

    # ── 20. Entropy of hostname ──────────────────────────────────────────────
    host_entropy = round(_entropy(host), 4)

    # ── 21. Entropy of path ──────────────────────────────────────────────────
    path_entropy = round(_entropy(full_path), 4)

    # ── 22. Brand keyword in URL ─────────────────────────────────────────────
    url_lower = url.lower()
    has_brand_keyword = 1 if any(k in url_lower for k in BRAND_KEYWORDS) else 0

    # ── 23. Phishing keyword in URL ──────────────────────────────────────────
    has_phishing_keyword = 1 if any(k in url_lower for k in PHISHING_KEYWORDS) else 0

    # ── 24. URL shortener ────────────────────────────────────────────────────
    SHORTENERS = {'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
                  'buff.ly', 'adf.ly', 'short.io', 'tiny.cc', 'rb.gy', 'cutt.ly'}
    is_shortener = 1 if host in SHORTENERS else 0

    # ── 25. Has port ─────────────────────────────────────────────────────────
    has_port = 1 if re.search(r':\d+', netloc) else 0

    # ── 26. Has fragment ─────────────────────────────────────────────────────
    has_fragment = 1 if fragment else 0

    # ── 27. Query param count ────────────────────────────────────────────────
    query_param_count = len(parse_qs(query))

    # ── 28. Double slash in path ─────────────────────────────────────────────
    has_double_slash = 1 if '//' in path else 0

    # ── 29. Hex in URL ───────────────────────────────────────────────────────
    has_hex = 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0

    # ── 30. Number of subdomains > 3 (deep nesting) ──────────────────────────
    deep_subdomain = 1 if subdomain_count > 3 else 0

    return {
        'url_len': url_len,
        'host_len': host_len,
        'path_len': path_len,
        'num_dots': num_dots,
        'num_hyphens': num_hyphens,
        'num_underscores': num_underscores,
        'num_slashes': num_slashes,
        'num_questions': num_questions,
        'num_at': num_at,
        'num_equals': num_equals,
        'num_ampersands': num_ampersands,
        'num_percent': num_percent,
        'num_digits': num_digits,
        'digit_ratio': digit_ratio,
        'subdomain_count': subdomain_count,
        'is_ip': is_ip,
        'has_https': has_https,
        'suspicious_tld': suspicious_tld,
        'domain_len': domain_len,
        'host_entropy': host_entropy,
        'path_entropy': path_entropy,
        'has_brand_keyword': has_brand_keyword,
        'has_phishing_keyword': has_phishing_keyword,
        'is_shortener': is_shortener,
        'has_port': has_port,
        'has_fragment': has_fragment,
        'query_param_count': query_param_count,
        'has_double_slash': has_double_slash,
        'has_hex': has_hex,
        'deep_subdomain': deep_subdomain,
    }


# Ordered list of feature names (must match training order)
FEATURE_NAMES = list(extract_url_features("http://example.com").keys())


def features_to_vector(url: str) -> list:
    """Return feature values as an ordered list matching FEATURE_NAMES."""
    feats = extract_url_features(url)
    return [feats[k] for k in FEATURE_NAMES]

