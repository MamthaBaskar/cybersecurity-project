"""
url_features.py
===============
Static URL feature extractor for QRShield.
All 30 features are derived purely from URL string analysis — no HTTP requests.
Safe for pre-navigation threat detection.
"""

import re
import math
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Tuple, Optional
import tldextract

# Setup tldextract with local cache
_extractor = tldextract.TLDExtract(cache_dir=None)

# Known suspicious / heavily abused TLDs (frequently used in disposable phishing)
SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'club', 'work', 'party',
    'loan', 'download', 'bid', 'win', 'review', 'trade', 'date', 'racing',
    'men', 'faith', 'accountant', 'science', 'stream', 'gdn', 'kim', 'shop',
    'buzz', 'monster', 'rest', 'fit', 'site', 'pw', 'space', 'website',
}

# Major brands commonly targeted for credential phishing
KNOWN_BRANDS = {
    'paypal', 'amazon', 'apple', 'google', 'microsoft', 'facebook',
    'instagram', 'twitter', 'netflix', 'ebay', 'coinbase', 'binance',
    'chase', 'wellsfargo', 'citibank', 'github', 'dropbox', 'linkedin',
    'steam', 'adobe', 'walmart', 'target', 'bankofamerica', 'usps',
    'fedex', 'dhl', 'ups', 'whatsapp', 'telegram'
}

# Promotional / urgent keywords common in phishing lures
PHISHING_KEYWORDS = {
    'free', 'prize', 'winner', 'click', 'limited', 'offer', 'deal',
    'urgent', 'suspend', 'bonus', 'gift', 'reward', 'crypto', 'nft',
    'airdrop', 'claim', 'redeem', 'discount', 'jackpot', 'lottery',
    'robux', 'vbucks', 'stimulus'
}

# Sensitive credential / account action keywords
SECURITY_KEYWORDS = {
    'login', 'signin', 'account', 'verify', 'update', 'confirm',
    'password', 'security', 'billing', 'recover', 'restore', 'authenticate'
}

SHORTENERS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'short.io', 'tiny.cc', 'rb.gy', 'cutt.ly'
}


def normalize_url(raw_url: str) -> Tuple[str, bool, Optional[str]]:
    """
    Normalize raw URL string:
    - Strips whitespace.
    - If URL starts with a scheme (http://, https://, ftp://), preserves it.
    - If URL lacks a scheme (e.g., www.google.com, google.com), assumes https://.
    Returns: (normalized_url, scheme_assumed, assumed_scheme)
    """
    url = raw_url.strip()
    if not url:
        return url, False, None

    if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url):
        return url, False, None

    # Bare hostname or path without scheme: default to https
    return 'https://' + url, True, 'https'


def validate_url_syntax(url_str: str) -> Tuple[bool, str]:
    """
    Validate that input looks like a valid URL or hostname.
    Returns (is_valid, error_message).
    """
    if not url_str or not url_str.strip():
        return False, "URL cannot be empty."

    s = url_str.strip()
    if len(s) > 2048:
        return False, "URL exceeds maximum permitted length of 2048 characters."

    # Check for illegal control characters or spaces
    if any(ord(c) < 32 or ord(c) == 127 for c in s):
        return False, "URL contains illegal control characters."
    if ' ' in s:
        return False, "URL cannot contain spaces."

    norm, _, _ = normalize_url(s)
    try:
        parsed = urlparse(norm)
    except Exception as e:
        return False, f"Malformed URL syntax: {str(e)}"

    host = parsed.netloc.split(':')[0]
    if not host:
        return False, "URL missing valid hostname or netloc."

    # Must contain at least one dot or be an IP or localhost
    if '.' not in host and host != 'localhost':
        return False, f"Hostname '{host}' does not contain a valid domain extension or IP address."

    return True, ""


def _entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _is_ip(host: str) -> int:
    """Return 1 if host is an IPv4 or IPv6 address."""
    parts = host.split('.')
    if len(parts) == 4:
        try:
            if all(0 <= int(p) <= 255 for p in parts):
                return 1
        except ValueError:
            pass
    if host.startswith('[') and host.endswith(']'):
        return 1
    return 0


def _check_brand_impersonation(domain: str, subdomain: str, path: str, url_lower: str) -> int:
    """
    Detect whether a known brand name is being impersonated.
    Returns 1 ONLY if a recognized brand name appears in:
      - Subdomain (e.g. paypal.security-alert.com)
      - Modified/hyphenated domain (e.g. paypal-security.com)
      - Path on a non-brand domain (e.g. phishingsite.com/paypal/login)
    Returns 0 if the registered domain legitimately IS the brand (e.g. google.com, paypal.com).
    """
    domain_clean = domain.lower()
    sub_parts = [p.lower() for p in subdomain.split('.') if p]
    path_lower = path.lower()

    for brand in KNOWN_BRANDS:
        # If the registered domain is the legitimate brand (e.g. google in google.com), not an impersonation
        if domain_clean == brand:
            continue

        # 1. Brand in subdomain of a different domain (e.g., google.com.attacker.xyz)
        if brand in sub_parts:
            return 1

        # 2. Brand modified with hyphens/digits in domain (e.g., paypal-update, secure-google)
        if brand in domain_clean and domain_clean != brand:
            return 1

        # 3. Brand directory in path on untrusted domain
        if f'/{brand}/' in path_lower or path_lower.endswith(f'/{brand}'):
            return 1

    return 0


def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extract 30 static lexical features from a URL string.
    Ensures normalized bare hostname handling.
    """
    norm_url, _, _ = normalize_url(url)

    try:
        parsed = urlparse(norm_url)
    except Exception:
        parsed = urlparse('http://invalid')

    scheme   = (parsed.scheme or '').lower()
    netloc   = (parsed.netloc or '').lower()
    path     = parsed.path or ''
    query    = parsed.query or ''
    fragment = parsed.fragment or ''
    host     = netloc.split(':')[0]  # strip port

    # Extract domain components via tldextract
    ext = _extractor(norm_url)
    domain = ext.domain.lower() if ext.domain else host
    suffix = ext.suffix.lower() if ext.suffix else ''
    subdomain = ext.subdomain.lower() if ext.subdomain else ''

    # Subdomains excluding standard 'www'
    non_www_subs = [s for s in subdomain.split('.') if s and s != 'www']
    subdomain_count = len(non_www_subs)

    full_path = path + ('?' + query if query else '') + ('#' + fragment if fragment else '')
    url_lower = norm_url.lower()

    # ── 30 Features ──────────────────────────────────────────────────────────
    # 1. URL length
    url_len = len(norm_url)

    # 2. Host length
    host_len = len(host)

    # 3. Path length
    path_len = len(path)

    # 4. Dot count
    num_dots = norm_url.count('.')

    # 5. Hyphen count
    num_hyphens = norm_url.count('-')

    # 6. Underscore count
    num_underscores = norm_url.count('_')

    # 7. Slash count
    num_slashes = norm_url.count('/')

    # 8. Question mark count
    num_questions = norm_url.count('?')

    # 9. At count
    num_at = norm_url.count('@')

    # 10. Equals count
    num_equals = norm_url.count('=')

    # 11. Ampersand count
    num_ampersands = norm_url.count('&')

    # 12. Percent count
    num_percent = norm_url.count('%')

    # 13. Digit count
    num_digits = sum(c.isdigit() for c in norm_url)

    # 14. Digit ratio
    digit_ratio = round(num_digits / max(url_len, 1), 4)

    # 15. Subdomain count (excluding www)
    # Reflects genuine nesting (0 for apex or www, >0 for extra subdomains)
    subdomain_count = subdomain_count

    # 16. Is IP address
    is_ip = _is_ip(host)

    # 17. Has HTTPS
    has_https = 1 if scheme == 'https' else 0

    # 18. Suspicious TLD
    suspicious_tld = 1 if suffix in SUSPICIOUS_TLDS else 0

    # 19. Domain length
    domain_len = len(domain)

    # 20. Host entropy
    host_entropy = round(_entropy(host), 4)

    # 21. Path entropy
    path_entropy = round(_entropy(full_path), 4)

    # 22. Brand impersonation (only 1 if brand appears on non-brand domain)
    brand_impersonation = _check_brand_impersonation(domain, subdomain, path, url_lower)

    # 23. Phishing promotional keyword in URL
    has_phishing_keyword = 1 if any(k in url_lower for k in PHISHING_KEYWORDS) else 0

    # 24. URL shortener
    is_shortener = 1 if host in SHORTENERS or f"{domain}.{suffix}" in SHORTENERS else 0

    # 25. Has port
    has_port = 1 if re.search(r':\d+', netloc) else 0

    # 26. Has fragment
    has_fragment = 1 if fragment else 0

    # 27. Query param count
    query_param_count = len(parse_qs(query))

    # 28. Double slash in path
    has_double_slash = 1 if '//' in path else 0

    # 29. Hex characters in URL
    has_hex = 1 if re.search(r'%[0-9a-fA-F]{2}', norm_url) else 0

    # 30. Deep subdomain (non-www subdomain levels >= 2)
    deep_subdomain = 1 if subdomain_count >= 2 else 0

    return {
        'url_len':             url_len,
        'host_len':            host_len,
        'path_len':            path_len,
        'num_dots':            num_dots,
        'num_hyphens':         num_hyphens,
        'num_underscores':     num_underscores,
        'num_slashes':         num_slashes,
        'num_questions':       num_questions,
        'num_at':              num_at,
        'num_equals':          num_equals,
        'num_ampersands':      num_ampersands,
        'num_percent':         num_percent,
        'num_digits':          num_digits,
        'digit_ratio':         digit_ratio,
        'subdomain_count':     subdomain_count,
        'is_ip':               is_ip,
        'has_https':           has_https,
        'suspicious_tld':      suspicious_tld,
        'domain_len':          domain_len,
        'host_entropy':        host_entropy,
        'path_entropy':        path_entropy,
        'brand_impersonation': brand_impersonation,
        'has_phishing_keyword':has_phishing_keyword,
        'is_shortener':        is_shortener,
        'has_port':            has_port,
        'has_fragment':        has_fragment,
        'query_param_count':   query_param_count,
        'has_double_slash':    has_double_slash,
        'has_hex':             has_hex,
        'deep_subdomain':      deep_subdomain,
    }


# Ordered list of feature names (must match training order)
FEATURE_NAMES = list(extract_url_features("https://example.com").keys())


def features_to_vector(url: str) -> list:
    """Return feature values as an ordered list matching FEATURE_NAMES."""
    feats = extract_url_features(url)
    return [feats[k] for k in FEATURE_NAMES]
