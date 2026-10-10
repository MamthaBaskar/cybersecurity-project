"""
risk_engine.py
==============
Explainable Rule-Based Threat Scoring and Guidance Engine for QRShield.
Combines 30 static URL features with machine-learning model probabilities.
Provides transparent indicators, consequences, and actionable safeguards.
"""

from typing import Dict, Any, List, Optional
import tldextract

_tld_extractor = tldextract.TLDExtract(cache_dir=None)

# Predefined heuristic threat rules
# (feature_key, condition_or_threshold, display_label, severity_level)
RULES = [
    ('is_ip',               1,    'IP address used instead of domain name',          'high'),
    ('suspicious_tld',      1,    'Suspicious/disposable top-level domain',          'high'),
    ('num_at',              lambda v: v >= 1, '@ symbol in URL (potential URL spoofing)',      'high'),
    ('brand_impersonation', 1,    'Impersonates a known brand on an untrusted domain','high'),
    ('is_shortener',        1,    'URL shortener used (obscures true destination)',  'medium'),
    ('has_phishing_keyword',1,    'Contains lure/phishing promotional keyword',      'medium'),
    ('deep_subdomain',      1,    'Deeply nested subdomains (≥2 levels beyond www)', 'medium'),
    ('has_double_slash',    1,    'Double slash in path (redirect evasion pattern)', 'medium'),
    ('has_hex',             1,    'Hex-encoded characters in URL',                   'low'),
    ('has_port',            1,    'Non-standard port specified in URL',              'low'),
    ('num_hyphens', lambda v: v >= 3, 'Excessive hyphens in URL (≥3)',               'low'),
    ('url_len',     lambda v: v > 90, 'Unusually long URL (>90 characters)',         'low'),
    ('subdomain_count', lambda v: v >= 2, 'Multiple subdomain levels',               'low'),
    ('query_param_count', lambda v: v >= 4, 'Multiple query parameters (≥4)',        'low'),
    ('digit_ratio',  lambda v: v > 0.25, 'High digit-to-character ratio (>25%)',     'low'),
    ('host_entropy', lambda v: v > 4.2, 'High hostname entropy (algorithmically generated)', 'medium'),
    ('num_percent',  lambda v: v >= 3, 'Multiple percent-encoded characters (≥3)',   'low'),
]

SEVERITY_SCORE = {'critical': 40, 'high': 20, 'medium': 8, 'low': 3}

CONSEQUENCE_MAP = {
    'Impersonates a known brand on an untrusted domain':
        'High risk of credential theft, session hijacking, or account takeover by spoofing a familiar service.',
    'IP address used instead of domain name':
        'Direct connection to an unverified server, commonly used for temporary or bulletproof phishing hosts.',
    'URL shortener used (obscures true destination)':
        'True destination is hidden; may lead through redirection chains to malicious landing pages or drive-by downloads.',
    '@ symbol in URL (potential URL spoofing)':
        'Browser may treat text before @ as authentication credentials and navigate to a completely different destination.',
    'Suspicious/disposable top-level domain':
        'Domain registered on a low-cost or abused TLD with minimal anti-abuse enforcement, frequent in automated phishing campaigns.',
    'Contains lure/phishing promotional keyword':
        'Social engineering tactic attempting to induce urgency or exploit financial incentives (e.g. fake prizes, airdrops).',
    'Deeply nested subdomains (≥2 levels beyond www)':
        'Designed to visually mimic legitimate institutional or brand hierarchy on mobile and QR scanner screens.',
    'High hostname entropy (algorithmically generated)':
        'Domain appears randomly generated (DGA), characteristic of evasive botnet or malware C2 infrastructure.',
    'Unusually long URL (>90 characters)':
        'Overly complex URL structured to conceal the true landing destination on small display interfaces.',
}

SAFEGUARD_MAP = {
    'critical': [
        'DO NOT proceed to this destination.',
        'Do not scan or share this QR code with others.',
        'Report this QR code to your organization or IT security team immediately.',
    ],
    'high': [
        'DO NOT enter passwords, credit cards, or two-factor authentication codes.',
        'Verify the destination URL independently using an official browser bookmark or search engine.',
        'Check the domain name character by character for typosquatting or look-alike characters.',
    ],
    'medium': [
        'Exercise caution before proceeding.',
        'Verify whether the QR code was placed in a publicly accessible or untrusted physical location.',
        'Refrain from downloading executable files or accepting profile installations.',
    ],
    'low': [
        'Destination presents low surface threat indicators.',
        'Always check that the address bar displays a valid SSL certificate before authenticating.',
    ],
}

# Small, documented list of globally recognized apex domains
# Used to calibrate borderline model noise ONLY when NO threat indicators are present
DOCUMENTED_TOP_DOMAINS = {
    'google.com', 'wikipedia.org', 'github.com', 'microsoft.com',
    'apple.com', 'amazon.com', 'python.org', 'w3.org', 'nist.gov',
    'who.int', 'mozilla.org', 'kernel.org', 'stackoverflow.com',
    'cloudflare.com', 'archive.org'
}


def assess_risk(
    features: Dict[str, Any],
    model_prob: Optional[float],
    model_label: Optional[int],
    model_available: bool = True,
    url_str: str = ""
) -> Dict[str, Any]:
    """
    Produce explainable risk assessment.
    Separates phishing probability from prediction confidence.
    Returns UNKNOWN when model is unavailable or evidence is ambiguous.
    """
    triggered: List[Dict[str, Any]] = []
    rule_score = 0

    for entry in RULES:
        feat_key, condition, label, severity = entry
        val = features.get(feat_key, 0)
        fired = condition(val) if callable(condition) else (val == condition)
        if fired:
            triggered.append({
                'label':    label,
                'severity': severity,
                'value':    val,
            })
            rule_score += SEVERITY_SCORE[severity]

    # Handle model unavailable / prediction failure
    if not model_available or model_label is None or model_prob is None:
        if rule_score >= 20:
            risk_level = 'HIGH'
            rec = 'DO NOT PROCEED — Suspicious indicators detected; ML model unavailable for deep analysis.'
        elif rule_score > 0:
            risk_level = 'MEDIUM'
            rec = 'PROCEED WITH CAUTION — Anomalies detected; ML model unavailable.'
        else:
            risk_level = 'UNKNOWN'
            rec = 'ANALYSIS UNAVAILABLE — Model not loaded. Do not assume destination is safe.'

        return {
            'risk_level':             risk_level,
            'risk_score':             rule_score,
            'indicators':             triggered,
            'consequences':           [CONSEQUENCE_MAP.get(i['label'], 'Uncertain threat potential.') for i in triggered] or ['Analysis incomplete.'],
            'safeguards':             SAFEGUARD_MAP['high'] if risk_level == 'HIGH' else SAFEGUARD_MAP['medium'],
            'recommendation':         rec,
            'model_prediction':       'Unknown',
            'phishing_probability':   None,
            'prediction_confidence':  None,
            'model_note':             'Machine learning model unavailable. Assessment based solely on static heuristic rules.',
            'trusted_domain_applied': False,
        }

    # Separate phishing probability and prediction confidence
    phishing_prob = float(model_prob)
    if model_label == 1:
        model_pred_str = 'Phishing'
        confidence = phishing_prob
    else:
        model_pred_str = 'Legitimate'
        confidence = 1.0 - phishing_prob

    # Model contribution to score
    model_score = int(phishing_prob * 45)
    total_score = rule_score + model_score

    # Check for documented trusted domain calibration
    ext = _tld_extractor(url_str)
    registered_domain = f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else ext.domain.lower()
    is_trusted_domain = registered_domain in DOCUMENTED_TOP_DOMAINS
    trusted_domain_applied = False

    # Risk level categorization
    if total_score >= 40 or (model_label == 1 and phishing_prob >= 0.80):
        risk_level = 'HIGH'
    elif total_score >= 18 or (model_label == 1 and phishing_prob >= 0.50):
        risk_level = 'MEDIUM'
    elif total_score >= 8 or phishing_prob >= 0.30:
        risk_level = 'LOW'
    else:
        risk_level = 'SAFE'

    # If domain is in documented top domains and NO rule-based threat indicators fired:
    if is_trusted_domain and rule_score == 0:
        if phishing_prob < 0.60:
            risk_level = 'SAFE' if phishing_prob < 0.15 else 'LOW'
            trusted_domain_applied = True

    # Safeguards & recommendation
    if risk_level == 'HIGH':
        safeguards = SAFEGUARD_MAP['high']
        recommendation = 'DO NOT PROCEED'
    elif risk_level == 'MEDIUM':
        safeguards = SAFEGUARD_MAP['medium']
        recommendation = 'PROCEED WITH CAUTION — verify independently'
    elif risk_level == 'LOW':
        safeguards = SAFEGUARD_MAP['low']
        recommendation = 'LOW CONCERN — verify source before submitting data'
    else:
        safeguards = ['URL shows no detected threat indicators. Still exercise standard awareness with physical QR codes.']
        recommendation = 'LIKELY SAFE'

    consequences = list({
        CONSEQUENCE_MAP[i['label']]
        for i in triggered if i['label'] in CONSEQUENCE_MAP
    })
    if not consequences:
        consequences = ['No specific threat patterns identified.']

    # Model note
    model_note = (
        'Model outputs are statistical estimates and do not guarantee safety. '
        'Never enter credentials or download files from untrusted sources.'
    )
    if trusted_domain_applied:
        model_note += f" (Registered domain '{registered_domain}' is a recognized public platform; safety is not guaranteed if account is compromised)."

    return {
        'risk_level':             risk_level,
        'risk_score':             total_score,
        'indicators':             triggered,
        'consequences':           consequences,
        'safeguards':             safeguards,
        'recommendation':         recommendation,
        'model_prediction':       model_pred_str,
        'phishing_probability':   round(phishing_prob, 4),
        'prediction_confidence':  round(confidence, 4),
        'model_note':             model_note,
        'trusted_domain_applied': trusted_domain_applied,
    }


def explain_url(url: str, features: Dict[str, Any], risk_report: Dict[str, Any]) -> str:
    """
    Generate transparent, evidence-based threat explanation.
    """
    parts = []
    pred = risk_report.get('model_prediction', 'Unknown')
    prob = risk_report.get('phishing_probability')
    conf = risk_report.get('prediction_confidence')

    if prob is not None and conf is not None:
        parts.append(
            f"The destination was classified as **{pred}** with "
            f"**{conf*100:.1f}% confidence** (calculated phishing probability: {prob*100:.1f}%)."
        )
    else:
        parts.append("ML classification was unavailable for this scan.")

    indicators = risk_report.get('indicators', [])
    if indicators:
        high = [i['label'] for i in indicators if i['severity'] in ('critical', 'high')]
        med  = [i['label'] for i in indicators if i['severity'] == 'medium']
        low  = [i['label'] for i in indicators if i['severity'] == 'low']

        if high:
            parts.append(f"**High-severity anomalies:** {'; '.join(high)}.")
        if med:
            parts.append(f"**Moderate indicators:** {'; '.join(med)}.")
        if low:
            parts.append(f"**Low-level indicators:** {'; '.join(low)}.")
    else:
        parts.append("No heuristic threat indicators were triggered.")

    if risk_report.get('trusted_domain_applied'):
        parts.append("Recognized platform domain confirmed.")

    parts.append(f"**Recommended action:** {risk_report.get('recommendation', 'Verify independently')}.")

    return " ".join(parts)
