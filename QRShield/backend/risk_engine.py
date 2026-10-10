"""
risk_engine.py
==============
Rule-based risk assessment and SHAP-lite explanation engine for QRShield.
Produces a structured threat report from URL features and model output.
"""

import re
from urllib.parse import urlparse
from typing import Dict, Any, List

# ── Severity weights ──────────────────────────────────────────────────────────
RULES = [
    # (key_in_features, threshold_or_exact, indicator_label, severity)
    # severity: 'critical', 'high', 'medium', 'low'
    ('is_ip',             1,    'IP address used instead of domain name',      'high'),
    ('suspicious_tld',    1,    'Suspicious top-level domain',                  'high'),
    ('num_at',            1,    '@ symbol in URL (potential URL spoofing)',      'high'),
    ('is_shortener',      1,    'URL shortener used (hides real destination)',   'high'),
    ('has_brand_keyword', 1,    'Impersonates a known brand',                   'high'),
    ('has_phishing_keyword', 1, 'Contains known phishing keyword',             'medium'),
    ('deep_subdomain',    1,    'Deeply nested subdomains (≥4 levels)',         'medium'),
    ('has_double_slash',  1,    'Double slash in URL path',                     'medium'),
    ('has_hex',           1,    'Hex-encoded characters in URL',                'medium'),
    ('has_port',          1,    'Non-standard port in URL',                     'medium'),
    ('num_hyphens', lambda v: v >= 3, '3+ hyphens in URL',                     'low'),
    ('url_len',     lambda v: v > 100, 'Unusually long URL (>100 chars)',       'low'),
    ('subdomain_count', lambda v: v >= 3, '3+ subdomains',                     'low'),
    ('query_param_count', lambda v: v >= 5, '5+ query parameters',             'low'),
    ('digit_ratio',  lambda v: v > 0.3, 'High digit-to-character ratio',       'low'),
    ('host_entropy', lambda v: v > 4.0, 'High hostname entropy (random-looking)', 'medium'),
    ('num_percent',  lambda v: v >= 3, '3+ percent-encoded characters',        'low'),
]

SEVERITY_SCORE = {'critical': 40, 'high': 20, 'medium': 8, 'low': 3}
CONSEQUENCE_MAP = {
    'Impersonates a known brand':                 'Credential theft and account takeover.',
    'IP address used instead of domain name':     'Likely a temporary phishing server.',
    'URL shortener used (hides real destination)':'Destination is hidden; may redirect to malware.',
    '@ symbol in URL (potential URL spoofing)':   'Browser may navigate to a different site than displayed.',
    'Suspicious top-level domain':                'Domain registered on a free or abused TLD, common in phishing campaigns.',
    'Contains known phishing keyword':            'May solicit sensitive personal or financial data.',
    'Deeply nested subdomains (≥4 levels)':       'Designed to mimic legitimate domain structure.',
    'High hostname entropy (random-looking)':     'Domain may be algorithmically generated.',
    'Unusually long URL (>100 chars)':            'Long URLs are used to bury the actual destination.',
}

SAFEGUARD_MAP = {
    'critical': [
        'Do NOT proceed to this URL.',
        'Report the QR code to your IT/security team.',
        'If you have already visited the URL, change passwords immediately and check for malware.',
    ],
    'high': [
        'Verify the URL with a trusted source before proceeding.',
        'Do not enter any credentials, payment or personal information.',
        'Use VirusTotal or Google Safe Browsing to manually check the URL.',
    ],
    'medium': [
        'Exercise caution — one or more suspicious indicators detected.',
        'Confirm the sender or source of the QR code before scanning.',
        'Avoid downloading files from this URL.',
    ],
    'low': [
        'URL appears mostly safe but verify the source before proceeding.',
        'Check the full URL carefully for typos or look-alike characters.',
    ],
}


def assess_risk(features: Dict[str, Any], model_prob: float, model_label: int) -> Dict:
    """
    Combine rule-based indicators with model output to produce a risk report.

    Parameters
    ----------
    features   : output of extract_url_features()
    model_prob : probability of phishing (0..1) from trained model
    model_label: model prediction (0=legit, 1=phishing)

    Returns
    -------
    dict with keys: risk_level, risk_score, indicators, consequences,
                    safeguards, recommendation, model_confidence
    """
    triggered: List[Dict] = []
    rule_score = 0

    for entry in RULES:
        feat_key, condition, label, severity = entry
        val = features.get(feat_key, 0)
        if feat_key == 'has_brand_keyword':
            # Only trigger brand impersonation if accompanied by another anomaly
            # (e.g. hyphens, deep subdomains, suspicious TLD, phishing keywords, or model prediction 1)
            # This avoids falsely flagging official domains like google.com or paypal.com
            is_suspicious_brand = (
                val == 1 and (
                    features.get('suspicious_tld', 0) == 1 or
                    features.get('num_hyphens', 0) > 0 or
                    features.get('subdomain_count', 0) > 1 or
                    features.get('has_phishing_keyword', 0) == 1 or
                    features.get('is_shortener', 0) == 1 or
                    model_label == 1
                )
            )
            fired = is_suspicious_brand
        elif callable(condition):
            fired = condition(val)
        else:
            fired = (val == condition)
        if fired:
            triggered.append({
                'label':    label,
                'severity': severity,
                'value':    val,
            })
            rule_score += SEVERITY_SCORE[severity]

    # Model contribution
    model_score = 0
    if model_label == 1:
        model_score = int(model_prob * 40)   # up to 40 extra points
    elif model_prob > 0.4:
        model_score = int((model_prob - 0.4) * 20)

    total_score = rule_score + model_score

    # ── Risk level ────────────────────────────────────────────────────────────
    if total_score >= 40 or (model_label == 1 and model_prob >= 0.85):
        risk_level = 'HIGH'
    elif total_score >= 20 or (model_label == 1 and model_prob >= 0.55):
        risk_level = 'MEDIUM'
    elif total_score >= 8 or model_prob >= 0.35:
        risk_level = 'LOW'
    else:
        risk_level = 'SAFE'

    # If model says phishing with high confidence, floor at MEDIUM
    if model_label == 1 and model_prob >= 0.60 and risk_level == 'SAFE':
        risk_level = 'MEDIUM'

    # ── Consequences ─────────────────────────────────────────────────────────
    consequences = list({CONSEQUENCE_MAP[i['label']]
                         for i in triggered if i['label'] in CONSEQUENCE_MAP})

    # ── Safeguards ───────────────────────────────────────────────────────────
    if risk_level == 'HIGH':
        safeguards = SAFEGUARD_MAP['high']
    elif risk_level == 'MEDIUM':
        safeguards = SAFEGUARD_MAP['medium']
    elif risk_level == 'LOW':
        safeguards = SAFEGUARD_MAP['low']
    else:
        safeguards = ['URL appears safe. Still verify the source of the QR code before proceeding.']

    # ── Recommendation ───────────────────────────────────────────────────────
    if risk_level == 'HIGH':
        recommendation = 'DO NOT PROCEED'
    elif risk_level == 'MEDIUM':
        recommendation = 'PROCEED WITH CAUTION — verify independently'
    elif risk_level == 'LOW':
        recommendation = 'LOW CONCERN — still verify source'
    else:
        recommendation = 'LIKELY SAFE — verify source of QR code'

    return {
        'risk_level':        risk_level,
        'risk_score':        total_score,
        'indicators':        triggered,
        'consequences':      consequences if consequences else ['No specific threat patterns identified.'],
        'safeguards':        safeguards,
        'recommendation':    recommendation,
        'model_prediction':  'Phishing' if model_label == 1 else 'Legitimate',
        'model_confidence':  round(float(model_prob), 4),
        'model_note': (
            'Model probability is an estimate and does not guarantee safety. '
            'Always verify the source of any QR code.'
        ),
    }


def explain_url(url: str, features: Dict, risk_report: Dict) -> str:
    """
    Generate a plain-English explanation paragraph for the threat report.
    Uses only verified indicators — no hallucinated content.
    """
    parts = []
    risk = risk_report['risk_level']
    indicators = risk_report['indicators']

    parts.append(f"This URL was classified as **{risk_report['model_prediction']}** "
                 f"with {risk_report['model_confidence']*100:.1f}% model confidence.")

    if indicators:
        high_sev = [i['label'] for i in indicators if i['severity'] in ('critical','high')]
        med_sev  = [i['label'] for i in indicators if i['severity'] == 'medium']
        low_sev  = [i['label'] for i in indicators if i['severity'] == 'low']

        if high_sev:
            parts.append(f"**High-severity indicators detected:** {'; '.join(high_sev)}.")
        if med_sev:
            parts.append(f"**Medium-severity indicators:** {'; '.join(med_sev)}.")
        if low_sev:
            parts.append(f"**Low-severity indicators:** {'; '.join(low_sev)}.")
    else:
        parts.append("No specific rule-based threat indicators were detected.")

    if risk_report['consequences'] and risk_report['consequences'][0] != 'No specific threat patterns identified.':
        parts.append(f"**Potential consequences:** {' '.join(risk_report['consequences'])}")

    parts.append(f"**Recommendation:** {risk_report['recommendation']}.")

    return ' '.join(parts)

