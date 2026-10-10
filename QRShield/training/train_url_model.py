"""
train_url_model.py
==================
Stage 2 — Download / prepare a labelled URL dataset and train:
  - Model A: Extra Trees Classifier
  - Model B: Random Forest Classifier

All results are saved to ../results/ and models to ../models/.

Run from the QRShield/training/ directory:
    python train_url_model.py
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(THIS_DIR)
DATA_DIR   = os.path.join(ROOT, 'datasets')
MODEL_DIR  = os.path.join(ROOT, 'models')
RESULT_DIR = os.path.join(ROOT, 'results')
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.insert(0, THIS_DIR)
from url_features import FEATURE_NAMES, features_to_vector

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Acquisition
# ─────────────────────────────────────────────────────────────────────────────
def load_or_download_dataset() -> pd.DataFrame:
    """
    Try to load a cached CSV first.  If not present, download the PhiUSIIL
    dataset from UCI ML Repository (or a mirror Kaggle CSV).
    Falls back to generating a balanced synthetic-features dataset from a
    published phishing URL list if network is unavailable.
    """
    cached = os.path.join(DATA_DIR, 'url_dataset.csv')
    if os.path.exists(cached):
        print(f"[INFO] Loading cached dataset from {cached}")
        df = pd.read_csv(cached)
        # Validate expected columns
        if 'url' in df.columns and 'label' in df.columns:
            return df
        print("[WARN] Cached file missing expected columns — re-downloading.")

    # ── Try downloading PhiUSIIL from UCI ────────────────────────────────────
    try:
        import urllib.request, zipfile, io
        print("[INFO] Attempting to download PhiUSIIL phishing URL dataset …")
        url = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        z = zipfile.ZipFile(io.BytesIO(data))
        csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if csv_names:
            raw = pd.read_csv(z.open(csv_names[0]))
            print(f"[INFO] Downloaded: {len(raw)} rows, columns: {list(raw.columns[:6])} …")
            # Identify URL and label columns heuristically
            url_col = next((c for c in raw.columns if 'url' in c.lower()), None)
            label_col = next((c for c in raw.columns if 'label' in c.lower() or 'class' in c.lower()), None)
            if url_col and label_col:
                df = raw[[url_col, label_col]].rename(columns={url_col: 'url', label_col: 'label'})
                # Normalize label: phishing=1, legit=0
                vals = df['label'].unique()
                if set(vals) == {-1, 1} or set(vals) == {'phishing', 'legitimate'}:
                    df['label'] = df['label'].map(lambda v: 1 if v in (1, 'phishing') else 0)
                df = df.dropna()
                df.to_csv(cached, index=False)
                print(f"[INFO] Saved to {cached}")
                return df
    except Exception as e:
        print(f"[WARN] UCI download failed: {e}")

    # ── Try Kaggle-style URL CSV files already present ────────────────────────
    for fname in ['phishing_site_urls.csv', 'phishing_urls.csv', 'urls.csv',
                  'malicious_phish.csv', 'url_data.csv']:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            print(f"[INFO] Found local file: {fpath}")
            raw = pd.read_csv(fpath)
            url_col   = next((c for c in raw.columns if 'url' in c.lower()), None)
            label_col = next((c for c in raw.columns if c.lower() in
                              ('label', 'class', 'type', 'status', 'result')), None)
            if url_col and label_col:
                df = raw[[url_col, label_col]].rename(columns={url_col: 'url', label_col: 'label'})
                df['label'] = df['label'].apply(
                    lambda v: 1 if str(v).lower() in ('1', 'phishing', 'bad', 'malicious') else 0)
                df = df.dropna()
                df.to_csv(cached, index=False)
                return df

    # ── Fallback: built-in URL list ──────────────────────────────────────────
    print("[WARN] No external dataset available. Using built-in URL examples.")
    return _builtin_url_dataset(cached)


def _builtin_url_dataset(cached: str) -> pd.DataFrame:
    """
    A small but representative labeled URL dataset built-in as fallback.
    Contains real URL patterns (no live requests needed) sourced from
    publicly documented phishing examples.
    """
    legit = [
        "https://www.google.com",
        "https://www.wikipedia.org/wiki/Python",
        "https://github.com/scikit-learn/scikit-learn",
        "https://stackoverflow.com/questions/tagged/python",
        "https://www.amazon.com/dp/B09G9FPHY6",
        "https://www.bbc.co.uk/news",
        "https://docs.python.org/3/library/urllib.html",
        "https://www.reddit.com/r/MachineLearning/",
        "https://www.linkedin.com/in/username",
        "https://www.microsoft.com/en-us/",
        "https://www.apple.com/iphone/",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.nytimes.com/section/technology",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://www.cloudflare.com/learning/security/",
        "https://www.shopify.com/free-trial",
        "https://www.coursera.org/learn/machine-learning",
        "https://www.nasa.gov/topics/technology/",
        "https://www.who.int/health-topics/",
        "https://www.harvard.edu/academics/",
        "https://www.python.org/downloads/",
        "https://www.mozilla.org/en-US/firefox/",
        "https://www.docker.com/get-started",
        "https://kubernetes.io/docs/home/",
        "https://www.postgresql.org/docs/",
        "https://aws.amazon.com/s3/",
        "https://azure.microsoft.com/en-us/",
        "https://cloud.google.com/products",
        "https://www.stripe.com/docs",
        "https://www.twilio.com/docs",
        "https://openai.com/research",
        "https://www.kaggle.com/datasets",
        "https://huggingface.co/models",
        "https://www.tensorflow.org/tutorials",
        "https://pytorch.org/tutorials/",
        "https://www.w3schools.com/python/",
        "https://developer.mozilla.org/en-US/docs/",
        "https://flask.palletsprojects.com/",
        "https://fastapi.tiangolo.com/tutorial/",
        "https://www.nginx.com/resources/",
        "https://www.elastic.co/guide/",
        "https://www.databricks.com/solutions",
        "https://www.snowflake.com/en/data-cloud/",
        "https://www.tableau.com/learn/",
        "https://www.salesforce.com/crm/",
        "https://zoom.us/pricing",
        "https://www.slack.com/intl/en-in/",
        "https://www.notion.so/product",
        "https://trello.com/en/tour",
        "https://www.figma.com/design/",
        "https://www.canva.com/templates/",
        "https://unsplash.com/images/nature",
        "https://www.pexels.com/videos/",
        "https://www.imdb.com/chart/top",
        "https://www.rottentomatoes.com/",
        "https://www.tripadvisor.com/Hotels",
        "https://www.booking.com/hotels/",
        "https://www.airbnb.com/rooms/",
        "https://www.expedia.com/Hotels",
        "https://www.yelp.com/search?find_desc=restaurants",
        "https://www.grubhub.com/delivery/",
        "https://www.doordash.com/food-delivery/",
        "https://www.uber.com/us/en/ride/",
        "https://www.lyft.com/rider/",
        "https://www.spotify.com/us/",
        "https://www.netflix.com/browse",
        "https://www.hulu.com/movies",
        "https://www.disneyplus.com/home",
        "https://www.hbomax.com/",
        "https://www.twitch.tv/directory",
        "https://www.tiktok.com/trending",
        "https://www.instagram.com/explore/",
        "https://www.facebook.com/marketplace/",
        "https://twitter.com/explore",
        "https://www.pinterest.com/ideas/",
        "https://medium.com/tag/machine-learning",
        "https://www.quora.com/topic/Artificial-Intelligence",
        "https://news.ycombinator.com/",
        "https://www.producthunt.com/",
        "https://techcrunch.com/category/artificial-intelligence/",
        "https://www.wired.com/category/security/",
        "https://arstechnica.com/security/",
        "https://krebsonsecurity.com/",
        "https://www.schneier.com/",
        "https://owasp.org/www-project-top-ten/",
        "https://nvd.nist.gov/vuln/search",
        "https://www.cve.org/",
        "https://www.virustotal.com/gui/home/upload",
        "https://www.shodan.io/",
        "https://www.exploit-db.com/",
        "https://www.metasploit.com/",
        "https://www.kali.org/tools/",
        "https://nmap.org/docs.html",
        "https://www.wireshark.org/docs/",
        "https://www.zaproxy.org/docs/",
        "https://burpsuite.portswigger.net/",
        "https://www.offensive-security.com/",
        "https://www.sans.org/courses/",
        "https://www.coursera.org/specializations/cybersecurity",
    ]

    phishing = [
        "http://paypa1.com/secure/login?redirect=account",
        "http://192.168.1.1/admin/login.php",
        "https://amazon-security-alert.xyz/verify/account",
        "http://www.googIe.com/login/secure",
        "http://bit.ly/3xFreeGift",
        "https://microsoft-update-required.tk/download",
        "http://secure-banking-update.ml/verify",
        "https://apple-id-suspended.gq/restore/account",
        "http://www.paypal-secure-account.com/login",
        "https://facebook-login-suspended.ga/recover",
        "http://free-bitcoin-generator.top/claim",
        "http://account-verify.net/paypal/login",
        "https://secure-server-update.work/authenticate",
        "http://prize-winner-2024.club/claim-reward",
        "https://netflix-account-suspended.loan/update",
        "http://amazon-prime-free.download/activate",
        "https://bank-alert-security.bid/login",
        "http://win-iphone-now.party/submit",
        "https://government-stimulus-payment.win/apply",
        "http://crypto-airdrop-free.review/claim",
        "http://www.ebay-account-verify.com/secure/update",
        "https://instagram-login-confirm.cf/account",
        "http://apple-refund-pending.xyz/process",
        "http://10.0.0.1/router/admin.php",
        "https://fake-antivirus-warning.trade/remove",
        "http://secure.paypal.com.phishing-site.com/login",
        "http://www.microsoft.com-update-now.xyz/download",
        "http://amazon.login.secure.account.verify.bad.tk",
        "https://coinbase-wallet-restore.ml/restore",
        "http://binance-airdrop-2024.ga/claim-now",
        "http://free-gift-card-generator.gq/amazon",
        "https://limited-offer-exclusive.xyz/deal?ref=spam",
        "http://urgent-account-suspended.top/verify-now",
        "https://claim-your-prize-today.club/winner?id=123",
        "http://verify-your-identity-now.loan/submit",
        "https://download-update-required.work/install",
        "http://exclusive-discount-offer.party/70percent",
        "https://suspicious-activity-alert.bid/confirm",
        "http://lottery-jackpot-winner.win/collect",
        "https://covid-relief-fund.review/apply?id=victim",
        "http://login.paypal.com@phishing.tk/account",
        "http://www.amazon.com.verify-account.ml/update",
        "http://apple.com-id-suspended.gq/restore",
        "http://secure.microsoft.com.badsite.xyz/login",
        "http://www.bankofamerica.com-secure.loan/login",
        "http://wellsfargo-alert.download/verify",
        "http://citibank-suspicious-login.trade/confirm",
        "http://chase-account-verify.bid/login",
        "http://www.irs-refund-pending.win/apply",
        "http://www.usps-delivery-failed.xyz/schedule",
        "http://fedex-package-held.ml/reschedule?tracking=fake",
        "http://dhl-delivery-notification.ga/confirm",
        "http://ups-tracking-update.cf/verify",
        "http://steam-gift-card-free.gq/redeem",
        "http://roblox-free-robux.top/generate",
        "http://minecraft-free-hack.club/download",
        "http://fortnite-free-vbucks.xyz/generate",
        "http://adobe-flash-update.work/install",
        "http://java-update-required.party/download",
        "http://antivirus-scan-result.loan/remove-virus",
        "http://your-computer-infected.download/fix-now",
        "http://windows-defender-alert.bid/scan",
        "http://tech-support-required.win/call-now?phone=fake",
        "https://cryptocurrency-investment.review/profit-500pct",
        "http://binary-options-profit.trade/invest?ref=fraud",
        "http://forex-guaranteed-profit.xyz/join",
        "http://nft-airdrop-free.ml/claim-whitelist",
        "https://metamask-wallet-recovery.ga/restore",
        "http://opensea-verification-required.cf/verify",
        "http://uniswap-liquidity-reward.gq/claim",
        "http://phishing-domain.net/paypal/en_US/login/?signin",
        "http://secure-update-required.com/microsoft/windows",
        "http://login-confirm-required.info/facebook/account",
        "http://verify-now.biz/google/gmail/signin",
        "http://account-suspended-action-required.co/apple",
        "http://update-billing-information.org/netflix",
        "http://free-premium-account.ru/spotify",
        "http://hack-wifi-password.cn/tool/download",
        "http://free-movie-download.cc/hd-movies",
        "http://pirated-software-free.co/adobe-photoshop",
        "http://crack-license-key.xyz/microsoft-office",
        "http://malware-dropper.pw/payload.exe",
        "http://ransomware-delivery.io/encrypted-files",
        "http://credential-harvester.site/phish",
        "http://keylogger-download.space/windows",
        "http://spyware-installer.host/android-update",
        "http://adware-bundle.website/free-software",
        "http://trojan-dropper.online/update.msi",
        "http://botnet-cnc.pw/checkin?bot=new",
        "http://data-exfiltration.io/upload",
        "http://sql-injection-test.site/login?id=1'OR'1'='1",
        "http://xss-attack.space/search?q=<script>alert(1)</script>",
        "http://clickjacking-frame.website/secure-payment",
        "http://man-in-middle.online/bank/login",
        "http://session-hijack.host/cookie-steal",
        "http://ip-logger.site/track?redirect=phish",
        "http://geo-tracker.pw/location?target=victim",
        "http://fake-captcha-human.space/verify-human",
        "http://social-engineering.online/pretexting",
        "http://vishing-script.host/call-center",
    ]

    rows = (
        [{'url': u, 'label': 0} for u in legit] +
        [{'url': u, 'label': 1} for u in phishing]
    )
    df = pd.DataFrame(rows)

    # Augment with simple string variants to increase dataset size to ~600
    augmented = []
    for _, row in df.iterrows():
        u, l = row['url'], row['label']
        augmented.append({'url': u, 'label': l})
        # Create slightly varied copies
        if l == 0 and len(augmented) % 3 == 0:
            augmented.append({'url': u + '/en', 'label': 0})
        if l == 1 and len(augmented) % 2 == 0:
            augmented.append({'url': u + '?session=abc123', 'label': 1})

    df = pd.DataFrame(augmented).drop_duplicates(subset='url')
    df.to_csv(cached, index=False)
    print(f"[INFO] Built-in dataset saved: {len(df)} URLs to {cached}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame):
    print("[INFO] Extracting URL features …")
    rows = []
    failed = 0
    for url in df['url']:
        try:
            rows.append(features_to_vector(str(url)))
        except Exception:
            rows.append([0] * len(FEATURE_NAMES))
            failed += 1
    if failed:
        print(f"[WARN] {failed} URLs failed feature extraction (zeroed).")
    X = np.array(rows, dtype=np.float32)
    y = df['label'].values.astype(int)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(name, clf, X_test, y_test, X_train, y_train):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

    metrics = {
        'model': name,
        'accuracy':  round(float(accuracy_score(y_test, y_pred)),  4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred,    zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_pred,        zero_division=0)), 4),
    }
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        metrics['roc_auc'] = round(float(auc(fpr, tpr)), 4)
        metrics['_roc_fpr'] = fpr.tolist()
        metrics['_roc_tpr'] = tpr.tolist()

    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'],
                                   output_dict=True)
    metrics['classification_report'] = report

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    for k in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
        if k in metrics:
            print(f"  {k:20s}: {metrics[k]:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}")

    return metrics


def plot_confusion_matrix(cm, name, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_roc(metrics_list, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in metrics_list:
        if '_roc_fpr' in m:
            ax.plot(m['_roc_fpr'], m['_roc_tpr'],
                    label=f"{m['model']} (AUC={m.get('roc_auc','?'):.4f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_feature_importance(clf, name, path):
    if not hasattr(clf, 'feature_importances_'):
        return
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(idx)), importances[idx], color='steelblue')
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Importance')
    ax.set_title(f'Top 15 Feature Importances — {name}')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_model_comparison(metrics_list, path):
    names  = [m['model'] for m in metrics_list]
    keys   = ['accuracy', 'precision', 'recall', 'f1']
    x      = np.arange(len(keys))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics_list):
        vals = [m.get(k, 0) for k in keys]
        ax.bar(x + i * width, vals, width, label=m['model'])
    ax.set_xticks(x + width * (len(metrics_list) - 1) / 2)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  QRShield — URL Model Training")
    print("="*60)

    # Load dataset
    df = load_or_download_dataset()
    df = df.dropna(subset=['url', 'label'])
    df['label'] = df['label'].astype(int)
    df = df.drop_duplicates(subset='url').reset_index(drop=True)

    counts = df['label'].value_counts()
    print(f"\n[INFO] Dataset: {len(df)} URLs")
    print(f"  Legitimate: {counts.get(0, 0)}")
    print(f"  Phishing:   {counts.get(1, 0)}")
    print(f"  Class balance: {counts.get(1,0)/len(df)*100:.1f}% phishing\n")

    # Feature extraction
    X, y = build_feature_matrix(df)
    print(f"[INFO] Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # Train / val / test split (60/20/20)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv)

    print(f"[INFO] Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # Models
    et_clf = ExtraTreesClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1)
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1)

    et_metrics = evaluate('Extra Trees',   et_clf, X_test, y_test, X_train, y_train)
    rf_metrics = evaluate('Random Forest', rf_clf, X_test, y_test, X_train, y_train)

    all_metrics = [et_metrics, rf_metrics]

    # ── Select best model by phishing recall then F1 ──────────────────────────
    def model_score(m):
        return (m['recall'], m['f1'], m['accuracy'])

    best_m  = max(all_metrics, key=model_score)
    best_clf = et_clf if best_m['model'] == 'Extra Trees' else rf_clf
    print(f"\n[INFO] Selected model: {best_m['model']} "
          f"(recall={best_m['recall']:.4f}, f1={best_m['f1']:.4f})")

    # ── Save models ───────────────────────────────────────────────────────────
    joblib.dump(best_clf,  os.path.join(MODEL_DIR, 'url_model.pkl'))
    joblib.dump(et_clf,    os.path.join(MODEL_DIR, 'extra_trees_url.pkl'))
    joblib.dump(rf_clf,    os.path.join(MODEL_DIR, 'random_forest_url.pkl'))
    joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, 'feature_names.pkl'))
    print(f"[INFO] Models saved to {MODEL_DIR}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    def clean(m):
        return {k: v for k, v in m.items() if not k.startswith('_')}

    summary = {
        'dataset': {
            'total': int(len(df)),
            'legitimate': int(counts.get(0, 0)),
            'phishing':   int(counts.get(1, 0)),
            'features':   int(X.shape[1]),
        },
        'split': {
            'train': int(len(y_train)),
            'val':   int(len(y_val)),
            'test':  int(len(y_test)),
        },
        'models': [clean(et_metrics), clean(rf_metrics)],
        'selected_model': best_m['model'],
        'feature_names': FEATURE_NAMES,
    }
    metrics_path = os.path.join(RESULT_DIR, 'url_model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Metrics saved to {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        et_metrics['confusion_matrix'], 'Extra Trees',
        os.path.join(RESULT_DIR, 'cm_extra_trees.png'))
    plot_confusion_matrix(
        rf_metrics['confusion_matrix'], 'Random Forest',
        os.path.join(RESULT_DIR, 'cm_random_forest.png'))
    plot_roc(all_metrics, os.path.join(RESULT_DIR, 'roc_comparison.png'))
    plot_feature_importance(et_clf, 'Extra Trees',
                            os.path.join(RESULT_DIR, 'feature_importance_et.png'))
    plot_feature_importance(rf_clf, 'Random Forest',
                            os.path.join(RESULT_DIR, 'feature_importance_rf.png'))
    plot_model_comparison(all_metrics, os.path.join(RESULT_DIR, 'model_comparison.png'))

    print(f"\n[DONE] Training complete. Selected: {best_m['model']}")
    print(f"       All results in: {RESULT_DIR}")
    print(f"       Models in:      {MODEL_DIR}")


if __name__ == '__main__':
    main()

