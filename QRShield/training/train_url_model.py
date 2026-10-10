"""
train_url_model.py
==================
QRShield — Explainable Pre-Navigation QR Code Phishing Detection
Stage 2 — URL Model Training with Validation-Guided Model Selection.

Methodology:
1. Loads and validates dataset labels (0 = Legitimate, 1 = Phishing).
2. De-biases synthetic artifacts (apex domains, paths, protocols).
3. 60% Train / 20% Validation / 20% Untouched Test split.
4. Trains Extra Trees and Random Forest on Train set.
5. Evaluates and compares both models on Validation set; selects superior model.
6. Retrains on Train + Validation combined.
7. Evaluates once on the untouched Test set and saves genuine metrics/plots.
"""

import os
import sys
import json
import warnings
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

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
from url_features import FEATURE_NAMES, features_to_vector, normalize_url, _extractor


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Acquisition & Strict Label Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_and_normalize_labels(df: pd.DataFrame, source_name: str = "dataset") -> pd.DataFrame:
    """
    Ensure 0 = Legitimate, 1 = Phishing.
    Validates ground truth against known anchor domains (edu/gov vs numeric IP attacks).
    """
    df = df.dropna(subset=['url', 'label']).copy()
    df['label'] = df['label'].astype(int)

    # Check anchor domains to verify label mapping
    edu_sample = df[df['url'].str.contains(r'\.edu|\.gov', case=False, na=False)]
    ip_sample = df[df['url'].str.contains(r'http://\d+\.\d+\.\d+\.\d+', case=False, na=False)]

    if len(edu_sample) > 10 and len(ip_sample) > 10:
        edu_label = edu_sample['label'].mode()[0]
        ip_label = ip_sample['label'].mode()[0]

        if edu_label == 1 and ip_label == 0:
            print(f"[INFO] Inverting {source_name} labels (detected 1=legit, 0=phish -> normalized to 0=legit, 1=phish).")
            df['label'] = 1 - df['label']
        elif edu_label == 0 and ip_label == 1:
            print(f"[INFO] Confirmed {source_name} label mapping: 0=Legitimate, 1=Phishing.")
        else:
            print(f"[WARN] Ambiguous anchor labels in {source_name}: edu={edu_label}, ip={ip_label}")

    return df


def load_or_download_dataset() -> pd.DataFrame:
    cached = os.path.join(DATA_DIR, 'url_dataset.csv')
    if os.path.exists(cached):
        print(f"[INFO] Loading cached dataset from {cached} …")
        df = pd.read_csv(cached)
        if 'url' in df.columns and 'label' in df.columns:
            return validate_and_normalize_labels(df, "cached dataset")

    # Download from UCI if not present
    try:
        import urllib.request, zipfile, io
        print("[INFO] Attempting to download PhiUSIIL phishing URL dataset from UCI ML Repository …")
        url = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        z = zipfile.ZipFile(io.BytesIO(data))
        csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if csv_names:
            raw = pd.read_csv(z.open(csv_names[0]))
            url_col = next((c for c in raw.columns if 'url' in c.lower()), None)
            label_col = next((c for c in raw.columns if 'label' in c.lower()), None)
            if url_col and label_col:
                df = raw[[url_col, label_col]].rename(columns={url_col: 'url', label_col: 'label'})
                df = validate_and_normalize_labels(df, "UCI download")
                df.to_csv(cached, index=False)
                print(f"[INFO] Saved normalized dataset to {cached}")
                return df
    except Exception as e:
        print(f"[WARN] UCI download unavailable: {e}")

    # Fallback to demonstration set
    print("[WARN] Using built-in demonstration URL dataset (Marked as DEMO ONLY).")
    return _builtin_demo_dataset(cached)


def _builtin_demo_dataset(cached: str) -> pd.DataFrame:
    """Fallback demonstration dataset."""
    demo_urls = [
        # Legitimate
        ("https://www.google.com", 0), ("https://google.com", 0),
        ("https://github.com", 0), ("https://github.com/torvalds/linux", 0),
        ("https://www.wikipedia.org", 0), ("https://en.wikipedia.org/wiki/Phishing", 0),
        ("https://www.amazon.com", 0), ("https://www.microsoft.com", 0),
        ("https://www.apple.com", 0), ("https://www.python.org", 0),
        ("https://stackoverflow.com", 0), ("https://www.bbc.co.uk", 0),
        ("https://www.nytimes.com", 0), ("https://docs.python.org/3/", 0),
        ("https://fastapi.tiangolo.com", 0), ("https://scikit-learn.org", 0),
        # Phishing
        ("http://paypal-security-update.xyz/login", 1),
        ("http://192.168.1.1/admin/login.php", 1),
        ("http://google.com.phish-account.top/verify", 1),
        ("http://bit.ly/free-bitcoin-airdrop-claim", 1),
        ("http://apple-id-suspended.gq/restore", 1),
        ("http://microsoft-update-required.tk/download.exe", 1),
        ("http://chase-bank-verify.bid/login", 1),
        ("http://amazon-gift-card-winner.club/claim", 1),
        ("http://free-vbucks-generator.party/enter", 1),
        ("http://netflix-billing-alert.loan/update", 1),
    ]
    df = pd.DataFrame(demo_urls, columns=['url', 'label'])
    df.to_csv(cached, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. De-biasing & Representative Data Preparation
# ─────────────────────────────────────────────────────────────────────────────
def debias_and_balance_dataset(df: pd.DataFrame, max_samples_per_class: int = 35000) -> pd.DataFrame:
    """
    Remove synthetic formatting shortcuts:
    - PhiUSIIL legitimate URLs are all 'https://www.domain.tld'.
    - Real benign web traffic contains apex domains (no www), paths, and http.
    - We diversify the legitimate URLs to represent modern internet architecture.
    """
    df = df.drop_duplicates(subset=['url']).dropna().copy()
    legit = df[df['label'] == 0].copy()
    phish = df[df['label'] == 1].copy()

    print(f"[INFO] Raw dataset: {len(legit)} legitimate, {len(phish)} phishing.")

    # Subsample if necessary for fast, balanced, high-quality training
    if len(legit) > max_samples_per_class:
        legit = legit.sample(n=max_samples_per_class, random_state=42).copy()
    if len(phish) > max_samples_per_class:
        phish = phish.sample(n=max_samples_per_class, random_state=42).copy()

    # De-bias legitimate URLs:
    # 50% keep https://www.domain.tld
    # 40% convert to apex https://domain.tld
    # 10% convert to http://domain.tld or add benign paths
    diversified_legit = []
    sample_paths = ['/about', '/contact', '/login', '/search', '/index.html', '/products', '/docs', '/help']

    for i, u in enumerate(legit['url']):
        if u.startswith('https://www.'):
            mod = i % 10
            apex = 'https://' + u[len('https://www.'):]
            if mod in (1, 2, 3, 4):
                diversified_legit.append(apex)
            elif mod == 5:
                diversified_legit.append(apex + sample_paths[i % len(sample_paths)])
            elif mod == 6:
                diversified_legit.append(u + sample_paths[i % len(sample_paths)])
            elif mod == 7:
                diversified_legit.append('http://' + u[len('https://'):])
            else:
                diversified_legit.append(u)
        else:
            diversified_legit.append(u)

    # Add key high-traffic anchor legitimate domains to ensure coverage
    anchor_legit = [
        "https://www.google.com", "https://google.com", "https://google.com/search?q=cybersecurity",
        "https://github.com", "https://github.com/scikit-learn/scikit-learn", "https://github.com/explore",
        "https://www.wikipedia.org", "https://en.wikipedia.org/wiki/Computer_security",
        "https://www.amazon.com", "https://www.microsoft.com", "https://www.apple.com",
        "https://www.python.org", "https://docs.python.org/3/", "https://stackoverflow.com",
        "https://www.bbc.com", "https://www.nytimes.com", "https://www.cloudflare.com",
        "https://www.mit.edu", "https://www.harvard.edu", "https://www.nih.gov"
    ]
    diversified_legit.extend(anchor_legit)

    balanced_df = pd.concat([
        pd.DataFrame({'url': diversified_legit, 'label': 0}),
        phish[['url', 'label']]
    ], ignore_index=True)

    balanced_df = balanced_df.drop_duplicates(subset=['url']).reset_index(drop=True)
    counts = balanced_df['label'].value_counts()
    print(f"[INFO] De-biased & balanced dataset: {len(balanced_df)} URLs")
    print(f"  Legitimate (0): {counts.get(0, 0)}")
    print(f"  Phishing   (1): {counts.get(1, 0)}")

    return balanced_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame):
    print("[INFO] Extracting static URL features for all samples …")
    features = []
    for u in df['url']:
        features.append(features_to_vector(str(u)))
    X = np.array(features, dtype=np.float32)
    y = df['label'].values.astype(int)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(name: str, clf, X_data, y_data, split_name: str = "Validation"):
    y_pred = clf.predict(X_data)
    y_prob = clf.predict_proba(X_data)[:, 1] if hasattr(clf, 'predict_proba') else None

    acc  = accuracy_score(y_data, y_pred)
    prec = precision_score(y_data, y_pred, zero_division=0)
    rec  = recall_score(y_data, y_pred, zero_division=0)
    f1   = f1_score(y_data, y_pred, zero_division=0)

    cm = confusion_matrix(y_data, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(fp + tn, 1)

    metrics = {
        'model':      name,
        'split':      split_name,
        'accuracy':   round(float(acc), 4),
        'precision':  round(float(prec), 4),
        'recall':     round(float(rec), 4),
        'f1':         round(float(f1), 4),
        'fpr':        round(float(fpr), 4),
        'confusion_matrix': cm.tolist(),
    }

    if y_prob is not None:
        fpr_arr, tpr_arr, _ = roc_curve(y_data, y_prob)
        metrics['roc_auc'] = round(float(auc(fpr_arr, tpr_arr)), 4)
        metrics['_roc_fpr'] = fpr_arr.tolist()
        metrics['_roc_tpr'] = tpr_arr.tolist()

    report = classification_report(y_data, y_pred, target_names=['Legitimate', 'Phishing'], output_dict=True)
    metrics['classification_report'] = report

    print(f"\n--- {name} ({split_name} Set) ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  FPR:       {metrics['fpr']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plotting Functions
# ─────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix_plot(cm, name, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'], ax=ax)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_roc_plot(metrics_list, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in metrics_list:
        if '_roc_fpr' in m:
            ax.plot(m['_roc_fpr'], m['_roc_tpr'],
                    label=f"{m['model']} (AUC = {m.get('roc_auc','?'):.4f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Chance (AUC = 0.5000)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Test Set')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance_plot(clf, name, path):
    if not hasattr(clf, 'feature_importances_'):
        return
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(idx)), importances[idx], color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Gini Importance')
    ax.set_title(f'Top 15 Feature Importances — {name}')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_comparison_plot(metrics_list, path):
    names = [m['model'] for m in metrics_list]
    keys = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics_list):
        vals = [m.get(k, 0) for k in keys]
        ax.bar(x + i * width, vals, width, label=m['model'])
    ax.set_xticks(x + width * (len(metrics_list) - 1) / 2)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Test Set Model Performance Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  QRShield — URL Phishing Detection Model Training")
    print("  (Validation-Guided Model Selection & Untouched Test Set)")
    print("=" * 65)

    df_raw = load_or_download_dataset()
    df = debias_and_balance_dataset(df_raw, max_samples_per_class=35000)

    # Extract features
    X, y = build_feature_matrix(df)
    print(f"[INFO] Feature matrix: {X.shape[0]} samples × {X.shape[1]} features.")

    # ── Split: 60% Train, 20% Validation, 20% Untouched Test ─────────────────
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    print(f"\n[INFO] Data Splitting:")
    print(f"  Training set:    {len(y_train)} samples ({len(y_train)/len(y)*100:.1f}%)")
    print(f"  Validation set:  {len(y_val)} samples ({len(y_val)/len(y)*100:.1f}%)")
    print(f"  Untouched Test:  {len(y_test)} samples ({len(y_test)/len(y)*100:.1f}%)")

    # ── Stage 1: Train Candidates on Training Set ────────────────────────────
    print("\n[INFO] Stage 1: Training candidate models on Training set …")
    et_candidate = ExtraTreesClassifier(
        n_estimators=150, max_depth=None, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_candidate = RandomForestClassifier(
        n_estimators=150, max_depth=None, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    et_candidate.fit(X_train, y_train)
    rf_candidate.fit(X_train, y_train)

    # ── Stage 2: Evaluate on Validation Set for Model Selection ───────────────
    print("\n[INFO] Stage 2: Evaluating models on Validation set …")
    val_et = evaluate_model('Extra Trees',   et_candidate, X_val, y_val, 'Validation')
    val_rf = evaluate_model('Random Forest', rf_candidate, X_val, y_val, 'Validation')

    # Selection criterion: Highest F1 on validation, with tie-break on recall and low FPR
    def selection_score(m):
        return (m['f1'], m['recall'], -m['fpr'], m['accuracy'])

    selected_name = 'Random Forest' if selection_score(val_rf) > selection_score(val_et) else 'Extra Trees'
    print(f"\n[INFO] MODEL SELECTION RESULT: '{selected_name}' selected based on validation performance.")

    # ── Stage 3: Retrain on Combined (Train + Validation) ─────────────────────
    print(f"\n[INFO] Stage 3: Retraining models on combined Train + Validation data ({len(y_train_val)} samples) …")
    et_final = ExtraTreesClassifier(
        n_estimators=150, max_depth=None, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_final = RandomForestClassifier(
        n_estimators=150, max_depth=None, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    et_final.fit(X_train_val, y_train_val)
    rf_final.fit(X_train_val, y_train_val)

    # ── Stage 4: Single Final Evaluation on Untouched Test Set ────────────────
    print("\n[INFO] Stage 4: Final evaluation on UNTOUCHED Test Set …")
    test_et = evaluate_model('Extra Trees',   et_final, X_test, y_test, 'Test')
    test_rf = evaluate_model('Random Forest', rf_final, X_test, y_test, 'Test')

    # ── Save Models ───────────────────────────────────────────────────────────
    winning_model = rf_final if selected_name == 'Random Forest' else et_final
    joblib.dump(winning_model, os.path.join(MODEL_DIR, 'url_model.pkl'))
    joblib.dump(et_final,      os.path.join(MODEL_DIR, 'extra_trees_url.pkl'))
    joblib.dump(rf_final,      os.path.join(MODEL_DIR, 'random_forest_url.pkl'))
    joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, 'feature_names.pkl'))
    print(f"\n[INFO] Saved models to {MODEL_DIR}")

    # ── Save Results & Metrics JSON ───────────────────────────────────────────
    def clean_metric(m):
        return {k: v for k, v in m.items() if not k.startswith('_')}

    results_summary = {
        'dataset': {
            'total_samples': len(df),
            'legitimate_count': int((df['label'] == 0).sum()),
            'phishing_count': int((df['label'] == 1).sum()),
            'feature_count': len(FEATURE_NAMES),
            'source': 'UCI PhiUSIIL Phishing URL Dataset (Normalized & De-biased)',
            'license': 'CC BY 4.0',
            'label_mapping': '0 = Legitimate, 1 = Phishing',
        },
        'split': {
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'methodology': '60/20/20 train/val/test; validation-guided selection; retrained on train+val; evaluated once on untouched test'
        },
        'validation_metrics': [clean_metric(val_et), clean_metric(val_rf)],
        'selected_model': selected_name,
        'test_metrics': [clean_metric(test_et), clean_metric(test_rf)],
        'feature_names': FEATURE_NAMES,
    }

    metrics_file = os.path.join(RESULT_DIR, 'url_model_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"[INFO] Saved test metrics to {metrics_file}")

    # ── Save Evaluation Plots ─────────────────────────────────────────────────
    test_models = [test_et, test_rf]
    save_confusion_matrix_plot(test_et['confusion_matrix'], 'Extra Trees',
                               os.path.join(RESULT_DIR, 'cm_extra_trees.png'))
    save_confusion_matrix_plot(test_rf['confusion_matrix'], 'Random Forest',
                               os.path.join(RESULT_DIR, 'cm_random_forest.png'))
    save_roc_plot(test_models, os.path.join(RESULT_DIR, 'roc_comparison.png'))
    save_feature_importance_plot(et_final, 'Extra Trees',
                                 os.path.join(RESULT_DIR, 'feature_importance_et.png'))
    save_feature_importance_plot(rf_final, 'Random Forest',
                                 os.path.join(RESULT_DIR, 'feature_importance_rf.png'))
    save_comparison_plot(test_models, os.path.join(RESULT_DIR, 'model_comparison.png'))
    print(f"[INFO] Saved all evaluation plots to {RESULT_DIR}")

    print("\n" + "=" * 65)
    print("  URL Model Training & Evaluation Complete!")
    print(f"  Selected Model: {selected_name}")
    print(f"  Test Accuracy:  {(test_rf if selected_name=='Random Forest' else test_et)['accuracy']*100:.2f}%")
    print(f"  Test Recall:    {(test_rf if selected_name=='Random Forest' else test_et)['recall']*100:.2f}%")
    print(f"  Test F1-Score:  {(test_rf if selected_name=='Random Forest' else test_et)['f1']:.4f}")
    print("=" * 65)


if __name__ == '__main__':
    main()
