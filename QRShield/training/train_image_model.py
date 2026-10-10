"""
train_image_model.py
====================
Stage 1 — Train image-based QR phishing classifier on the QuishingDataset.
Dataset: 9,987 QR images (69×69 px), labels: 0=benign, 1=phishing.

Run from the QRShield/training/ directory:
    python train_image_model.py
"""

import os, sys, pickle, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)

warnings.filterwarnings('ignore')

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(THIS_DIR)
DATASET_DIR = os.path.join(os.path.dirname(ROOT), 'QuishingDataset_extracted')
MODEL_DIR  = os.path.join(ROOT, 'models')
RESULT_DIR = os.path.join(ROOT, 'results')
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def load_dataset():
    X_path = os.path.join(DATASET_DIR, 'qr_codes_29.pickle')
    y_path = os.path.join(DATASET_DIR, 'qr_codes_29_labels.pickle')
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Dataset not found: {X_path}")
    print(f"[INFO] Loading image dataset from {DATASET_DIR} …")
    with open(X_path, 'rb') as f:
        X = pickle.load(f)
    with open(y_path, 'rb') as f:
        y = pickle.load(f)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=int)
    # Flatten 69x69 images to 4761-dim vectors
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    # Normalize pixel values to [0, 1]
    if X.max() > 1.0:
        X = X / 255.0
    return X, y


def evaluate_model(name, clf, X_train, y_train, X_test, y_test):
    print(f"\n[INFO] Training {name} …")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

    metrics = {
        'model':     name,
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
    metrics['classification_report'] = classification_report(
        y_test, y_pred, target_names=['Benign', 'Phishing'], output_dict=True)

    print(f"\n{'='*55}")
    print(f"  {name} (Image Model)")
    print(f"{'='*55}")
    for k in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
        if k in metrics:
            print(f"  {k:20s}: {metrics[k]:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Benign','Phishing'])}")
    return metrics, clf


def save_plots(metrics_list):
    # Confusion matrices
    for m in metrics_list:
        cm = np.array(m['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Phishing'],
                    yticklabels=['Benign', 'Phishing'], ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f"Confusion Matrix — {m['model']} (Image)")
        plt.tight_layout()
        fname = f"cm_image_{m['model'].lower().replace(' ','_')}.png"
        plt.savefig(os.path.join(RESULT_DIR, fname), dpi=150)
        plt.close()
        print(f"[INFO] Saved: {fname}")

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in metrics_list:
        if '_roc_fpr' in m:
            ax.plot(m['_roc_fpr'], m['_roc_tpr'],
                    label=f"{m['model']} (AUC={m.get('roc_auc','?'):.4f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve — Image Models'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'roc_image_comparison.png'), dpi=150)
    plt.close()
    print("[INFO] Saved: roc_image_comparison.png")

    # Model comparison bar
    names = [m['model'] for m in metrics_list]
    keys  = ['accuracy', 'precision', 'recall', 'f1']
    x     = np.arange(len(keys)); width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics_list):
        ax.bar(x + i * width, [m.get(k, 0) for k in keys], width, label=m['model'])
    ax.set_xticks(x + width * (len(metrics_list) - 1) / 2)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
    ax.set_title('Image Model Comparison'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'model_comparison_image.png'), dpi=150)
    plt.close()
    print("[INFO] Saved: model_comparison_image.png")


def main():
    print("\n" + "="*60)
    print("  QRShield — QR Image Model Training")
    print("="*60)

    X, y = load_dataset()
    n_total = len(y)
    n_benign  = int((y == 0).sum())
    n_phishing = int((y == 1).sum())
    print(f"[INFO] Loaded: {n_total} samples ({n_benign} benign, {n_phishing} phishing)")
    print(f"[INFO] Feature shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"[INFO] Split: train={len(y_train)}, test={len(y_test)}")

    # Extra Trees
    et = ExtraTreesClassifier(
        n_estimators=150, max_depth=None, class_weight='balanced',
        random_state=42, n_jobs=-1)
    et_metrics, et_clf = evaluate_model('Extra Trees', et, X_train, y_train, X_test, y_test)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=None, class_weight='balanced',
        random_state=42, n_jobs=-1)
    rf_metrics, rf_clf = evaluate_model('Random Forest', rf, X_train, y_train, X_test, y_test)

    all_metrics = [et_metrics, rf_metrics]

    # Select best
    def score(m): return (m['recall'], m['f1'])
    best_m = max(all_metrics, key=score)
    best_clf = et_clf if best_m['model'] == 'Extra Trees' else rf_clf
    print(f"\n[INFO] Selected: {best_m['model']} (recall={best_m['recall']:.4f})")

    # Save models
    joblib.dump(best_clf, os.path.join(MODEL_DIR, 'image_model.pkl'))
    joblib.dump(et_clf,   os.path.join(MODEL_DIR, 'extra_trees_image.pkl'))
    joblib.dump(rf_clf,   os.path.join(MODEL_DIR, 'random_forest_image.pkl'))
    print(f"[INFO] Models saved to {MODEL_DIR}")

    # Save plots
    save_plots(all_metrics)

    # Save metrics JSON
    def clean(m):
        return {k: v for k, v in m.items() if not k.startswith('_')}

    summary = {
        'dataset': {
            'total': n_total,
            'benign': n_benign,
            'phishing': n_phishing,
            'image_size': '69x69',
            'features': int(X.shape[1]),
        },
        'split': {'train': int(len(y_train)), 'test': int(len(y_test))},
        'models': [clean(et_metrics), clean(rf_metrics)],
        'selected_model': best_m['model'],
    }
    metrics_path = os.path.join(RESULT_DIR, 'image_model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Metrics saved to {metrics_path}")
    print(f"\n[DONE] Image model training complete.")


if __name__ == '__main__':
    main()

