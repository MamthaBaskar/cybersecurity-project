# QRShield — Explainable Pre-Navigation QR Code Phishing Detection

QRShield analyzes QR code destinations **before** you visit them, using machine learning and static URL feature extraction to detect phishing and quishing threats.

> **Security guarantee:** QRShield operates strictly pre-navigation. It decodes QR payloads locally and analyzes URL strings statically — no external HTTP requests or redirects are ever performed during analysis.

---

## Project Structure

```
QRShield/
├── frontend/          Clean single-page interface (Scanner, Threat Report, Model Performance)
│   └── index.html
├── backend/           FastAPI app, local QR decoder, hardened upload validation, risk engine
│   ├── main.py        FastAPI API server (restricted CORS to localhost)
│   ├── qr_decoder.py  Local QR decoder (pyzbar + OpenCV fallback)
│   ├── risk_engine.py Explainable threat scoring & guidance engine
│   └── url_features.py  30 static lexical URL feature extractors
├── training/
│   ├── url_features.py   Static feature extractor pipeline (synced with backend)
│   ├── train_url_model.py  Validation-guided training of Extra Trees & Random Forest
│   └── train_image_model.py  Training on Quishing QR image pixel features
├── datasets/          Labelled datasets and provenance documentation
│   ├── url_dataset.csv
│   └── dataset_info.md
├── models/            Saved models and feature pipeline
│   ├── url_model.pkl         Selected URL classifier (Extra Trees)
│   ├── extra_trees_url.pkl   Extra Trees candidate
│   ├── random_forest_url.pkl Random Forest candidate
│   ├── feature_names.pkl     Ordered 30-feature list
│   └── image_model.pkl       Selected image classifier
├── results/           Genuine evaluation metrics, confusion matrices, ROC curves
├── test_qrshield.py   Automated unit and integration test suite (30 tests)
├── test_api.py        Live server endpoint verification script
├── requirements.txt   Python dependencies
└── README.md
```

---

## Quick Start Commands

### 1. Install Dependencies
```cmd
cd e:\cybersecurity-project\QRShield
pip install -r requirements.txt
```

### 2. Train the URL Model (Validation-Guided Selection & Retraining)
```cmd
python training\train_url_model.py
```
This script:
- Verifies label mapping (`0 = Legitimate`, `1 = Phishing`).
- De-biases synthetic artifacts (apex domains, subdomains, benign paths).
- Splits into 60% Train, 20% Validation, 20% Untouched Test.
- Evaluates Extra Trees and Random Forest on the Validation set and selects the superior model.
- Retrains the selected model on Train + Validation combined.
- Evaluates once on the untouched Test set and saves genuine metrics/charts to `results/`.

### 3. Run the Automated Test Suites
```cmd
python test_qrshield.py
python test_api.py
```

### 4. Start the Web Server (Localhost Default)
```cmd
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```
Open your browser at: **http://127.0.0.1:8000**

---

## Datasets & Label Convention

| Dataset | Source | Samples | Labels |
|---|---|---|---|
| **URL Phishing (PhiUSIIL)** | [UCI ML Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (CC BY 4.0) | 70,020 balanced, de-biased samples | **0 = Legitimate**, **1 = Phishing** |
| **QR Image (Quishing)** | Workspace archive (`QuishingDataset_extracted`) | 9,987 (69×69 px grayscale) | **0 = Benign**, **1 = Phishing** |

---

## Evaluated Model Performance (Untouched Test Set)

### URL Classifier (14,004 Untouched Test URLs)
| Model | Accuracy | Phishing Recall | Precision | F1-Score | False Positive Rate | ROC-AUC |
|---|---|---|---|---|---|---|
| **Extra Trees (Selected)** | **93.87%** | **96.23%** | **91.88%** | **0.9401** | **8.50%** | **0.9776** |
| **Random Forest** | 93.94% | 96.07% | 92.15% | 0.9407 | 8.18% | 0.9843 |

### QR Image Classifier (1,998 Untouched Test Images)
| Model | Accuracy | Phishing Recall | Precision | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Extra Trees (Selected)** | **82.08%** | **74.82%** | **87.46%** | **0.8065** | **0.9026** |
| **Random Forest** | 81.28% | 72.82% | 87.58% | 0.7952 | 0.8981 |

---

## Security & Pre-Navigation Protections

1. **Bare Hostname Normalization**: Inputs like `www.google.com` or `google.com` are normalized to `https://...` for feature extraction. Explicit schemes (`http://...`) are preserved, and any assumed scheme is reported.
2. **True Brand Impersonation**: Brand names (Google, PayPal, GitHub, etc.) are only flagged when detected on untrusted/spoofed domains or subdomains, preventing false positives on official domains.
3. **Hardened Upload Validation**:
   - Streaming 10 MB maximum size enforcement (aborts early on oversized payloads).
   - Magic bytes verification (JPEG, PNG, GIF, WebP, BMP).
   - PIL verification and max dimension limits (4096 × 4096 px).
4. **Missing Model Handling**: If the model is missing or fails, QRShield returns `UNKNOWN` / `INSUFFICIENT EVIDENCE` — never defaulting to `SAFE`.
5. **Separated Confidence & Probability**: Reports both calculated $P(\text{Phishing})$ and the model's confidence in its prediction.
6. **Restricted CORS**: API restricted to `localhost:8000` / `127.0.0.1:8000`.
