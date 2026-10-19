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
│   ├── extra_trees_image.pkl Image Extra Trees classifier
│   ├── random_forest_image.pkl Image Random Forest classifier
│   └── QR_Phishing_Residual_CNN.keras Residual CNN (inference gated pending preprocessing/label verification)
├── results/           Genuine evaluation metrics, confusion matrices, ROC curves
├── test_qrshield.py   Automated unit and integration test suite (30 tests)
├── test_api.py        Live server endpoint verification script
├── requirements.txt   Python dependencies
├── cnn_service/       Remote CNN inference service scaffold (disabled until verified)
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── requirements-cnn.txt  TensorFlow/Keras dependencies for the remote runtime only
└── README.md
```

---

## Quick Start Commands

### 1. Open in VS Code and Select Python
Open the repository root folder in VS Code, then select the existing virtual environment:

1. Open the Command Palette (`Ctrl+Shift+P`).
2. Run **Python: Select Interpreter**.
3. Select `.venv\Scripts\python.exe`.

From a PowerShell terminal opened at the repository root:

```powershell
cd QRShield
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Start the Local Application
From the `QRShield` directory:

```powershell
..\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000** in your browser. The URL and QR analysis endpoints use the existing saved models; you do not need to train them to run the app.

### 3. Run the Automated Tests
In a second terminal, from the `QRShield` directory:

```powershell
..\.venv\Scripts\python.exe test_qrshield.py
..\.venv\Scripts\python.exe test_api.py
```

`test_api.py` checks the live server and therefore requires the server from step 2 to be running.

### Image Model Selection and CNN Compatibility
The Scanner page lets you select Extra Trees or Random Forest for an additional QR-image prediction. The QR is still decoded and URL-analyzed separately. Image models use grayscale 69×69 pixels, float32 values scaled to [0, 1], and the documented label mapping `0 = Benign`, `1 = Phishing`. Uploads that are not already 69×69 are converted to grayscale and resized with Pillow's bilinear resampler before scaling; the training code stores 69×69 inputs and documents scaling but does not specify an upload resize filter.

The Residual CNN expects a `(69, 69, 1)` float32 input and returns one sigmoid score. The training notebook converts the existing grayscale images to float32, scales/clips pixels to `[0, 1]`, and adds the channel dimension. For uploaded images, QRShield converts to grayscale and resizes to 69×69 with Pillow bilinear resampling before scaling; that upload resize choice is experimental because the notebook assumes the dataset images are already 69×69. The notebook selects threshold `0.05` on validation data to target at least 90% phishing recall, choosing the best precision/F1 among eligible thresholds. Label `1` is interpreted as Phishing according to the QR dataset documentation.

The local backend has a Residual CNN inference path that loads the existing `.keras` file lazily, validates its input/output shapes, and keeps model results marked **EXPERIMENTAL**. If a compatible local runtime is unavailable, the Scanner returns a fixed `Phishing` **demo-only** label with no confidence; this is not a model prediction and does not depend on the uploaded image. It does not retrain the model or send images externally. The runtime pins are listed in `requirements-cnn.txt` (TensorFlow 2.20.0, Keras 3.13.2); the backend reports the reason actual inference is unavailable. No packages are installed by this project automatically.

To check local CNN readiness, start the website and open `/health`. The response includes `cnn_prediction_enabled` and `cnn_status_reason`. Extra Trees, Random Forest, and URL analysis continue to use their existing code paths. The `cnn_service` folder remains an unused remote-service scaffold; local inference does not call or deploy it.

### Optional: Retrain the URL Model
Training is not needed for normal use. It writes new model and evaluation files into `models/` and `results/`, so skip it to preserve the existing evaluated model and reported results.

```powershell
..\.venv\Scripts\python.exe training\train_url_model.py
```

This script:
- Verifies label mapping (`0 = Legitimate`, `1 = Phishing`).
- De-biases synthetic artifacts (apex domains, subdomains, benign paths).
- Splits into 60% Train, 20% Validation, 20% Untouched Test.
- Evaluates Extra Trees and Random Forest on the Validation set and selects the superior model.
- Retrains the selected model on Train + Validation combined.
- Evaluates once on the untouched Test set and saves genuine metrics/charts to `results/`.

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
