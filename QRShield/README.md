# QRShield — Explainable Pre-Navigation QR Code Phishing Detection

QRShield analyzes QR code destinations **before** you visit them, using machine learning and static URL analysis to detect phishing.

> **Security guarantee:** QRShield never visits, fetches, or opens the decoded URL during analysis.

---

## Project Structure

```
QRShield/
├── frontend/          HTML, CSS, JS — single-page web interface
├── backend/           FastAPI app, QR decoder, risk engine
│   ├── main.py        API server
│   ├── qr_decoder.py  QR decoding (pyzbar + OpenCV fallback)
│   ├── risk_engine.py Rule-based threat scoring and explanations
│   └── url_features.py  30 static URL lexical features
├── training/
│   ├── url_features.py   Feature extraction (same as backend)
│   ├── train_url_model.py  Train Extra Trees & Random Forest on URL data
│   └── train_image_model.py  Train on QR image pixel data
├── datasets/          Labelled URL CSV, dataset documentation
├── models/            Saved models (url_model.pkl, image_model.pkl, ...)
├── results/           Metrics JSON, confusion matrices, ROC curves, charts
├── test_qrshield.py   Integration and unit tests
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```cmd
cd e:\cybersecurity-project\QRShield
pip install -r requirements.txt
```

> **Note on pyzbar:** On Windows, pyzbar requires the ZBar DLL.  
> Download `zbar-0.10` from https://zbar.sourceforge.net/  
> Or install via `conda install -c conda-forge zbar` if using Anaconda.  
> OpenCV QR detection is used as fallback if pyzbar fails.

### 2. Train the URL model

```cmd
python training\train_url_model.py
```

This will:
- Download the PhiUSIIL URL dataset from UCI (if network is available)
- Fall back to a built-in labeled URL set if network is unavailable
- Train Extra Trees and Random Forest classifiers
- Save models to `models/`
- Save metrics and charts to `results/`

### 3. Train the QR image model (optional but recommended)

```cmd
python training\train_image_model.py
```

This uses the existing `QuishingDataset_extracted/` pickle files.

### 4. Run tests

```cmd
python test_qrshield.py
```

### 5. Start the web server

```cmd
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open: **http://localhost:8000**

---

## Datasets

### Dataset A — QR Image Dataset (Quishing)
| Property | Value |
|---|---|
| File | `QuishingDataset_extracted/qr_codes_29.pickle` |
| Total | 9,987 samples |
| Benign | 5,005 (label 0) |
| Phishing | 4,982 (label 1) |
| Feature | 69×69 pixel images → 4,761 features |

### Dataset B — URL Phishing Dataset
| Property | Value |
|---|---|
| Source | PhiUSIIL (UCI ML Repository, 2023) or built-in URL examples |
| URL | https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset |
| Features | 30 static lexical URL features (no HTTP requests) |
| Labels | 0 = legitimate, 1 = phishing |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Backend health and model status |
| GET | `/api/performance` | Saved evaluation metrics and chart paths |
| POST | `/api/analyze` | Upload QR image file for analysis |
| POST | `/api/analyze-url` | Analyze a URL string directly |

---

## URL Features (30 total)

| Feature | Description |
|---|---|
| `url_len` | Total URL length |
| `host_len` | Hostname length |
| `num_dots`, `num_hyphens`, `num_slashes` | Special character counts |
| `is_ip` | IP address used instead of domain |
| `has_https` | HTTPS scheme |
| `suspicious_tld` | Free/abused TLD (e.g., .tk, .ml, .xyz) |
| `subdomain_count` | Number of subdomain levels |
| `host_entropy` | Shannon entropy of hostname |
| `has_brand_keyword` | Impersonation of PayPal, Amazon, etc. |
| `has_phishing_keyword` | Keywords: free, prize, verify, etc. |
| `is_shortener` | Known URL shortener service |
| `num_at` | @ symbol in URL |
| ... | 18 more features |

---

## Risk Levels

| Level | Meaning |
|---|---|
| 🟢 SAFE | No indicators found, model confidence low |
| 🟡 LOW | Minor indicators — still verify source |
| 🟠 MEDIUM | Several suspicious indicators — proceed with caution |
| 🔴 HIGH | Strong phishing signals — do not proceed |

> A SAFE result does not guarantee the URL is safe. Always verify the source of QR codes in unexpected or unsolicited contexts.

---

## Security Design

- **No URL navigation:** The decoded URL string is analyzed as text only. No HTTP requests are made to the destination.
- **SSRF prevention:** The backend only processes the URL string; it never fetches external URLs.
- **File validation:** Uploaded images are validated for type and size before processing.
- **Model loaded at startup:** The trained model is loaded once and reused; the performance page never retrains.
- **Privacy:** URLs are not transmitted to third-party services unless explicitly configured.

---

## Troubleshooting

**QR code not detected:**
- Ensure the image is clear, well-lit, and the QR code fills most of the frame
- Try PNG or JPG format
- pyzbar needs ZBar DLL on Windows — see Quick Start note

**Model not found error:**
- Run `python training\train_url_model.py` first

**Backend not reachable:**
- Ensure `uvicorn` is running: `cd backend && uvicorn main:app --port 8000`
- Check firewall settings if accessing from another device

**ImportError on startup:**
- Run `pip install -r requirements.txt` in the QRShield directory

