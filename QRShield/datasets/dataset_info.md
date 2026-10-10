# QRShield Dataset Documentation

## Dataset A — QR Image Dataset (Quishing)

| Property | Value |
|---|---|
| **Source** | `QuishingDataset.zip` (Local repository archive) |
| **Format** | Serialized numpy arrays: `qr_codes_29.pickle` (images), `qr_codes_29_labels.pickle` (labels) |
| **Sample Count** | 9,987 total samples |
| **Class Distribution** | Benign: 5,005 (50.1%), Phishing: 4,982 (49.9%) |
| **Dimensions** | 69 × 69 pixels (grayscale) = 4,761 raw pixel features |
| **Task** | Image-level pixel pattern classifier |
| **Split Strategy** | 80% Train (7,989), 20% Test (1,998) stratified |
| **Model** | Extra Trees Classifier (Accuracy: 82.08%, Recall: 74.82%, ROC-AUC: 0.9026) |

---

## Dataset B — URL Phishing Dataset (PhiUSIIL)

| Property | Value |
|---|---|
| **Primary Source** | [UCI Machine Learning Repository — PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Label Mapping** | **Original UCI**: `1 = Legitimate`, `0 = Phishing`<br>**QRShield Standard**: `0 = Legitimate`, `1 = Phishing` *(Validated & Normalized)* |
| **Total Samples** | 70,020 balanced, de-biased samples (35,020 Legitimate, 35,000 Phishing) |
| **Features Extracted** | 30 static lexical features (lengths, entropy, character counts, IP detection, TLD reputation, brand impersonation) |
| **Network Policy** | Purely static string parsing — zero HTTP/DNS network requests during extraction |

### Preprocessing & De-Biasing
- **Synthetic Artifact Elimination**: In raw PhiUSIIL, 100% of legitimate URLs were artificially prefixed with `https://www.`. Real-world benign URLs frequently operate as apex domains (`https://domain.tld`), use `http://`, or have path routes. The dataset was de-biased so benign samples represent apex domains (45%), `www.` (50%), and standard path routes.
- **Normalization**: Bare hostnames (e.g. `www.google.com`, `google.com`) are normalized by assuming `https://` during inference, while preserving explicit schemes and reporting the assumption transparently.

### Evaluation Methodology
1. **Splitting**: 60% Train (42,012), 20% Validation (14,004), 20% Untouched Test (14,004) stratified.
2. **Model Selection**: Extra Trees and Random Forest trained on the 60% Train set and evaluated on the 20% Validation set.
3. **Retraining**: Winning candidate (Extra Trees, Validation F1: 0.9394) retrained on combined Train + Validation (56,016 samples).
4. **Final Reporting**: Evaluated exactly once on the untouched held-out Test set (14,004 samples).

### Measured Test Set Results
- **Selected Model**: Extra Trees Classifier
- **Test Accuracy**: 93.87%
- **Phishing Recall**: 96.23%
- **Precision**: 91.88%
- **F1-Score**: 0.9401
- **False Positive Rate**: 8.50%
- **ROC-AUC**: 0.9776

> **Notice on Fallback Dataset**: If UCI repository connectivity is unavailable during setup, a built-in fallback dataset is utilized. Fallback results are explicitly marked in training logs as **DEMONSTRATION-ONLY** and should be replaced with the genuine dataset for production deployment.
