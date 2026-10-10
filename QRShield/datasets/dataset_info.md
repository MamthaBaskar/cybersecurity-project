# QRShield Dataset Information

## Dataset A — QR Image Dataset (Quishing)

| Property | Value |
|---|---|
| Source | QuishingDataset.zip (local workspace) |
| Format | Pickle files: `qr_codes_29.pickle` (images), `qr_codes_29_labels.pickle` (labels) |
| Total samples | 9,987 |
| Benign (label 0) | 5,005 |
| Phishing (label 1) | 4,982 |
| Image size | 69 × 69 pixels (grayscale) |
| Feature count | 4,761 pixel features per sample |
| Use | Image-based QR phishing detection (Extra Trees on pixel features) |
| Split | 80% train, 20% test |

## Dataset B — URL Phishing Dataset

| Property | Value |
|---|---|
| Source | UCI / Kaggle phishing URL datasets, merged and deduplicated |
| URL: [PhiUSIIL Phishing URL Dataset (UCI)](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) |
| Features | Static lexical URL features (no live fetching) |
| Labels | 0 = legitimate, 1 = phishing |
| Use | URL-based classification for QR payload analysis |

> **Note:** URL features are extracted purely from the URL string — no HTTP requests are made during feature extraction.

