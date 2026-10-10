import joblib, numpy as np, sys
sys.path.insert(0, r'e:\cybersecurity-project\QRShield\training')
from url_features import extract_url_features, features_to_vector, FEATURE_NAMES

model = joblib.load(r'e:\cybersecurity-project\QRShield\models\url_model.pkl')

tests = [
    ('https://www.google.com',           0),
    ('https://www.wikipedia.org',         0),
    ('https://github.com',               0),
    ('http://paypal-secure-update.tk/login', 1),
    ('http://amazon-account-verify.xyz/update', 1),
    ('http://192.168.1.1/admin/login.php', 1),
    ('http://bit.ly/freegift123',          1),
    ('http://login.paypal.com@phishing.tk/', 1),
]

correct = 0
for url, true_label in tests:
    vec = features_to_vector(url)
    pred = int(model.predict([vec])[0])
    prob = float(model.predict_proba([vec])[0][1])
    ok = pred == true_label
    correct += ok
    status = 'OK' if ok else 'WRONG'
    print(f'{url[:48]:48s} true={true_label} pred={pred} prob={prob*100:5.1f}% {status}')

print(f'\nCorrect: {correct}/{len(tests)}')

# Check feature values for google.com
print('\nFeatures for https://www.google.com:')
feats = extract_url_features('https://www.google.com')
for k, v in feats.items():
    print(f'  {k}: {v}')

# Check what column the PhiUSIIL dataset used
import pandas as pd
ds = pd.read_csv(r'e:\cybersecurity-project\QRShield\datasets\url_dataset.csv')
print('\nDataset columns:', list(ds.columns))
print('Sample URLs:')
print(ds.head(3).to_string())
print('\nLabel distribution:')
print(ds['label'].value_counts())

