import urllib.request, json

# Test 1: Health check
req = urllib.request.urlopen('http://localhost:8000/health', timeout=5)
data = json.loads(req.read())
print('HEALTH CHECK:')
print(json.dumps(data, indent=2))

# Test 2: Analyze phishing URL
payload = json.dumps({'url': 'http://paypal-secure-update.tk/login'}).encode()
req2 = urllib.request.Request('http://localhost:8000/api/analyze-url',
    data=payload, headers={'Content-Type': 'application/json'}, method='POST')
resp2 = urllib.request.urlopen(req2, timeout=10)
data2 = json.loads(resp2.read())
print('\nURL ANALYSIS (phishing):')
print('  URL:         ', data2['url'])
print('  Risk Level:  ', data2['risk_level'])
print('  Prediction:  ', data2['model_prediction'])
conf2 = data2['model_confidence'] * 100
print('  Confidence:  ', f'{conf2:.1f}%')
print('  Indicators:  ', [i['label'] for i in data2['indicators']])
print('  Recommend:   ', data2['recommendation'])

# Test 3: Analyze safe URL
payload3 = json.dumps({'url': 'https://www.google.com'}).encode()
req3 = urllib.request.Request('http://localhost:8000/api/analyze-url',
    data=payload3, headers={'Content-Type': 'application/json'}, method='POST')
resp3 = urllib.request.urlopen(req3, timeout=10)
data3 = json.loads(resp3.read())
print('\nURL ANALYSIS (safe):')
print('  URL:        ', data3['url'])
print('  Risk Level: ', data3['risk_level'])
print('  Prediction: ', data3['model_prediction'])
conf3 = data3['model_confidence'] * 100
print('  Confidence: ', f'{conf3:.1f}%')

# Test 4: Performance endpoint
req4 = urllib.request.urlopen('http://localhost:8000/api/performance', timeout=5)
data4 = json.loads(req4.read())
print('\nPERFORMANCE ENDPOINT:')
um = data4.get('url_model', {})
if 'models' in um:
    for m in um['models']:
        auc = m.get('roc_auc', 'N/A')
        print(f"  {m['model']}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} auc={auc}")
charts = list(data4.get('charts', {}).keys())
print('  Charts:', charts)

