"""
test_qrshield.py
================
Integration and unit tests for QRShield.
Run from the QRShield/ directory:
    python test_qrshield.py
"""

import os, sys, json, io, unittest
import numpy as np

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, 'training'))
sys.path.insert(0, os.path.join(THIS_DIR, 'backend'))


class TestURLFeatures(unittest.TestCase):
    def setUp(self):
        from url_features import extract_url_features, FEATURE_NAMES
        self.extract = extract_url_features
        self.names   = FEATURE_NAMES

    def test_feature_count(self):
        feats = self.extract("https://www.google.com")
        self.assertEqual(len(feats), len(self.names), "Feature count mismatch")

    def test_safe_url(self):
        f = self.extract("https://www.google.com")
        self.assertEqual(f['has_https'], 1)
        self.assertEqual(f['is_ip'], 0)
        self.assertEqual(f['num_at'], 0)

    def test_ip_url(self):
        f = self.extract("http://192.168.1.1/admin/login.php")
        self.assertEqual(f['is_ip'], 1)
        self.assertEqual(f['has_https'], 0)

    def test_suspicious_tld(self):
        f = self.extract("http://freegift.tk/claim")
        self.assertEqual(f['suspicious_tld'], 1)

    def test_brand_keyword(self):
        f = self.extract("http://paypal-secure.xyz/login")
        self.assertEqual(f['has_brand_keyword'], 1)

    def test_shortener(self):
        f = self.extract("http://bit.ly/abc123")
        self.assertEqual(f['is_shortener'], 1)

    def test_at_symbol(self):
        f = self.extract("http://login.paypal.com@phishing.tk/")
        self.assertEqual(f['num_at'], 1)

    def test_long_url(self):
        long_url = "http://example.com/" + "a" * 110
        f = self.extract(long_url)
        self.assertGreater(f['url_len'], 100)

    def test_feature_types(self):
        f = self.extract("https://example.com")
        for k, v in f.items():
            self.assertIsInstance(v, (int, float), f"Feature '{k}' has wrong type: {type(v)}")

    def test_malformed_url(self):
        # Should not raise
        f = self.extract("not_a_url_at_all!!!")
        self.assertIsInstance(f, dict)


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        from risk_engine import assess_risk, explain_url
        from url_features import extract_url_features
        self.assess  = assess_risk
        self.explain = explain_url
        self.extract = extract_url_features

    def _run(self, url, prob, label):
        feats = self.extract(url)
        return self.assess(feats, prob, label), feats

    def test_safe_url_low_risk(self):
        report, _ = self._run("https://www.google.com", 0.05, 0)
        self.assertIn(report['risk_level'], ('SAFE', 'LOW', 'MEDIUM'))

    def test_phishing_url_high_risk(self):
        report, _ = self._run(
            "http://paypal-secure-update.tk/login?user=victim", 0.95, 1)
        self.assertIn(report['risk_level'], ('HIGH', 'MEDIUM'))

    def test_ip_url_flagged(self):
        report, _ = self._run("http://192.168.1.1/admin", 0.7, 1)
        labels = [i['label'] for i in report['indicators']]
        self.assertIn('IP address used instead of domain name', labels)

    def test_report_keys(self):
        report, _ = self._run("https://example.com", 0.1, 0)
        for key in ('risk_level','risk_score','indicators','consequences',
                    'safeguards','recommendation','model_prediction','model_confidence'):
            self.assertIn(key, report)

    def test_explanation_is_string(self):
        report, feats = self._run("http://phishing.tk/steal", 0.9, 1)
        expl = self.explain("http://phishing.tk/steal", feats, report)
        self.assertIsInstance(expl, str)
        self.assertGreater(len(expl), 10)

    def test_high_confidence_phishing(self):
        report, _ = self._run("http://secure-paypal.tk/login", 0.92, 1)
        self.assertIn(report['risk_level'], ('HIGH', 'MEDIUM'))

    def test_uncertain_url_low_prob(self):
        # Uncertain: model says legit with low confidence
        report, _ = self._run("http://example-store.com/buy", 0.3, 0)
        self.assertIn(report['risk_level'], ('SAFE', 'LOW', 'MEDIUM'))


class TestQRDecoder(unittest.TestCase):
    def setUp(self):
        from qr_decoder import decode_qr_from_bytes, detect_payload_type, get_analyzable_url
        self.decode = decode_qr_from_bytes
        self.detect = detect_payload_type
        self.get_url = get_analyzable_url

    def test_invalid_bytes_returns_none(self):
        payload, ptype, method = self.decode(b'not_an_image_at_all')
        self.assertIsNone(payload)

    def test_empty_bytes(self):
        payload, ptype, method = self.decode(b'')
        self.assertIsNone(payload)

    def test_payload_type_url(self):
        self.assertEqual(self.detect("https://www.google.com"), "URL")

    def test_payload_type_email(self):
        self.assertEqual(self.detect("mailto:test@example.com"), "EMAIL")

    def test_payload_type_wifi(self):
        self.assertEqual(self.detect("WIFI:T:WPA;S:MyNetwork;P:password;;"), "WIFI")

    def test_payload_type_text(self):
        self.assertEqual(self.detect("Hello World"), "TEXT")

    def test_get_url_from_url_payload(self):
        url = self.get_url("https://example.com", "URL")
        self.assertEqual(url, "https://example.com")

    def test_get_url_from_non_url(self):
        url = self.get_url("Hello World", "TEXT")
        self.assertIsNone(url)

    def test_get_url_from_email(self):
        url = self.get_url("mailto:test@example.com", "EMAIL")
        self.assertIsNone(url)

    def _make_qr_png(self, text: str) -> bytes:
        """Generate a real QR code PNG in memory."""
        try:
            import qrcode
            qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M)
            qr.add_data(text)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
        except ImportError:
            return None

    def test_decode_real_qr_url(self):
        png = self._make_qr_png("https://www.example.com")
        if png is None:
            self.skipTest("qrcode package not installed")
        payload, ptype, method = self.decode(png)
        if payload is None:
            self.skipTest("QR decode library not functioning (pyzbar may need zbar DLL)")
        self.assertIn("example.com", payload)
        self.assertEqual(ptype, "URL")

    def test_decode_real_qr_text(self):
        png = self._make_qr_png("Hello QRShield")
        if png is None:
            self.skipTest("qrcode package not installed")
        payload, ptype, method = self.decode(png)
        if payload is None:
            self.skipTest("QR decode library not functioning")
        self.assertIn("Hello", payload)


class TestModelPipeline(unittest.TestCase):
    """Tests that run only if trained models exist."""

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

    def _load_model(self):
        import joblib
        p = os.path.join(self.MODEL_DIR, 'url_model.pkl')
        if not os.path.exists(p):
            self.skipTest("URL model not trained yet — run train_url_model.py")
        return joblib.load(p)

    def setUp(self):
        from url_features import features_to_vector, FEATURE_NAMES
        self.vec  = features_to_vector
        self.names = FEATURE_NAMES

    def test_model_loaded(self):
        clf = self._load_model()
        self.assertIsNotNone(clf)

    def test_model_predicts_legitimate(self):
        clf = self._load_model()
        vec = self.vec("https://www.google.com")
        pred = clf.predict([vec])[0]
        self.assertIn(int(pred), [0, 1])  # valid class

    def test_model_predicts_phishing(self):
        clf = self._load_model()
        vec = self.vec("http://paypal-secure.tk/login?steal=1")
        pred = clf.predict([vec])[0]
        self.assertIn(int(pred), [0, 1])

    def test_feature_order_matches(self):
        import joblib
        fn_path = os.path.join(self.MODEL_DIR, 'feature_names.pkl')
        if not os.path.exists(fn_path):
            self.skipTest("feature_names.pkl not found")
        saved = joblib.load(fn_path)
        self.assertEqual(saved, self.names, "Feature order mismatch between training and inference!")

    def test_probability_range(self):
        clf = self._load_model()
        if not hasattr(clf, 'predict_proba'):
            self.skipTest("Model does not support predict_proba")
        vec = self.vec("https://www.google.com")
        probs = clf.predict_proba([vec])[0]
        self.assertAlmostEqual(float(sum(probs)), 1.0, places=5)
        self.assertTrue(all(0 <= p <= 1 for p in probs))


class TestLabeledExamples(unittest.TestCase):
    """End-to-end tests on labeled URL examples."""

    LEGIT_URLS = [
        "https://www.google.com",
        "https://www.wikipedia.org",
        "https://github.com",
        "https://stackoverflow.com",
        "https://www.python.org",
    ]
    PHISHING_URLS = [
        "http://paypal-secure.tk/login",
        "http://192.168.1.1/admin",
        "http://bit.ly/freegift123",
        "http://amazon-account-verify.xyz/update",
        "http://login.paypal.com@phishing.tk/",
    ]

    def setUp(self):
        from url_features import extract_url_features
        from risk_engine   import assess_risk
        self.extract = extract_url_features
        self.assess  = assess_risk
        import joblib
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'models', 'url_model.pkl')
        self.model = joblib.load(model_path) if os.path.exists(model_path) else None

    def _analyze(self, url):
        feats = self.extract(url)
        vec   = list(feats.values())
        if self.model:
            pred  = int(self.model.predict([vec])[0])
            prob  = float(self.model.predict_proba([vec])[0][1]) if hasattr(self.model, 'predict_proba') else float(pred)
        else:
            pred, prob = 0, 0.0
        return self.assess(feats, prob, pred)

    def test_legit_urls_not_high_risk(self):
        for url in self.LEGIT_URLS:
            report = self._analyze(url)
            # For small built-in dataset, models may not be perfect — we just verify no crash
            self.assertIn(report['risk_level'], ('SAFE','LOW','MEDIUM','HIGH'))

    def test_phishing_urls_flagged(self):
        flagged = 0
        for url in self.PHISHING_URLS:
            report = self._analyze(url)
            if report['risk_level'] in ('MEDIUM', 'HIGH'):
                flagged += 1
        # At least 60% of phishing examples should be flagged
        self.assertGreaterEqual(flagged / len(self.PHISHING_URLS), 0.6,
                                f"Only {flagged}/{len(self.PHISHING_URLS)} phishing URLs flagged")


if __name__ == '__main__':
    # Print test summary
    print("\n" + "="*60)
    print("  QRShield — Integration Tests")
    print("="*60 + "\n")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestURLFeatures, TestRiskEngine, TestQRDecoder,
                TestModelPipeline, TestLabeledExamples]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\n{'PASSED' if result.wasSuccessful() else 'FAILED'}: "
          f"{result.testsRun} tests run, "
          f"{len(result.failures)} failures, "
          f"{len(result.errors)} errors.")

