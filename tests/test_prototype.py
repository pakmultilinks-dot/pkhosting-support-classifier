"""Sanity tests for the PKHosting support-query classifier prototype.
Run: PYTHONPATH=/home/hatch/tmp/pylibs python3 -m pytest tests/ -q
(or plain unittest: python3 -m unittest discover tests)
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import joblib  # noqa: E402
import pandas as pd  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..")
LABELS = {"account_login", "billing_invoices", "domain_dns", "email_problems",
          "performance_speed", "plans_upgrades", "refunds_cancellations",
          "setup_migration", "ssl_security", "website_down_errors"}


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(os.path.join(BASE, "data", "support_queries.csv"))

    def test_300_rows(self):
        self.assertEqual(len(self.df), 300)

    def test_balanced_classes(self):
        counts = self.df["category"].value_counts()
        self.assertTrue((counts == 30).all(), counts.to_dict())

    def test_labels_valid(self):
        self.assertEqual(set(self.df["category"]), LABELS)

    def test_no_duplicate_queries(self):
        self.assertEqual(self.df["query"].nunique(), 300)

    def test_no_empty_queries(self):
        self.assertTrue(self.df["query"].str.strip().str.len().min() > 10)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(os.path.join(BASE, "output", "model.joblib"))
        cls.results = json.load(open(os.path.join(BASE, "output", "results.json")))

    def test_predicts_valid_labels(self):
        preds = self.model.predict([
            "My website is down with a 500 error",
            "I want a refund for my hosting plan",
            "How do I renew my domain name?",
        ])
        for p in preds:
            self.assertIn(p, LABELS)

    def test_probabilities_sum_to_one(self):
        proba = self.model.predict_proba(["invoice copy please"])[0]
        self.assertAlmostEqual(float(proba.sum()), 1.0, places=5)

    def test_results_cover_all_classes(self):
        self.assertEqual(set(self.results["per_class"]), LABELS)
        for lab, m in self.results["per_class"].items():
            self.assertGreaterEqual(m["f1"], 0.0)
            self.assertLessEqual(m["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
