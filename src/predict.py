#!/usr/bin/env python3
"""Classify one support query with the trained model.

Usage:
  python3 src/predict.py "My website is down and shows a 500 error"
Prints the predicted category and the top-3 category probabilities.
"""
import os
import sys

import joblib

MODEL = os.path.join(os.path.dirname(__file__), "..", "output", "model.joblib")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    query = " ".join(sys.argv[1:])
    model = joblib.load(MODEL)
    pred = model.predict([query])[0]
    proba = model.predict_proba([query])[0]
    top = sorted(zip(model.classes_, proba), key=lambda x: -x[1])[:3]
    print(f"Query:     {query}")
    print(f"Predicted: {pred}")
    print("Top 3:     " + ", ".join(f"{c} ({p:.2f})" for c, p in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
