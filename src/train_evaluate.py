#!/usr/bin/env python3
"""
Train and evaluate the PKHosting support-query classifier.

Evaluation design (all three views are reported, none hidden):
  1. PRIMARY: stratified random 80/20 split (240 train / 60 test, seed 42).
     This is the headline result with F1 by class and the confusion matrix.
  2. ROBUSTNESS: 5-fold stratified cross-validation on all 300 queries.
  3. STRESS TEST: train on the 240 "core" phrasings only, test on the 60
     "held-out" phrasings (wording never seen in training). Scores drop here;
     it estimates how much the prototype depends on familiar wording.

Model: TF-IDF (word 1-2 grams + char 3-5 grams) + Logistic Regression.
Baselines on the primary split: Multinomial Naive Bayes, Linear SVM.

Outputs (output/): results.json, classification_report.txt,
confusion_matrix.png, model.joblib
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

LABELS = ["account_login", "billing_invoices", "domain_dns", "email_problems",
          "performance_speed", "plans_upgrades", "refunds_cancellations",
          "setup_migration", "ssl_security", "website_down_errors"]


def make_pipeline(clf):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 sublinear_tf=True, min_df=1)),
    ])
    return Pipeline([("tfidf", feats), ("clf", clf)])


def lr():
    return LogisticRegression(max_iter=1000, C=6.0, solver="lbfgs")


def main():
    df = pd.read_csv(os.path.join(BASE, "data", "support_queries.csv"))

    # ---- 1. primary stratified split ----
    train, test = train_test_split(df, test_size=0.2, stratify=df["category"],
                                   random_state=42)
    Xtr, ytr, Xte, yte = train["query"], train["category"], test["query"], test["category"]
    print(f"primary split: train {len(train)} / test {len(test)}")

    candidates = {
        "Logistic Regression": make_pipeline(lr()),
        "Multinomial Naive Bayes": make_pipeline(MultinomialNB(alpha=0.3)),
        "Linear SVM": make_pipeline(LinearSVC(C=0.6)),
    }
    comparison = {}
    for name, pipe in candidates.items():
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        comparison[name] = {"accuracy": round(accuracy_score(yte, p), 4),
                            "macro_f1": round(f1_score(yte, p, average="macro"), 4)}
        print(f"{name:26s} acc={comparison[name]['accuracy']:.4f} macroF1={comparison[name]['macro_f1']:.4f}")

    model = candidates["Logistic Regression"]
    pred = model.predict(Xte)
    rep = classification_report(yte, pred, labels=LABELS, output_dict=True, zero_division=0)
    acc = accuracy_score(yte, pred)
    macro = f1_score(yte, pred, average="macro")
    weighted = f1_score(yte, pred, average="weighted")
    print(f"PRIMARY TEST acc={acc:.4f} macroF1={macro:.4f} weightedF1={weighted:.4f}")

    # ---- 2. 5-fold CV on all 300 ----
    cv_pred = cross_val_predict(make_pipeline(lr()), df["query"], df["category"],
                                cv=StratifiedKFold(5, shuffle=True, random_state=42))
    cv_acc = accuracy_score(df["category"], cv_pred)
    cv_macro = f1_score(df["category"], cv_pred, average="macro")
    print(f"CV(5) acc={cv_acc:.4f} macroF1={cv_macro:.4f}")

    # ---- 3. stress test: core phrasings -> held-out phrasings ----
    core = df[df.template_group == "core"]
    held = df[df.template_group == "heldout"]
    stress = make_pipeline(lr()).fit(core["query"], core["category"])
    sp = stress.predict(held["query"])
    s_acc = accuracy_score(held["category"], sp)
    s_macro = f1_score(held["category"], sp, average="macro")
    print(f"STRESS (unseen phrasings) acc={s_acc:.4f} macroF1={s_macro:.4f}")

    mis = [{"query": q, "true": t, "predicted": p}
           for q, t, p in zip(Xte, yte, pred) if t != p]
    sample_preds = [{"query": q, "true": t, "predicted": p, "correct": bool(t == p)}
                    for q, t, p in list(zip(Xte, yte, pred))[:15]]

    results = {
        "dataset": {"total": int(len(df)), "classes": 10,
                    "primary_split": "stratified 80/20, seed 42",
                    "train": int(len(train)), "test": int(len(test))},
        "model_comparison_on_test": comparison,
        "primary_model": "TF-IDF (word 1-2g + char 3-5g) + Logistic Regression",
        "test_metrics": {"accuracy": round(acc, 4), "macro_f1": round(macro, 4),
                         "weighted_f1": round(weighted, 4)},
        "cv5_metrics": {"accuracy": round(cv_acc, 4), "macro_f1": round(cv_macro, 4)},
        "stress_test_unseen_phrasings": {"accuracy": round(s_acc, 4),
                                         "macro_f1": round(s_macro, 4),
                                         "train": "240 core phrasings",
                                         "test": "60 held-out phrasings"},
        "per_class": {lab: {"precision": round(rep[lab]["precision"], 4),
                            "recall": round(rep[lab]["recall"], 4),
                            "f1": round(rep[lab]["f1-score"], 4),
                            "support": int(rep[lab]["support"])} for lab in LABELS},
        "misclassifications": mis,
        "sample_predictions": sample_preds,
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "classification_report.txt"), "w") as f:
        f.write(classification_report(yte, pred, labels=LABELS, zero_division=0))
    joblib.dump(model, os.path.join(OUT, "model.joblib"))

    cm = confusion_matrix(yte, pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), [l.replace("_", "\n") for l in LABELS], fontsize=7.5)
    ax.set_yticks(range(len(LABELS)), [l.replace("_", " ") for l in LABELS], fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix - primary test set (60 queries, stratified split)")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
    print("misclassified:", len(mis), "of", len(test), "| outputs saved to", OUT)


if __name__ == "__main__":
    main()
