# PKHosting Support Query Classifier - Prototype

A text-classification prototype that routes PKHosting-style customer support
queries into 10 categories, trained and evaluated on a **fully synthetic**
dataset of 300 queries (no real customer data).

## Categories

| Category | Covers |
|---|---|
| account_login | cPanel / client-area login, password resets, account access |
| billing_invoices | invoices, charges, payment methods, receipts |
| domain_dns | domain registration, transfer, nameservers, DNS records |
| email_problems | sending/receiving mail, webmail, mailbox quota, spam-folder issues |
| performance_speed | slow sites, resource limits, caching, CDN |
| plans_upgrades | plan comparison, pricing questions, upgrades (pre-sales) |
| refunds_cancellations | refund requests, cancellations, money-back guarantee |
| setup_migration | new-site setup, WordPress install, host-to-host migration |
| ssl_security | SSL certificates, "not secure" warnings, hacked sites, malware |
| website_down_errors | site down, 500/502/404 errors, database connection errors |

## Dataset

`data/support_queries.csv` - 300 synthetic queries, 30 per category, generated
by `src/generate_dataset.py` (seed 42) from hand-written phrasing templates with
randomized slots (fictional domains, plan names, PKR amounts). About 7% are
Roman Urdu, reflecting PKHosting's customer base. Each row also carries a
`template_group` flag (`core` / `heldout`): the 60 held-out phrasings are used
only for the stress test described below.

## Model

TF-IDF features (word 1-2 grams + character 3-5 grams) with Logistic
Regression, in one scikit-learn pipeline (`src/train_evaluate.py`).

## Results (23 Oct 2026 run)

Primary evaluation, stratified 80/20 split (240 train / 60 test):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (primary) | 0.900 | 0.898 |
| Linear SVM | 0.917 | 0.915 |
| Multinomial Naive Bayes | 0.883 | 0.882 |

F1 by class (Logistic Regression): billing_invoices 1.00, plans_upgrades 1.00,
domain_dns 0.92, email_problems 0.92, refunds_cancellations 0.92,
account_login 0.91, ssl_security 0.91, setup_migration 0.83,
website_down_errors 0.83, performance_speed 0.73.

Additional views, reported so the headline number is not read in isolation:

- 5-fold cross-validation on all 300 queries: macro F1 0.810.
- Stress test on completely unseen phrasings (train on core wording only,
  test on held-out wording): macro F1 0.449. The prototype leans on familiar
  wording; with real ticket history (thousands of naturally varied queries)
  this gap is expected to narrow substantially. Full numbers and every
  misclassification are in `output/results.json`.

## Run it

```bash
pip install -r requirements.txt
python3 src/generate_dataset.py     # rebuilds data/support_queries.csv
python3 src/train_evaluate.py       # trains, evaluates, writes output/
python3 src/predict.py "My website is down and shows a 500 error"
python3 -m unittest discover tests  # 8 sanity tests
```

## Layout

- `data/support_queries.csv` - the synthetic dataset
- `src/generate_dataset.py` - dataset generator
- `src/train_evaluate.py` - training + evaluation
- `src/predict.py` - single-query demo CLI
- `output/` - results.json, classification_report.txt, confusion_matrix.png,
  model.joblib
- `tests/test_prototype.py` - dataset/model sanity tests

## Limitations

Synthetic data is cleaner than real tickets, so treat the primary-split score
as an upper bound. Real queries are messier, multi-intent, and drift over time.
This is a routing prototype, not a production system: next steps are training
on anonymized real tickets, a confidence threshold with human fallback, and
periodic re-evaluation.
