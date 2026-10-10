# Fraud Risk Lab

[![Tests](https://github.com/yasaswireddy7032/fraud-risk-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/yasaswireddy7032/fraud-risk-lab/actions)

A reproducible synthetic fraud-classification experiment with validation-selected thresholds and held-out evaluation.

**Status:** runnable portfolio reference implementation. Uses synthetic or hand-authored demonstration data. No employer code, customer data, or production performance claims.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
python -m unittest -v
```

The default command prints a reproducible JSON demonstration. See `app.py` for the reusable Python API. `example-output.json` records the locally executed demo; it is not an external benchmark.

## Architecture

Seeded synthetic transaction features → sequential train/validation/test split → train-only scaling and logistic regression → validation threshold selection → untouched test report.

## Validation

Tests exercise failure cases and behavioral invariants, in addition to the executable demo. GitHub Actions runs the tests and demo on pushes and pull requests. See the Actions tab for current results.

## Limitations and next steps

Synthetic features and labels cannot establish financial efficacy or fairness. There is no live bank integration or real fraud decisioning. Validate with consented representative data, account-level separation, delayed labels, calibration, subgroup analysis, and human review before any operational use.

## Container

```bash
docker build -t fraud-risk-lab .
docker run --rm fraud-risk-lab
```

The Docker recipe is provided; local validation uses Python, not Docker.

## Responsible use

Review the code and understand its assumptions before deploying. Keep credentials in environment variables and exclude private data from commits. This project is an educational engineering baseline, not a production-certified service.
