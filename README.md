# ReorderPilot

LLM sales assistant for B2B sellers: predicts which products each customer will re-order in the next 30 days,
explains why, abstains when unsure, and learns from sales-rep feedback through persistent memory.

> Work in progress. See `SPEC.md` for the specification. Full README (results, limitations, attribution) comes in M6.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
cp .env.example .env          # then fill in your keys
pytest -q
```

## Data and attribution

Uses the UCI "Online Retail II" dataset (CC BY 4.0). Full citation added in M6. No raw data is stored in this repo.
