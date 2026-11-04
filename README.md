# ReorderPilot

An LLM sales assistant for B2B sellers. It predicts which products each customer will **re-order in the next 30 days**,
explains why, **abstains when unsure**, and **learns from the sales rep's feedback** through persistent memory
(e.g. "declined: overstocked" → that product is not offered again for 30 days).

Built on the public UCI *Online Retail II* dataset (a UK gift wholesaler, 2009-2011). The system runs at a simulated
"today" of **2011-11-01** (`AS_OF_DATE`), so every result is reproducible.

---

## 1. What it does, for whom, why this topic

**User:** a B2B sales rep with hundreds of wholesale customers who doesn't know whom to call today or what to offer.

**Pain:** untargeted calls; offering items the customer just declined; nothing remembered between calls.

**What ReorderPilot does:**
- ranks customers by expected re-order value ("who should I call today?");
- gives a per-customer briefing: top products with calibrated probabilities, a plain-language reason for each,
  caveats and a short suggested message;
- abstains ("not enough history") instead of guessing for new or low-confidence customers;
- records the rep's feedback and applies **deterministic rules** to it: a declined-overstocked product is
  suppressed for 30 days, and the suppression is visible and auditable.

**Why this topic:** it combines a classic ML problem (tabular prediction with a temporal split and honest baselines)
with an LLM agent. The agent never makes up numbers: every product and probability comes from a tool.

## 2. Choices and alternatives

Full reasoning is in [DECISIONS.md](DECISIONS.md). The most important ones:

| Decision | Chosen | Alternatives considered | Why |
|---|---|---|---|
| Model | HistGradientBoosting + Platt calibration | logistic regression, rule baselines only | handles missing values natively, beats baselines on PR-AUC |
| Evaluation split | temporal (train ≤ 2011-06, valid Jul-Aug, test Sep-Nov) | random split | random split leaks the future |
| Uncertainty | abstain (cold start < 3 orders, or below threshold tau) | always answer | a wrong confident offer costs the rep's trust |
| Memory | SQLite + rules in code | vector DB, LLM decides | small data, auditable, testable |
| Agent | hand-written loop (~150 lines) | LangChain / agent frameworks | every step explainable, easy to test with a fake LLM |
| LLM | any OpenAI-compatible API (Groq `gpt-oss-120b` by default) | one vendor SDK | provider-agnostic; free tier for development |
| Product matching | word coverage over descriptions | fuzzy WRatio | WRatio matched wrong products on short queries (D9) |

## 3. How it works

```mermaid
flowchart LR
    REP(["Sales rep"]) --> UI["Streamlit UI"]
    UI -- HTTP --> API["FastAPI"]
    API --> LOOP["Agent loop<br/>(max 6 steps)"]
    LOOP <--> LLM["LLM<br/>(OpenAI-compatible)"]
    LOOP --> TOOLS["Tools<br/>(Pydantic-validated)"]
    TOOLS --> MODEL["Re-order model<br/>calibrated + abstain"]
    TOOLS --> MEM["Memory policy"]
    MEM <--> DB[("SQLite memory.db")]
    MODEL --> TX[("transactions.parquet")]
```

The agent has 6 tools: `list_customers_due`, `predict_reorder`, `get_customer_history`, `read_memory`,
`write_memory`, `find_product`. The policy that turns feedback into suppression lives in `memory/store.py`, not in
the prompt. The detailed component diagram, the sequence of one chat turn and the feedback loop are in
[docs/architecture.md](docs/architecture.md).

## 4. How I got here (problems met and fixes)

- **Duplicate rows:** the two sheets of the xlsx overlap by 9 days (≈34k duplicate rows). Found by checking row counts,
  fixed in cleaning (DECISIONS D3).
- **Calibration vs season:** Platt scaling fitted on Jul-Aug slightly *hurt* Brier on the Sep-Nov test months
  (Christmas season re-orders more). Kept and documented instead of tuning on test.
- **LLM rate limits:** the Groq free tier allows ~8K tokens/min; one briefing costs ~5-6K. Added a client-side rate
  limiter, retry on HTTP 429 honouring `retry-after`, and an on-disk response cache so eval reruns are free.
- **Wrong product resolved:** fuzzy matching (WRatio) picked wrong items for short names; replaced with word coverage (D9).
- **My own metric bugs:** a non-ASCII dash (U+2011) and substring product names changed eval results. Lesson: read
  raw answers before trusting an automatic metric.
- **Prompt scope:** "Who should I call today?" was refused as out-of-scope (eval D, d01). Fixed in prompt v2 (D12),
  checked with 5 phrasings against the real LLM; the full eval D rerun on v2 passes all 3 due-customer scenarios.
- **Invented product codes:** the prompt says "only use tool data", but a prompt is not a guarantee. A deterministic
  guard now removes answer lines citing stock codes that no tool returned (D14).
- **Docker had no data:** compose mounted `./data` while `.env` pointed elsewhere; fixed with a mount driven by the
  same `.env` value and a one-shot `init` service (D15).
- **Git branch overlap & test trace leakage:** Merging parallel feature tracks (`feat/model` and `feat/memory-api`) created conflicts in data loader stubs and `DECISIONS.md`. Resolved by inspecting `data/clean.py` manually to preserve full ML logic. Additionally, `pytest` runs were initially leaking live traces into the production LangSmith project; fixed by explicitly disabling tracing in `tests/conftest.py`.

## 5. How I verified it

All numbers below come from running the scripts in `eval/`; raw outputs are in `eval/results/`.

### A. Model vs baselines (test cutoffs 2011-09-01, 10-01, 11-01)

Eligible = customers with ≥ 3 prior orders; per-customer metrics use the 2,732 (customer, cutoff) pairs that
re-ordered at least one previously bought product. 1,030,502 candidate pairs, base rate 4.5%.

| method | precision@5 | recall@5 | hit_rate@5 | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| B1 repeat last order | 0.311 | 0.171 | 0.702 | 0.117 | 0.703 | n/a |
| B2 most frequent | 0.333 | 0.191 | 0.751 | 0.162 | 0.716 | n/a |
| B3 overdue | 0.233 | 0.150 | 0.627 | 0.105 | 0.706 | n/a |
| **HGB calibrated** | **0.381** | **0.222** | **0.796** | **0.261** | **0.838** | 0.038 |

- vs the strongest baseline (B2): PR-AUC 0.261 vs 0.162 (+61%), precision@5 0.381 vs 0.333.
- Abstention at tau = 0.08: coverage 32.3% of customers with history, precision of recommended items 0.203
  (21,100 items), vs the 4.5% base rate.
- Plots: [calibration](eval/results/calibration.png), [abstention curve](eval/results/abstention_curve.png).

### B. Memory ablation (does feedback stop repeat offers?)

Each test customer's top-1 product is logged as `declined / overstocked`, then the agent is asked for a briefing.

Model `openai/gpt-oss-120b`, prompt v2, grounding guard on.

| metric (n = 30 customers) | memory ON | memory OFF |
|---|---|---|
| repeat-recommendation rate of the declined item | 10% | 100% |
| answer mentions the suppression | 90% | 0% |
| `predict_reorder` flagged the item as suppressed | 100% | 0% |

The rule in code worked every time: the tool flagged the declined item in 30 of 30 briefings. But in 3 of 30
(12908, 16933, 16637) the LLM still listed it with its probability despite `suppressed=true`. I read those raw answers:
they are real failures, not a metric artefact. An earlier run on n = 4 showed 0%; the larger sample is the honest number.

### C. Feedback extraction (rep sentence → structured `write_memory`)

Model `openai/gpt-oss-120b`, prompt v2, grounding guard on. All 4 fields exact: 90% (18 of 20 cases). Per field: customer_id 100%, outcome 100%, stock_code 100%, reason 90%.
Both failures are `asked_alternative` events logged with `reason: other` instead of no reason (c08, c14): the outcome
and product are right, and an `asked_alternative` reason has no effect on suppression.

### D. Agent behaviour scenarios

Pass rate 95% (19 of 20 scenarios) across: due-customers 3/3, briefing 4/4, explain-why 3/3, log-feedback 3/3,
cold-start abstention 2/3, out-of-scope 2/2, malicious-note injection 2/2. Checks: expected tools called, no product
absent from tool outputs, abstention respected. The one failure (d17) is a **false negative of the metric**: the agent
did abstain ("there isn't enough ordering history"), but the keyword check does not know the phrase "isn't enough".
I report the automatic score as measured and did not change the check after seeing the result.

### Tests

`pytest -q`: 89 tests, including a leakage test (future rows change no feature), memory policy rules,
API endpoints, the agent loop with a scripted fake LLM (malformed JSON retry, max steps), a malicious-note
injection test and headless UI tests.

### Limitations (honest)

- One dataset, one retailer, historical data; the "today" is simulated.
- Test months are the Christmas season: calibration fitted on summer months is slightly off.
- Cold-start customers (< 3 orders) always get an abstention: no recommendation for new customers.
- Agent evals B-D run on the full sets (B n = 30, C n = 20, D n = 20) with one model and one run each: good enough
  to spot failure modes, still small for precise rates (one case = 3-5 percentage points).
- Suppression is enforced in the tool output, not in the final text: the LLM ignored the `suppressed` flag in 3 of 30
  briefings (eval B).
- Automatic checks miss some errors seen in manual review (e.g. a wrong date in one answer, an invented
  "back in stock" claim). The no-hallucination check covers products, not every sentence.
- The LLM sometimes skips `read_memory`. Suppression still works, because it is applied inside `predict_reorder`.

## 6. What I'd do next

- A deterministic post-check for suppressed items, like the grounding guard: drop answer lines that recommend a
  product the tool flagged as suppressed (closes the 3/30 gap from eval B).
- Make the eval D abstention check robust to phrasing (an LLM judge or a structured `abstained` flag), fixing the
  d17 false negative; rerun the agent evals on more than one model.
- Recommend *new* products (cross-sell), not only re-orders; a cold-start model for new customers.
- Learn suppression lengths from data instead of fixed 30/90 days.
- Out of scope for this project: authentication, multi-tenant, real CRM integration, voice, cloud deployment,
  Postgres instead of SQLite for many concurrent users.

## How I worked with AI agents

The project was built with AI coding agents (mainly Claude Code) working in parallel, each limited to one area
("lane"). I wrote the spec, split the work, reviewed and merged.

**1. Contract first (T0).** One scaffold commit created `SPEC.md`, `config.py`, the shared Pydantic models in
`api/schemas.py` and stub functions for every module. The schemas are the contract between lanes: the API lane could
build `/predict` against a `PredictionResult` before the model existed, and test it with a fake predictor.

**2. One lane per agent, one git worktree per lane.** A git worktree is a second working folder of the same
repository on its own branch, so agents edit files in parallel without overwriting each other.

| lane | branch | owns |
|---|---|---|
| A — data + model | `feat/model` | cleaning, features, training, calibration, model eval |
| B — memory + API | `feat/memory-api` | SQLite store, suppression policy, FastAPI endpoints |
| C — agent | `feat/agent-core` | LLM client, tools, hand-written loop, evals B-D |
| D — UI + packaging + docs | `feat/ui-docs` | Streamlit app, Docker, README, architecture |

Worktrees share one data folder outside the repo (`REORDERPILOT_DATA_DIR`), because a worktree has no `.env` and raw
data must not be committed (SPEC section 5).

**3. Prompt isolation.** Each agent got a task prompt naming only its lane's files and the SPEC sections it
implements; the shared rules live in `CLAUDE.md` / `AGENTS.md` (stay in your lane, never change a contract silently,
never invent metrics). An agent could read the whole repo but not edit outside its lane, so a contract change had
to come back to me as a proposal instead of a silent edit.

**4. pytest as the gate.** "Done" meant tests + `ruff` green, plus a `LEARNING_LOG` entry and, for any choice
between alternatives, a `DECISIONS` entry. Cross-lane pieces were tested against fakes: a scripted fake LLM for the
agent loop, an injectable predictor and temp SQLite file for the API. When lane C wired the real agent in, the API's
"501 not implemented" test was replaced by an end-to-end test of the endpoint with a fake LLM.

**5. Handoffs and merging.** Each lane ended with a HANDOFF note in `docs/LEARNING_LOG.md` (done, half-done, next
steps, how to test), so the next session, or a different tool, could continue. Lane branches were merged into
`main`; the one real conflict (lanes A + B) was resolved in its own commit.

### Reflection: Orchestrating AI Agents & Code Ownership

Working with multi-agent workflows (Agents A, B, C, and D) dramatically accelerated velocity, but it required strict human oversight to maintain architecture coherence. 

* **What I reviewed line-by-line vs. skimmed:** I performed meticulous line-by-line code reviews on the core domain logic—specifically feature engineering in `features.py`, deterministic suppression rules in `store.py`, and the anti-hallucination grounding checks in `grounding.py`. I skimmed generated boilerplate unit tests and raw Streamlit CSS styling, verifying them via execution output rather than manual code audit.
* **Concrete corrections made:** When integrating Agent C, an existing API test (`test_unwired_agent_returns_501`) failed because it expected a `501 Not Implemented` stub; I updated the assertion to validate proper `200 OK` execution once the agent loop was wired. I also manually resolved Git merge conflicts between model and memory tracks in `DECISIONS.md` and `data/clean.py`, ensuring live data loaders were preserved over mock stubs.
* **What I would not delegate next time:** I would not delegate initial guardrail/scope boundary definitions or multi-branch Git merges. Agents tend to duplicate file stubs or write overly restrictive system prompts (such as refusing "Who should I call today?" queries). Setting scope definitions and handling branch integrations are tasks best kept strictly under human control.

---

## Run it

Requirements: Python 3.11+ (tested on 3.12), an API key for an OpenAI-compatible LLM (Groq free tier works),
the dataset (below). Optional: Docker.

### 1. Get the data

Download *Online Retail II* from the [UCI page](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
and put the `.xlsx` (or a `.csv`) into `data/raw/`. The data folder can be elsewhere: set `REORDERPILOT_DATA_DIR`.

### 2a. Local run

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
cp .env.example .env                   # then set LLM_API_KEY (and LANGSMITH_API_KEY, or LANGSMITH_TRACING=false)

python -m reorderpilot.data.loader     # raw xlsx -> data/processed/transactions.parquet (a few minutes)
python -m reorderpilot.model.train     # -> artifacts/model.joblib + artifacts/metrics.json
pytest -q                              # tests (use temp files, never the real memory.db)

uvicorn reorderpilot.api.main:app --reload       # terminal 1 -> http://localhost:8000/docs
streamlit run app/streamlit_app.py               # terminal 2 -> http://localhost:8501
```

### 2b. Docker run

Prerequisites: `cp .env.example .env` (set `LLM_API_KEY`), and the dataset file in `<data dir>/raw/`, where
`<data dir>` is `REORDERPILOT_DATA_DIR` from `.env` (default `./data`). Then:

```bash
docker compose build
docker compose up                      # API http://localhost:8000/docs, UI http://localhost:8501
```

Start order: `init` → `api` (healthy) → `ui`. `init` builds `transactions.parquet` and the model only if they are
missing (about 3.5 min the first time), then exits; later starts skip it. The data dir and `./artifacts` are mounted
as volumes: the image contains code only, and the data and model stay on your disk.

End-to-end check (builds, starts, and tests `/health`, `/customers/due` and the UI health endpoint):
`python scripts/verify_docker.py` (add `--down` to stop the containers afterwards).

Windows: Docker Desktop needs WSL2. If `wsl --install` fails with `HCS_E_SERVICE_NOT_AVAILABLE`, enable the Windows
feature "Virtual Machine Platform" (`dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all`) and
reboot.

### Evaluation

```bash
python eval/run_model_eval.py
python eval/run_memory_eval.py --limit 5     # LLM evals accept --limit and --model (token budget)
python eval/run_agent_eval.py --limit 5
```

## Data and attribution

This project uses the **Online Retail II** dataset:

- Source: UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- Author: Daqing Chen. DOI: 10.24432/C5CG6D
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

No raw or processed data is redistributed in this repository: `data/` is gitignored, and the Docker image does not
contain it. Changes made to the data (cleaning rules) are described in `src/reorderpilot/data/clean.py` and
DECISIONS D3.
