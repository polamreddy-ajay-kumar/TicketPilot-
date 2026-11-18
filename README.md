# TicketPilot: Support Ticket Triage Agent

A triage agent that classifies support tickets, drafts replies **grounded in a company policy
knowledge base** (RAG exposed as a tool), and **escalates to a human instead of guessing**.
Week 4 wraps it in an HTTP API, adds an evaluation set, guardrails and request logging.

| Week | What was added |
|---|---|
| 3 | Policy knowledge base, chunk cache, `search_policy_docs` tool, tool-calling loop, citations |
| 4 | Flask API (`/triage`, `/triage/batch`), input validation and safe errors, `eval_tickets.json` + `evaluate.py`, escalation guardrails, request log |

## Week 4 demo (real output)

**1. `POST /triage`: a real ticket returns category, reply, outcome and tool list. Empty body gives a clean 400, empty fields give a 422.**

![POST /triage, valid ticket and clean errors](screenshots/04_api_triage_and_errors.png)

**2. Guardrails: an ungrounded question, a low-confidence ticket and a safety ticket are escalated instead of answered. Every request is logged.**

![Guardrails and request log](screenshots/05_guardrails_and_request_log.png)

**3. `python evaluate.py --api ...` runs 28 checks through the live API and prints the accuracy; 54 tests pass.**

![Evaluation accuracy and test suite](screenshots/06_eval_accuracy_and_tests.png)

**4. The H08 bug fix: the policy now states the rule for orders still in the warehouse, and the agent answers conditionally with a citation.**

![H08 warehouse cancellation fix](screenshots/07_h08_warehouse_cancellation_fix.png)

The screenshots are rendered captures of real command output (live Flask server plus `curl`).

## Setup

```bash
pip install -r requirements.txt     # Flask only; Python 3.8+
```

## Run the API

```bash
python app.py                       # http://127.0.0.1:8000  (PORT=9000 python app.py to change)
```

```bash
curl -s -X POST localhost:8000/triage -H 'Content-Type: application/json' \
  -d '{"subject":"Refund deadline","body":"I received my order last week and want a refund. What is the refund deadline?"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /triage` | Body `{"subject", "body", "ticket_id"?}`. Returns `category`, `priority`, `confidence`, `outcome` (`answered` or `escalate_to_human`), `escalation_reason`, `reply`, `policy_chunks_used`, `policy_chunks_retrieved`, `tools_used`, `tool_trace`, `latency_ms` |
| `POST /triage/batch` | Stretch goal. `{"tickets": [...]}` (max 50). Returns per-ticket status and `elapsed_ms`, plus `total_latency_ms`. One bad ticket does not fail the batch |
| `GET /health` | Liveness check and number of loaded policy chunks |

### Error handling (the API never returns an HTML page or a stack trace)

| Situation | Response |
|---|---|
| Empty or non-JSON body | `400 invalid_json` |
| Empty or missing `subject` / `body`, wrong types, over 5000 chars | `422 invalid_ticket` with per-field details |
| Agent or tool raised an exception | `502 agent_failure` (nothing is answered) |
| Agent ran longer than 10 s | `504 agent_timeout` |
| More than 60 POSTs per minute from one IP (`TRIAGE_RATE_LIMIT_PER_MIN`) | `429 rate_limited` with `Retry-After` |
| Unknown route, wrong method, body over 256 KB | JSON `404` / `405` / `413` |
| Anything unexpected | JSON `500 internal_error` (details only in the server log) |

## Architecture

```
app.py        Flask: validation errors, status codes, rate limit        (thin)
triage.py     triage_ticket(): runs the agent, applies guardrails, logs  <- used by API AND eval
main.py       agent loop, tools (classify_ticket, lookup_order, search_policy_docs), RAG, cache
evaluate.py   runs eval_tickets.json through triage_ticket() or through POST /triage
```

The API calls the same `triage_ticket()` that `evaluate.py` calls, so the eval tests the real code
path rather than a duplicate.

## Guardrails (`triage.py`)

The first rule that fires wins and forces `outcome = "escalate_to_human"`:

1. **Policy ticket, zero policy lookups**: the agent skipped `search_policy_docs` on a ticket that
   mentions a policy topic (`policy_ticket_without_policy_lookup`). Its reply is discarded.
2. **Low confidence**: category confidence is below 0.5 (`low_category_confidence`).
   `classify_ticket` gives low confidence to ties between two intents (cancel and refund) and to tickets with no known keyword.
3. **Ungrounded**: the ticket is policy-related but no retrieved chunk supports a rule
   (`ungrounded_no_policy_support`).
4. **Escalation policy**: safety, fraud, legal or exception tickets go to a supervisor
   (`escalation_policy_requires_supervisor`).

Escalated replies never contain an invented rule.

## Request log

Every request appends one JSON line to `triage_requests.log` (override with `TRIAGE_LOG_FILE`):
timestamp, ticket id, category, confidence, outcome, escalation reason, tools called, latency.
Log failures never break a request. The log file is git-ignored.

## Evaluation

```bash
python evaluate.py                                   # calls triage_ticket() directly
python evaluate.py --api http://127.0.0.1:8000       # same tickets through POST /triage
python evaluate.py -v                                # also print replies of failing tickets
```

`eval_tickets.json` holds 28 tickets with an expected category, expected outcome, expected policy
file and expected rule text (`expected_phrase`). A ticket passes only if **all** checks pass:
category, outcome, policy cited, rule stated, tools called. The exit code is 0 only at 100%.

### Results

| Split | Result | Notes |
|---|---|---|
| `dev` (17) | 17/17 | The tickets I looked at while tuning |
| `held_out_round1` (5) | 5/5 | **First run was 3/5 (60%)**; fixed after reading the failures |
| `held_out_round2` (5) | 5/5 | **First run was 1/5 (20%)** on unseen tickets; see below |
| `held_out_round3` (1) | 1/1 | Written after the H08 fix; a weak generalization check (it reuses the word "warehouse") |
| **Overall** | **28/28 = 100%** | |

I read the failing replies by hand before changing anything. The fixes were real bugs, not score tuning:

| Failing case | Root cause | Fix |
|---|---|---|
| "Can I still cancel? It was already dispatched." was escalated | `cancel` (intent) tied with `dispatch` (context) so confidence was low | Intent words count double in `classify_ticket` |
| Fraud and legal ticket got "cannot confirm" | Long tickets diluted the keyword share, so nothing was retrieved. "alleging" was not in the alias table | Capped the dilution (`MAX_LEXICAL_TERMS`), added the alias |
| "Never arrived, want a refund" cited the 30-days-after-delivery refund rule | Shipping chunk was not retrieved and the refund chunk won | "never/not arrived" counts as a shipping signal; the classifier category is added to the search query and its policy file is preferred |
| "Ship to Canada, what does it cost?" answered with the refund shipping-charge rule | A single shared word (`shipping`) counted as grounding | Fallback rule selection now needs at least 2 specific shared words |
| "Cancel order ORD-2087" said "you may cancel" for an in-transit order | `lookup_order` result was ignored | Order status is folded into the ticket text before retrieval and rule selection |
| "Caught fire", "promised an exception", "damaged item" | The policy says safety, exception and defective; the customer says other words | Synonym aliases; `exception` and `supervisor` added to the escalation keywords (the policy lists them) |
| Eval expected "general" for "Do you deliver on Sundays?" | **My label was wrong**; it is a shipping ticket the docs do not cover | Corrected the expected label |
| "Cancel and refund" expected an answer | Two competing intents; a human should decide | Changed the expectation to `escalate_to_human` (a deliberate choice) |

Because the round-1 and round-2 tickets were used for fixes, they are no longer fully held out; the
first-run numbers above are the honest generalization estimate. **A 100% score on 28 tickets,
many of which shaped the fixes, is not a real-world accuracy claim.**

### H08 fix: "It hasn't left the warehouse. Can I cancel?"

This was the last failing case. It failed *safe* (escalated, never a wrong rule), but I read
`cancellation_policy.txt` before fixing it, and the policy does **not** say "not dispatched, so
cancellable". It says cancellation is allowed while the order is still being processed *and has not
entered fulfillment*, and that an order cannot be cancelled once it has entered fulfillment *or* has
been dispatched. An order sitting in the warehouse may already be in fulfillment, so the correct answer
is conditional, not "yes". My original eval expectation implied "yes", so that label was wrong too.

| Change | Where |
|---|---|
| Added the conditional rule to the policy: not yet left the warehouse can be cancelled **only if** it has not entered fulfillment; not being dispatched yet does not make an order cancellable; if unsure, contact support with the order number | `policy_docs/cancellation_policy.txt` (lands in `chunk-1`, since the whole document is under 300 words) |
| Agent prompt rule 5: map the customer's wording ("warehouse", "packed", "hasn't left") to the policy's stages (processing, fulfillment, dispatch) and search with both | `SYSTEM_PROMPT` in `main.py` |
| Agent prompt rule 6: never turn "not dispatched yet" into "you can cancel"; if the chunk says it depends on status, say that and send the customer to support | `SYSTEM_PROMPT` in `main.py` |
| `search_policy_docs` tool description now tells the model to include stage words in the query | `TOOL_SCHEMAS` |
| Local stand-in model: `stage_terms()` query expansion for cancellation tickets, plus a rule that selects the new sentence | `main.py` |
| Eval label corrected: H08 now expects the conditional rule (`has not entered fulfillment`). Added H11 with different wording | `eval_tickets.json` |

The policy clarification restates the existing rule and adds only the "contact support to check the
order status" step. **If the business wants warehouse orders to be cancellable, the policy owner must
decide that and change the document.** The agent will follow whatever the document says.

## Tests

```bash
python -m unittest discover -s tests      # 54 tests
```

Covered:
- API: valid ticket, empty and malformed body, validation, agent failure (502), timeout (504), unexpected exception (500), rate limit (429), batch.
- Guardrails: ungrounded, low confidence, zero tool calls, escalation.
- Log format.
- Eval set: well-formed, and the dev split passes through the real `triage_ticket()`.
- H08: conditional warehouse-cancellation reply, never an unconditional "yes", after-dispatch rule unchanged, prompt and tool-description rules present.
- Week 3: retrieval, cache, citations.

## Week 3 retrieval layer (unchanged design)

- `policy_docs/`: 4 policy `.txt` files. `chunk_text()` splits them (300 words, 40 overlap).
- Embeddings are cached in `embedding_cache.json`, keyed by the SHA-256 of the chunk text. The first run reports `0 HIT, 4 MISS`; later runs report `4 HIT, 0 MISS`.
- `search_policy_docs(query, top_k=3)` ranks chunks by cosine similarity plus keyword overlap, drops chunks below `MIN_RELEVANCE = 0.18`, and is exposed as a tool next to `classify_ticket` and `lookup_order`.
- `SYSTEM_PROMPT` requires the agent to search before any policy reply, use only retrieved text, cite the chunk id and say it cannot confirm when nothing supports the answer.
- `python main.py` runs 8 sample tickets with the full tool trace; `python main.py --topk-test` compares `top_k=1` and `top_k=3`.

Week 3 screenshots, captured at the Week 3 commit (scores may differ slightly from the current code):

![Week 3 fresh run](screenshots/01_fresh_run_cache_miss.png)
![Week 3 cache hit and cannot confirm](screenshots/02_cache_hit_and_cannot_confirm.png)
![Week 3 summary](screenshots/03_summary_and_tests.png)

## Known limitations

- **The "model" is a deterministic local stand-in** (`LocalPolicyModel` in `main.py`), not an LLM; no API key or network is needed. Pass your own callable as `model=` that returns `{"content", "tool_calls"}` to use a real LLM. The guardrails still apply to it. They were tested with a "lazy model" that skips retrieval.
- Embeddings are hashed bag-of-words, so paraphrases and synonyms outside the alias table are missed. H08 was fixed with a policy clarification plus query expansion, not with better retrieval; the next unseen paraphrase may fail again. A real embedding model is the next step.
- `classify_ticket` and `lookup_order` are keyword and mock-data placeholders for the Week 2 tools.
- The rate limiter is in-memory and per process. Use a shared store (for example Redis) behind several workers.
- Flask's built-in server is for development. Run it behind gunicorn or waitress in production.
- The agent timeout abandons the result but cannot kill the worker thread; a stuck tool would keep that thread busy.
- Replies are drafted from a single chunk, so a ticket mixing two policies gets one rule or is escalated.
