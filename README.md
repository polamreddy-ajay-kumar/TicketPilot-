# TicketPilot: Support Ticket Triage Agent

A small, offline support-ticket triage demo. It loads exactly 15 synthetic tickets in `tickets.json`, gathers relevant context with tools, and prints a suggested category, tool-call history, and draft reply for every ticket.

## Sample output

Output of `python main.py --quiet`: one row per ticket, then the Run Summary.

![TicketPilot console output](assets/output_preview.png)

## Toolset

- `classify_ticket_category(text)` uses simple keyword rules to return `billing`, `shipping`, `bug`, or `other`. Software failures are checked before shipping keywords, so a crashing tracking page is treated as a bug rather than a delivery-status request.
- `lookup_order(order_id)` returns synthetic order status and estimated delivery information; `ORD-1007` also has mock payment-review context for T-014.
- `check_refund_eligibility(order_id)` returns a synthetic eligibility result and reason.
- `escalate_to_human(reason)` records an in-process request for a support specialist to review an issue and returns a structured result: `{"recorded": true, "reason": ...}`. It does not contact a person or external service.

The tools and order data live in `tool.py`; the agent loop, system prompt and console output live in `main.py`. The order data is intentionally fictional. Estimated delivery dates are generated relative to the current date so the demo remains timely. An eligible result means a request can be reviewed; it does not mean a refund has been issued. The agent does not issue refunds or send replies. Drafts are suggestions, and uncertain cases should be reviewed by a person.

## Multi-round agent loop

`SYSTEM_PROMPT` describes the triage role, tool boundaries, and when order/refund context is required. `run_agent` repeatedly asks the model adapter for either tool calls or a final response. Each tool call is executed, logged with its arguments and result, and made available to the next turn. The loop is limited to five tool-call rounds; when the cap is reached, it logs a warning and returns a safe handoff draft. Identical tool calls (same tool and arguments) are executed only once; later repeats are logged and skipped. Escalation is additionally limited to one call per ticket, even if a repeated request changes its reason. The round cap still applies even when the model repeatedly requests a skipped call.

## Prompt Fix Log

- Ticket `T-003` exposed a classification issue: an app crash report that says no payment was made must be `bug`, not `billing`. The system prompt and classification rules were clarified so crashes, software errors, and login failures take priority over payment terms; the expected category is `bug`.
- Ticket `T-007` exposed a classification-order issue: “tracking page keeps crashing” should be `bug`, not `shipping`. The classifier now checks software-failure terms before shipping keywords; the expected category is `bug`.
- Classification rules now give app crashes, software errors, and login failures priority over billing keywords. This ensures a crash ticket mentioning payment, including “no payment was made,” is categorized as `bug`.
- T-015 exposed an overly specific bug reply: it referred to an app crash even though the ticket reports a website error. Bug replies now acknowledge technical issues generally and request relevant reproduction/device/browser/version details without assuming the affected product.
- Invoice requests now receive an invoice-specific draft rather than the generic billing response. Technical issue replies explicitly acknowledge that the app is crashing or not working as expected.
- Refund replies explain eligibility without suggesting that a refund has already been completed.
- Reviewed `T-013`: its deterministic test adapter deliberately repeats the classification request. The loop now suppresses duplicate executions while continuing to enforce the five-round cap and produce the safe handoff.
- Ticket `T-012` exercises the required multi-tool path: order lookup followed by refund-eligibility lookup before drafting its reply.
- T-014 is classified as billing, then `ORD-1007` is looked up. Because its mock order record says payment review is required, the agent calls `escalate_to_human(reason)`. The tool records the request in-process and returns `recorded: true`; the draft claims a request was recorded only when that result is present. No external person or service is contacted, and the ticket is not treated as a refund request.
- T-015 uses the software-error rule and should be classified as `bug`.

## 15-ticket test matrix (verified)

The table below was verified by running `python main.py`: all 15 tickets produced the category and tool path shown.

| Ticket | Expected category | Expected tools |
| --- | --- | --- |
| T-001 | shipping | `classify_ticket_category`, `lookup_order` |
| T-002 | billing | `classify_ticket_category`, `lookup_order`, `check_refund_eligibility` |
| T-003 | bug | `classify_ticket_category` |
| T-004 | billing | `classify_ticket_category` |
| T-005 | shipping | `classify_ticket_category`, `lookup_order` |
| T-006 | billing | `classify_ticket_category`, `lookup_order`, `check_refund_eligibility` |
| T-007 | bug | `classify_ticket_category` |
| T-008 | billing | `classify_ticket_category` (invoice-specific draft) |
| T-009 | shipping | `classify_ticket_category`, `lookup_order` |
| T-010 | bug | `classify_ticket_category` |
| T-011 | other | `classify_ticket_category` |
| T-012 | billing | `classify_ticket_category`, `lookup_order`, `check_refund_eligibility` |
| T-013 | bug | `classify_ticket_category` (repeated request; cap should be reached) |
| T-014 | billing | `classify_ticket_category`, `lookup_order`, `escalate_to_human` |
| T-015 | bug | `classify_ticket_category` |

Refund eligibility is checked for T-002, T-006, and T-012. Eligible wording says support will review the request; it does not claim a refund was issued. T-014 produces one escalation record (`recorded: true`). T-013 hits the five-round cap and returns the safe handoff reply. Remaining tool errors: 0.

This starter workspace has no model SDK, credentials, or network access, so `LocalTriageModel` is a deterministic offline stand-in. It follows the same tool-call/final-response protocol and makes the workflow reproducible. Replace its `respond` implementation with your model provider's tool-calling API adapter to use a real LLM; keep the `run_agent` loop and tool interface.

## Safety-cap and reviewed classification case

Ticket `T-013` is configured to request the classification tool on every turn; a run should show the five-round cap, skipped duplicate requests, and safe fallback. Ticket `T-007` describes a tracking page that crashes. The initial keyword ordering would have marked it as shipping; bug keywords now take priority so its expected category is `bug`.

## Project layout

```
main.py         agent loop, system prompt, offline model adapter, coloured console output
tool.py         mock data + tools (classify, lookup, refund check, escalate)
tickets.json    15 synthetic support tickets
assets/         output_preview.png (screenshot used in this README)
```

## How to run

```bash
python main.py            # tool-call log per ticket, then the table and Run Summary
python main.py --quiet    # table and Run Summary only
```

Output is a coloured table (#, Ticket ID, Category, Tools Used, Draft Reply preview) followed by a Run Summary: tickets processed, category counts, multi-tool tickets, safety-cap hits, escalations and tool errors. No packages need installing (Python 3.8+).
