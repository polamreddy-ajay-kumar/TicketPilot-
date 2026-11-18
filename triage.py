"""Triage service layer: runs the real agent, applies guardrails, logs every request.

The Flask API (app.py) and the eval script (evaluate.py) both call `triage_ticket`,
so the evaluation exercises the same code path as production requests.
"""
import concurrent.futures
import datetime
import json
import os
import threading
import time
import uuid

import main

LOW_CONFIDENCE = 0.5          # below this the agent must not guess
AGENT_TIMEOUT_S = 10          # hard limit for one agent run
LOG_FILE = os.environ.get("TRIAGE_LOG_FILE", os.path.join(main.BASE_DIR, "triage_requests.log"))
MAX_FIELD_CHARS = 5000

HANDOFF_REPLY = ("Thanks for contacting us. I can't answer this reliably, so I've passed your "
                 "ticket to a human agent who will follow up.")
UNGROUNDED_REPLY = ("I couldn't confirm that policy from the company documents available to me. "
                    "I've escalated your ticket to a human agent who can confirm it. Source: None.")
SUPERVISOR_REPLY = ("Thank you for flagging this. Because it involves a safety, fraud, legal or "
                    "exception request, I've escalated your ticket to a supervisor.")

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_log_lock = threading.Lock()


class AgentTimeout(Exception):
    pass


class AgentFailure(Exception):
    pass


def validate_ticket(payload):
    """Return (clean_ticket, errors). Rejects empty/non-string subject and body."""
    errors = []
    if not isinstance(payload, dict):
        return None, [{"field": "body", "message": "request body must be a JSON object"}]
    clean = {}
    for field in ("subject", "body"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append({"field": field, "message": "{} is required and must be a non-empty string".format(field)})
        elif len(value) > MAX_FIELD_CHARS:
            errors.append({"field": field, "message": "{} must be at most {} characters".format(field, MAX_FIELD_CHARS)})
        else:
            clean[field] = value.strip()
    ticket_id = payload.get("ticket_id")
    if ticket_id is not None and (not isinstance(ticket_id, str) or not ticket_id.strip() or len(ticket_id) > 64):
        errors.append({"field": "ticket_id", "message": "ticket_id must be a non-empty string of at most 64 characters"})
    elif ticket_id:
        clean["ticket_id"] = ticket_id.strip()
    return (None, errors) if errors else (clean, [])


def _summarize(name, result):
    if name == "search_policy_docs" and isinstance(result, list):
        return [{"chunk_id": r["chunk_id"], "score": r["score"]} for r in result]
    return result


def _guardrail(ticket, names, classification, agent):
    """Return (escalate: bool, reason or None). Order matters: the first rule that fires wins."""
    policy_related = main.needs_policy_lookup(ticket)
    if policy_related and "search_policy_docs" not in names:
        return True, "policy_ticket_without_policy_lookup"
    if classification["confidence"] < LOW_CONFIDENCE:
        return True, "low_category_confidence"
    if policy_related and not agent["chunks_used"]:
        return True, "ungrounded_no_policy_support"
    if classification["category"] == "escalation":
        return True, "escalation_policy_requires_supervisor"
    return False, None


def triage_ticket(subject, body, ticket_id=None, model=None):
    """Run one ticket through the agent + guardrails. Raises AgentTimeout / AgentFailure."""
    started = time.time()
    ticket_id = ticket_id or "TKT-" + uuid.uuid4().hex[:8]
    ticket = "{}. {}".format(subject.strip(), body.strip())
    future = _executor.submit(main.run_agent, ticket_id, ticket, model, False)
    try:
        agent = future.result(timeout=AGENT_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        future.cancel()
        _log(ticket_id, None, None, "error:timeout", [], started)
        raise AgentTimeout("agent did not finish within {}s".format(AGENT_TIMEOUT_S))
    except Exception as error:  # a tool/model failure must never become a 500 crash
        _log(ticket_id, None, None, "error:agent_failure", [], started)
        raise AgentFailure("{}: {}".format(type(error).__name__, error))

    names = [name for _, name, _, _ in agent["trace"]]
    classification = next((r for _, n, _, r in agent["trace"]
                           if n == "classify_ticket" and isinstance(r, dict) and "confidence" in r), None)
    if classification is None:  # model skipped the classifier: classify here, deterministically
        classification = main.classify_ticket(ticket)
    escalate, reason = _guardrail(ticket, names, classification, agent)

    reply, chunks_used = agent["reply"], agent["chunks_used"]
    if escalate:
        if reason == "ungrounded_no_policy_support":
            reply = UNGROUNDED_REPLY
        elif reason == "escalation_policy_requires_supervisor":
            reply = SUPERVISOR_REPLY
        else:
            reply, chunks_used = HANDOFF_REPLY, []
    outcome = "escalate_to_human" if escalate else "answered"
    result = {
        "ticket_id": ticket_id,
        "category": classification["category"],
        "priority": classification["priority"],
        "confidence": classification["confidence"],
        "outcome": outcome,
        "escalation_reason": reason,
        "reply": reply,
        "policy_chunks_used": chunks_used,
        "policy_chunks_retrieved": agent["retrieved"],
        "tools_used": names,
        "tool_trace": [{"step": step, "tool": name, "arguments": _short_args(args),
                        "result": _summarize(name, res)} for step, name, args, res in agent["trace"]],
        "latency_ms": round((time.time() - started) * 1000, 1),
    }
    _log(ticket_id, classification["category"], classification["confidence"], outcome, names, started, reason)
    return result


def _short_args(args):
    return {k: ("<ticket text>" if k in ("ticket", "query") else v) for k, v in args.items()}


def _log(ticket_id, category, confidence, outcome, tools, started, reason=None):
    """Append one JSON line per request: id, category, tools called, timestamp (+ outcome/latency)."""
    if not LOG_FILE:
        return
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "ticket_id": ticket_id, "category": category, "confidence": confidence,
        "outcome": outcome, "escalation_reason": reason, "tools_called": tools,
        "latency_ms": round((time.time() - started) * 1000, 1),
    }
    try:
        with _log_lock, open(LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError as error:  # logging problems must never break triage
        print("Warning: could not write request log: {}".format(error))
