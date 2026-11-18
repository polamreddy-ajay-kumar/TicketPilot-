"""Thin Flask API around the triage agent.

    POST /triage          one ticket  -> category, reply, outcome, tool trace
    POST /triage/batch    many tickets -> results + total/per-ticket latency (stretch goal)
    GET  /health

The API only validates input and shapes errors; all triage logic lives in triage.py
(the same function evaluate.py calls), so the eval tests the real code path.
"""
import collections
import os
import threading
import time

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

import triage

MAX_BATCH = 50
RATE_LIMIT_PER_MIN = int(os.environ.get("TRIAGE_RATE_LIMIT_PER_MIN", "60"))


class RateLimiter(object):
    """Sliding-window limiter, per client IP, in memory (fine for one process)."""

    def __init__(self, limit, window_s=60.0):
        self.limit, self.window_s = limit, window_s
        self._hits = collections.defaultdict(collections.deque)
        self._lock = threading.Lock()

    def check(self, client):
        """Return 0 if allowed, otherwise the number of seconds to wait."""
        now = time.time()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, int(self.window_s - (now - hits[0])) + 1)
            hits.append(now)
            return 0


def error(status, code, message, details=None):
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return jsonify(body), status


def create_app(rate_limit=RATE_LIMIT_PER_MIN):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
    limiter = RateLimiter(rate_limit)

    @app.before_request
    def apply_rate_limit():
        if request.method == "POST":
            wait = limiter.check(request.remote_addr or "unknown")
            if wait:
                response, status = error(429, "rate_limited",
                                         "too many requests; retry in {}s".format(wait))
                response.headers["Retry-After"] = str(wait)
                return response, status

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "policy_chunks": len(triage.main.POLICY_CHUNKS)})

    def run_one(payload):
        """Validate + triage one ticket. Returns (status_code, body_dict)."""
        clean, problems = triage.validate_ticket(payload)
        if problems:
            return 422, {"error": {"code": "invalid_ticket",
                                   "message": "ticket failed validation", "details": problems}}
        try:
            return 200, triage.triage_ticket(clean["subject"], clean["body"], clean.get("ticket_id"))
        except triage.AgentTimeout as exc:
            return 504, {"error": {"code": "agent_timeout", "message": str(exc)}}
        except triage.AgentFailure as exc:
            return 502, {"error": {"code": "agent_failure",
                                   "message": "the triage agent failed; ticket was not answered",
                                   "details": [{"message": str(exc)}]}}

    @app.post("/triage")
    def triage_endpoint():
        payload = request.get_json(silent=True)
        if payload is None:
            return error(400, "invalid_json", "request body must be a non-empty JSON object "
                         "with 'subject' and 'body' (send Content-Type: application/json)")
        status, body = run_one(payload)
        return jsonify(body), status

    @app.post("/triage/batch")
    def triage_batch():
        payload = request.get_json(silent=True)
        tickets = payload.get("tickets") if isinstance(payload, dict) else payload
        if not isinstance(tickets, list) or not tickets:
            return error(400, "invalid_json", "send a JSON list of tickets, or {\"tickets\": [...]}")
        if len(tickets) > MAX_BATCH:
            return error(413, "batch_too_large", "at most {} tickets per batch".format(MAX_BATCH))
        started = time.time()
        results = []
        for index, ticket in enumerate(tickets):
            t0 = time.time()
            status, body = run_one(ticket)
            results.append({"index": index, "status": status, "result": body,
                            "elapsed_ms": round((time.time() - t0) * 1000, 1)})
        return jsonify({
            "count": len(results),
            "succeeded": sum(1 for r in results if r["status"] == 200),
            "total_latency_ms": round((time.time() - started) * 1000, 1),
            "results": results,
        })

    @app.errorhandler(404)
    def not_found(_):
        return error(404, "not_found", "no such endpoint")

    @app.errorhandler(405)
    def method_not_allowed(_):
        return error(405, "method_not_allowed", "method not allowed for this endpoint")

    @app.errorhandler(413)
    def too_large(_):
        return error(413, "payload_too_large", "request body is too large")

    @app.errorhandler(Exception)
    def unexpected(exc):
        # Last-resort net: never leak a stack trace or an HTML 500 page.
        if isinstance(exc, HTTPException):
            return error(exc.code or 400, exc.name.lower().replace(" ", "_"), exc.description)
        app.logger.exception("unhandled error")
        return error(500, "internal_error", "unexpected server error; the ticket was not processed")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
