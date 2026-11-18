import json
import os
import tempfile
import time
import unittest

import app as app_module
import evaluate
import main
import triage

POLICY_TICKET = {"subject": "Refund deadline",
                 "body": "I received my order last week and want a refund. What is the refund deadline?"}


class ApiBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.tmp.name, "requests.log")
        self._old_log, triage.LOG_FILE = triage.LOG_FILE, self.log_path
        self.client = app_module.create_app(rate_limit=1000).test_client()

    def tearDown(self):
        triage.LOG_FILE = self._old_log
        self.tmp.cleanup()


class TriageEndpointTests(ApiBase):
    def test_valid_ticket_returns_category_reply_and_tool_trace(self):
        response = self.client.post("/triage", json=POLICY_TICKET)
        self.assertEqual(200, response.status_code)
        data = response.get_json()
        self.assertEqual("refund", data["category"])
        self.assertEqual("answered", data["outcome"])
        self.assertIn("30 calendar days", data["reply"])
        self.assertEqual(["refund_policy.txt#chunk-1"], data["policy_chunks_used"])
        self.assertEqual(["classify_ticket", "search_policy_docs"], data["tools_used"])
        self.assertEqual("search_policy_docs", data["tool_trace"][1]["tool"])
        self.assertIn("latency_ms", data)

    def test_empty_body_is_a_clean_400_not_a_crash(self):
        for kwargs in ({}, {"data": ""}, {"data": "not json", "content_type": "application/json"},
                       {"data": "{broken", "content_type": "application/json"}):
            with self.subTest(kwargs=kwargs):
                response = self.client.post("/triage", **kwargs)
                self.assertEqual(400, response.status_code)
                self.assertEqual("invalid_json", response.get_json()["error"]["code"])

    def test_empty_subject_or_body_is_rejected_with_422(self):
        for payload in ({"subject": "", "body": "hi"}, {"subject": "hi", "body": "   "},
                        {"subject": "hi"}, {"body": "hi"}, {"subject": 5, "body": "x"},
                        {"subject": "x", "body": "y", "ticket_id": ""}):
            with self.subTest(payload=payload):
                response = self.client.post("/triage", json=payload)
                self.assertEqual(422, response.status_code)
                self.assertEqual("invalid_ticket", response.get_json()["error"]["code"])

    def test_json_array_body_is_rejected(self):
        self.assertEqual(422, self.client.post("/triage", json=[1, 2]).status_code)

    def test_oversized_field_is_rejected(self):
        response = self.client.post("/triage", json={"subject": "a", "body": "b" * 6000})
        self.assertEqual(422, response.status_code)

    def test_unknown_route_and_wrong_method_return_json(self):
        self.assertEqual("not_found", self.client.get("/nope").get_json()["error"]["code"])
        self.assertEqual("method_not_allowed", self.client.get("/triage").get_json()["error"]["code"])

    def test_agent_failure_returns_502_json(self):
        original = main.run_agent
        main.run_agent = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tool exploded"))
        try:
            response = self.client.post("/triage", json=POLICY_TICKET)
        finally:
            main.run_agent = original
        self.assertEqual(502, response.status_code)
        self.assertEqual("agent_failure", response.get_json()["error"]["code"])
        self.assertNotIn("Traceback", response.get_data(as_text=True))

    def test_agent_timeout_returns_504_json(self):
        original, original_timeout = main.run_agent, triage.AGENT_TIMEOUT_S
        main.run_agent = lambda *a, **k: time.sleep(0.4)
        triage.AGENT_TIMEOUT_S = 0.05
        try:
            response = self.client.post("/triage", json=POLICY_TICKET)
        finally:
            main.run_agent, triage.AGENT_TIMEOUT_S = original, original_timeout
        self.assertEqual(504, response.status_code)
        self.assertEqual("agent_timeout", response.get_json()["error"]["code"])

    def test_unexpected_exception_is_json_500_without_stack_trace(self):
        original = triage.triage_ticket
        triage.triage_ticket = lambda *a, **k: 1 / 0
        try:
            response = self.client.post("/triage", json=POLICY_TICKET)
        finally:
            triage.triage_ticket = original
        self.assertEqual(500, response.status_code)
        self.assertEqual("internal_error", response.get_json()["error"]["code"])
        self.assertNotIn("ZeroDivision", response.get_data(as_text=True))

    def test_rate_limit_returns_429_with_retry_after(self):
        client = app_module.create_app(rate_limit=2).test_client()
        codes = [client.post("/triage", json=POLICY_TICKET).status_code for _ in range(3)]
        self.assertEqual([200, 200, 429], codes)
        limited = client.post("/triage", json=POLICY_TICKET)
        self.assertIn("Retry-After", limited.headers)

    def test_health(self):
        self.assertEqual("ok", self.client.get("/health").get_json()["status"])


class GuardrailTests(ApiBase):
    def test_ungrounded_policy_question_is_escalated_not_guessed(self):
        data = self.client.post("/triage", json={
            "subject": "Price match", "body": "Do you offer price matching if I find it cheaper elsewhere?"}).get_json()
        self.assertEqual("escalate_to_human", data["outcome"])
        self.assertEqual([], data["policy_chunks_used"])
        self.assertNotIn("Our policy states", data["reply"])

    def test_low_confidence_ticket_is_escalated(self):
        data = self.client.post("/triage", json={
            "subject": "Question", "body": "Hi, I have a question about my account."}).get_json()
        self.assertEqual("escalate_to_human", data["outcome"])
        self.assertEqual("low_category_confidence", data["escalation_reason"])

    def test_policy_ticket_with_zero_tool_calls_is_forced_to_escalate(self):
        lazy_model = lambda messages, tools, system_prompt: {
            "content": "Refunds are always available for 90 days!", "tool_calls": []}
        result = triage.triage_ticket(POLICY_TICKET["subject"], POLICY_TICKET["body"], model=lazy_model)
        self.assertEqual("escalate_to_human", result["outcome"])
        self.assertEqual("policy_ticket_without_policy_lookup", result["escalation_reason"])
        self.assertNotIn("90 days", result["reply"])

    def test_escalation_category_goes_to_supervisor(self):
        data = self.client.post("/triage", json={
            "subject": "Sparks", "body": "The charger overheated and sparked. This is a safety issue."}).get_json()
        self.assertEqual("escalate_to_human", data["outcome"])
        self.assertEqual(["escalation_policy.txt#chunk-1"], data["policy_chunks_used"])


class LoggingAndBatchTests(ApiBase):
    def test_every_request_is_logged_with_id_category_tools_timestamp(self):
        self.client.post("/triage", json=dict(POLICY_TICKET, ticket_id="TKT-LOG1"))
        with open(self.log_path, encoding="utf-8") as handle:
            entry = json.loads(handle.readlines()[-1])
        self.assertEqual("TKT-LOG1", entry["ticket_id"])
        self.assertEqual("refund", entry["category"])
        self.assertEqual(["classify_ticket", "search_policy_docs"], entry["tools_called"])
        self.assertTrue(entry["timestamp"].endswith("Z"))

    def test_log_write_failure_never_breaks_triage(self):
        triage.LOG_FILE = os.path.join(self.tmp.name, "missing-dir", "x.log")
        self.assertEqual(200, self.client.post("/triage", json=POLICY_TICKET).status_code)

    def test_batch_returns_timing_and_isolates_bad_tickets(self):
        data = self.client.post("/triage/batch", json={"tickets": [
            POLICY_TICKET, {"subject": "", "body": ""}, {"subject": "Thanks", "body": "Thanks, great service!"}]}).get_json()
        self.assertEqual(3, data["count"])
        self.assertEqual(2, data["succeeded"])
        self.assertEqual([200, 422, 200], [r["status"] for r in data["results"]])
        self.assertIn("total_latency_ms", data)
        self.assertTrue(all("elapsed_ms" in r for r in data["results"]))

    def test_batch_validation(self):
        self.assertEqual(400, self.client.post("/triage/batch", json={"tickets": []}).status_code)
        self.assertEqual(413, self.client.post("/triage/batch", json=[POLICY_TICKET] * 51).status_code)


class WarehouseCancellationTests(ApiBase):
    """H08: 'hasn't left the warehouse' must get the conditional rule, never a bare 'yes'."""

    def ask(self, body):
        return self.client.post("/triage", json={"subject": "Cancel", "body": body}).get_json()

    def test_warehouse_cancellation_is_answered_conditionally_with_citation(self):
        data = self.ask("Is it possible to cancel my order right now? It hasn't left the warehouse.")
        self.assertEqual("answered", data["outcome"])
        self.assertEqual(["cancellation_policy.txt#chunk-1"], data["policy_chunks_used"])
        self.assertIn("only if it has not entered fulfillment", data["reply"])
        self.assertIn("does not by itself make an order cancellable", data["reply"])

    def test_reply_never_promises_unconditional_cancellation(self):
        reply = self.ask("It hasn't left the warehouse, can I cancel?")["reply"].lower()
        self.assertNotIn("yes", reply.split())
        self.assertNotIn("you can cancel", reply)

    def test_after_dispatch_cancellation_rule_is_unchanged(self):
        reply = self.ask("My order was already dispatched. Can I still cancel it?")["reply"]
        self.assertIn("cannot be cancelled", reply)

    def test_policy_document_states_the_conditional_rule(self):
        with open(os.path.join(os.path.dirname(main.__file__), "policy_docs", "cancellation_policy.txt")) as handle:
            text = handle.read()
        self.assertIn("not yet left the warehouse", text)
        self.assertIn("only if it has not entered fulfillment", text)

    def test_prompt_and_tool_description_carry_the_stage_vocabulary_rules(self):
        self.assertIn("Never turn", main.SYSTEM_PROMPT)
        self.assertIn("fulfillment", main.SYSTEM_PROMPT)
        search_tool = [t for t in main.TOOL_SCHEMAS if t["name"] == "search_policy_docs"][0]
        self.assertIn("fulfillment", search_tool["description"])
        self.assertEqual(["fulfillment"], main.stage_terms("It is in the warehouse"))


class EvalSetTests(unittest.TestCase):
    def test_eval_set_has_at_least_ten_well_formed_tickets(self):
        with open(evaluate.EVAL_FILE, encoding="utf-8") as handle:
            tickets = json.load(handle)
        self.assertGreaterEqual(len(tickets), 10)
        for ticket in tickets:
            for key in ("id", "subject", "body", "expected_category", "expected_outcome"):
                self.assertIn(key, ticket)
        self.assertEqual(len(tickets), len(set(t["id"] for t in tickets)))

    def test_check_detects_wrong_category_and_missing_citation(self):
        ticket = {"expected_category": "refund", "expected_outcome": "answered",
                  "expected_policy": "refund_policy.txt", "expected_tools": ["search_policy_docs"]}
        bad = {"category": "shipping", "outcome": "answered", "policy_chunks_used": [],
               "tools_used": ["classify_ticket"], "reply": "x"}
        checks = evaluate.check(ticket, bad)
        self.assertFalse(checks["category"][0])
        self.assertFalse(checks["policy"][0])
        self.assertFalse(checks["tools"][0])

    def test_dev_set_passes_through_the_real_triage_function(self):
        triage_log, triage.LOG_FILE = triage.LOG_FILE, None
        try:
            with open(evaluate.EVAL_FILE, encoding="utf-8") as handle:
                dev = [t for t in json.load(handle) if t.get("split") == "dev"]
            for ticket in dev:
                result = triage.triage_ticket(ticket["subject"], ticket["body"])
                failed = [n for n, (ok, _) in evaluate.check(ticket, result).items() if not ok]
                self.assertEqual([], failed, ticket["id"])
        finally:
            triage.LOG_FILE = triage_log


if __name__ == "__main__":
    unittest.main()
