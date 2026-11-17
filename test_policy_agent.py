import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import main


class PolicyRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = {chunk["source"]: chunk["text"] for chunk in main.POLICY_CHUNKS}

    def rule(self, ticket, filename):
        return main.select_policy_rule(ticket, self.documents[filename])

    def test_refund_within_30_calendar_days(self):
        self.assertIn("30 calendar days", self.rule(
            "What is the refund deadline?", "refund_policy.txt"))

    def test_refund_processing_time(self):
        self.assertIn("5 to 10 business days", self.rule(
            "How long is refund processing?", "refund_policy.txt"))

    def test_shipping_processing_time(self):
        self.assertIn("1 to 2 business days", self.rule(
            "What is the shipping processing time?", "shipping_policy.txt"))

    def test_standard_shipping_time(self):
        self.assertIn("3 to 5 business days", self.rule(
            "What is the shipping delivery timeframe?", "shipping_policy.txt"))

    def test_delayed_shipment_support_rule(self):
        self.assertIn("7 business days after dispatch", self.rule(
            "My shipment is delayed; when should I contact support?", "shipping_policy.txt"))

    def test_cancellation_while_processing(self):
        self.assertIn("while an order is still being processed", self.rule(
            "Can I cancel while processing?", "cancellation_policy.txt"))

    def test_cancellation_after_dispatch(self):
        self.assertIn("Once an order has entered fulfillment", self.rule(
            "Can I cancel after it was dispatched?", "cancellation_policy.txt"))

    def test_final_sale_refund_eligibility(self):
        self.assertIn("Final-sale items are not eligible", self.rule(
            "Can I get a refund for a final-sale item?", "refund_policy.txt"))

    def test_defective_item_shipping_charge_exception(self):
        self.assertIn("unless the item arrived defective or the wrong item was shipped",
                      self.rule("Are shipping charges refundable for a defective item?",
                                "refund_policy.txt"))

    def test_safety_issue_escalation(self):
        self.assertIn("Escalate a ticket to a supervisor", self.rule(
            "The customer reports a safety issue.", "escalation_policy.txt"))

    def test_fraud_allegation_escalation(self):
        self.assertIn("Escalate a ticket to a supervisor", self.rule(
            "The customer alleges fraud.", "escalation_policy.txt"))

    def test_unsupported_price_matching_is_not_invented(self):
        ticket = "Do you offer price matching for the same item?"
        self.assertEqual([], main.search_policy_docs(ticket))
        self.assertIn("couldn't confirm", main.draft_reply(ticket, []))
        self.assertIn("Source: None", main.draft_reply(ticket, []))

    def test_empty_query_validation(self):
        for query in ("", "   ", None, 12):
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    main.search_policy_docs(query)

    def test_invalid_top_k_validation(self):
        for top_k in (0, -1, 1.5, "2", True, main.MAX_TOP_K + 1):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    main.search_policy_docs("refund deadline", top_k=top_k)

    def test_missing_and_corrupted_embedding_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = os.path.join(directory, "embedding_cache.json")
            self.assertEqual({}, main.load_embedding_cache(cache_path))
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                cache_file.write("{broken json")
            self.assertEqual({}, main.load_embedding_cache(cache_path))

    def test_missing_policy_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            absent = os.path.join(directory, "missing")
            self.assertEqual([], main.load_policy_chunks(absent))

    def test_embedding_cache_miss_then_hit(self):
        original_cache = main.EMBEDDING_CACHE
        original_stats = main.CACHE_STATS
        original_save = main.save_embedding_cache
        try:
            main.EMBEDDING_CACHE = {}
            main.CACHE_STATS = {"HIT": 0, "MISS": 0}
            main.save_embedding_cache = lambda cache: True
            first = main.cached_embedding("cache verification text", "test")
            second = main.cached_embedding("cache verification text", "test")
            self.assertEqual(first, second)
            self.assertEqual({"HIT": 1, "MISS": 1}, main.CACHE_STATS)
        finally:
            main.EMBEDDING_CACHE = original_cache
            main.CACHE_STATS = original_stats
            main.save_embedding_cache = original_save

    def test_search_returns_policy_source_and_citation(self):
        results = main.search_policy_docs("What is my refund deadline?")
        self.assertTrue(results)
        reply = main.draft_reply("What is my refund deadline?", results)
        self.assertIn("30 calendar days", reply)
        self.assertIn("Source: refund_policy.txt#chunk-", reply)

    def test_shipping_draft_includes_support_trigger(self):
        ticket = "My package was dispatched four business days ago. What is the shipping delivery timeframe?"
        results = main.search_policy_docs(ticket)
        reply = main.draft_reply(ticket, results)
        self.assertIn("3 to 5 business days", reply)
        self.assertIn("7 business days after dispatch", reply)
        self.assertIn("Source: shipping_policy.txt#chunk-", reply)

    def test_tool_registry_and_argument_validation(self):
        result = main.call_tool("search_policy_docs", {"query": "refund deadline"})
        self.assertTrue(result)
        with self.assertRaises(ValueError):
            main.call_tool("not_a_tool", {})
        with self.assertRaises(ValueError):
            main.execute_tool_calls([{"name": "search_policy_docs"}])


class AgentLoopTests(unittest.TestCase):
    def run_quiet(self, ticket_id, ticket):
        return main.run_agent(ticket_id, ticket, verbose=False)

    def test_refund_ticket_cites_real_refund_rule(self):
        outcome = self.run_quiet("T-101", dict(main.SAMPLE_TICKETS)["T-101"])
        self.assertIn("30 calendar days", outcome["reply"])
        self.assertEqual(["refund_policy.txt#chunk-1"], outcome["chunks_used"])

    def test_retrieval_is_called_before_reply_for_policy_tickets(self):
        policy_ids = ("T-101", "T-102", "T-104", "T-105", "T-106", "T-107")
        for ticket_id in policy_ids:
            with self.subTest(ticket=ticket_id):
                outcome = self.run_quiet(ticket_id, dict(main.SAMPLE_TICKETS)[ticket_id])
                names = [name for _, name, _, _ in outcome["trace"]]
                self.assertIn("search_policy_docs", names)
                self.assertTrue(outcome["chunks_used"])

    def test_uncovered_policy_says_cannot_confirm(self):
        outcome = self.run_quiet("T-103", dict(main.SAMPLE_TICKETS)["T-103"])
        self.assertTrue(outcome["retrieval_called"])
        self.assertEqual([], outcome["chunks_used"])
        self.assertIn("couldn't confirm", outcome["reply"])
        self.assertNotIn("price match", outcome["reply"].lower().replace("price matching", ""))

    def test_non_policy_ticket_does_not_force_retrieval(self):
        outcome = self.run_quiet("T-108", dict(main.SAMPLE_TICKETS)["T-108"])
        self.assertFalse(outcome["retrieval_called"])

    def test_week2_style_tools_run_alongside_retrieval(self):
        outcome = self.run_quiet("T-106", dict(main.SAMPLE_TICKETS)["T-106"])
        names = [name for _, name, _, _ in outcome["trace"]]
        self.assertEqual(["classify_ticket", "lookup_order", "search_policy_docs"], names)

    def test_system_prompt_requires_search_before_policy_reply(self):
        self.assertIn("MUST call", main.SYSTEM_PROMPT)
        self.assertIn("search_policy_docs", main.SYSTEM_PROMPT)

    def test_second_run_reuses_cache_instead_of_reembedding(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as directory:
            shutil.copy(os.path.join(root, "main.py"), directory)
            shutil.copytree(os.path.join(root, "policy_docs"), os.path.join(directory, "policy_docs"))
            outputs = []
            for _ in range(2):
                done = subprocess.run([sys.executable, "main.py"], cwd=directory,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      universal_newlines=True, check=True)
                outputs.append(done.stdout)
            self.assertIn("0 HIT, 4 MISS", outputs[0])
            self.assertIn("4 HIT, 0 MISS", outputs[1])


if __name__ == "__main__":
    unittest.main()
