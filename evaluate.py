"""Evaluate the triage agent against eval_tickets.json.

    python evaluate.py                    # calls triage.triage_ticket() directly
    python evaluate.py --api http://127.0.0.1:8000   # same tickets through POST /triage
    python evaluate.py -v                 # also print the reply of every FAILING ticket

A ticket passes only if ALL checks pass:
  category   - predicted category == expected
  outcome    - 'answered' vs 'escalate_to_human' == expected
  policy     - the expected policy document is cited (or, if none is expected, no chunk is cited)
  rule       - the reply contains the exact rule text expected ("expected_phrase"), so citing the
               right FILE is not enough: the right RULE must be stated
  tools      - every expected tool was called (and search_policy_docs was NOT called when
               the ticket needs no retrieval)
Exit code is 0 only when accuracy is 100%, so it can gate CI.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = os.path.join(HERE, "eval_tickets.json")


def local_runner():
    import triage
    triage.LOG_FILE = None  # keep eval runs out of the request log

    def run(ticket):
        return triage.triage_ticket(ticket["subject"], ticket["body"], "EVAL-" + ticket["id"])
    return run


def api_runner(base_url):
    def run(ticket):
        data = json.dumps({"subject": ticket["subject"], "body": ticket["body"],
                           "ticket_id": "EVAL-" + ticket["id"]}).encode("utf-8")
        request = urllib.request.Request(base_url.rstrip("/") + "/triage", data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError("HTTP {}: {}".format(error.code, error.read().decode("utf-8")[:200]))
    return run


def check(ticket, result):
    """Return {check_name: (passed, detail)} for one ticket."""
    cited = result["policy_chunks_used"]
    expected_policy = ticket.get("expected_policy")
    if expected_policy:
        policy_ok = any(chunk.startswith(expected_policy) for chunk in cited)
    else:
        policy_ok = not cited
    phrase = ticket.get("expected_phrase")
    rule_ok = (phrase.lower() in result["reply"].lower()) if phrase else True
    tools = result["tools_used"]
    tools_ok = all(name in tools for name in ticket.get("expected_tools", []))
    if ticket.get("expect_no_retrieval"):
        tools_ok = tools_ok and "search_policy_docs" not in tools
    return {
        "category": (result["category"] == ticket["expected_category"],
                     "{} (expected {})".format(result["category"], ticket["expected_category"])),
        "outcome": (result["outcome"] == ticket["expected_outcome"],
                    "{} (expected {})".format(result["outcome"], ticket["expected_outcome"])),
        "policy": (policy_ok, "cited {} (expected {})".format(cited or "nothing", expected_policy or "no citation")),
        "rule": (rule_ok, "reply {} '{}'".format("contains" if rule_ok else "is missing", phrase)),
        "tools": (tools_ok, "called {}".format(tools)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", help="base URL of a running API (default: call the agent directly)")
    parser.add_argument("--file", default=EVAL_FILE)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    with open(args.file, encoding="utf-8") as handle:
        tickets = json.load(handle)
    run = api_runner(args.api) if args.api else local_runner()

    totals = {"category": 0, "outcome": 0, "policy": 0, "rule": 0, "tools": 0}
    passed = 0
    print("{:<5}{:<20}{:<14}{:<19}{:<7}{:<6}{:<7}{}".format(
        "ID", "Subject", "Category", "Outcome", "Policy", "Rule", "Tools", "Result"))
    print("-" * 88)
    failures = []
    for ticket in tickets:
        try:
            result = run(ticket)
            checks = check(ticket, result)
        except Exception as error:  # an eval must report a crash as a failure, not die
            result, checks = None, {name: (False, "error: {}".format(error)) for name in totals}
        ok = all(flag for flag, _ in checks.values())
        passed += ok
        for name, (flag, _) in checks.items():
            totals[name] += flag
        mark = lambda name: "ok" if checks[name][0] else "FAIL"
        print("{:<5}{:<20}{:<14}{:<19}{:<7}{:<6}{:<7}{}".format(
            ticket["id"], ticket["subject"][:18],
            (result["category"] if result else "-"), (result["outcome"] if result else "-"),
            mark("policy"), mark("rule"), mark("tools"), "PASS" if ok else "FAIL"))
        if not ok:
            failures.append((ticket, result, checks))

    count = len(tickets)
    print("-" * 88)
    for name in ("category", "outcome", "policy", "rule", "tools"):
        print("{:<22}{}/{}  ({:.0f}%)".format(name + " accuracy:", totals[name], count, 100.0 * totals[name] / count))
    print("OVERALL ACCURACY: {}/{} = {:.1f}%".format(passed, count, 100.0 * passed / count))
    for split in sorted(set(t.get("split", "dev") for t in tickets)):
        group = [t["id"] for t in tickets if t.get("split", "dev") == split]
        bad = set(t["id"] for t, _, _ in failures)
        good = len([i for i in group if i not in bad])
        print("  {:<18}{}/{} = {:.1f}%".format(split + ":", good, len(group), 100.0 * good / len(group)))
    for ticket, result, checks in failures:
        print("\nFAILED {} - {}".format(ticket["id"], ticket["body"]))
        for name, (flag, detail) in checks.items():
            if not flag:
                print("   {:<9}{}".format(name, detail))
        if args.verbose and result:
            print("   reply:   {}".format(result["reply"]))
            print("   reason:  {}".format(result.get("escalation_reason")))
    return 0 if passed == count else 1


if __name__ == "__main__":
    sys.exit(main())
