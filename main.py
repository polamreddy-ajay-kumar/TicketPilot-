"""TicketPilot: support-ticket triage agent with a multi-round tool-calling loop."""
import json
import os
import sys

from tool import TOOLS

MAX_TOOL_ROUNDS = 5
VERBOSE = "--quiet" not in sys.argv

if os.name == "nt":
    os.system("")  # enable ANSI colours in Windows terminals

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
CYAN, GREEN, YELLOW, RED, MAGENTA, BLUE = (
    "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[35m", "\033[34m",
)


def log(message):
    """Tool-call log (arguments + result). Hidden with --quiet."""
    if not VERBOSE:
        return
    colour = YELLOW if "WARNING" in message else DIM
    print(colour + message + RESET)


SYSTEM_PROMPT = """You are a support-ticket triage agent. Classify each ticket, gather relevant context using tools, then draft a short, polite reply. Use only facts returned by tools; never promise a refund. Always check order details for shipping issues, and always check order details and refund eligibility before replying about a refund. Choose billing, shipping, bug, or other. If a ticket reports an app crash, software error, website error, or login failure, classify it as bug, even if it mentions payment; this includes cases where the customer says no payment was made. For invoice requests, provide an invoice-specific reply. For refund requests, explain eligibility without claiming a refund was completed. Acknowledge technical issues without assuming whether the app or website is affected. If an order lookup reports payment review is required for a failed payment with a bank deduction, call escalate_to_human(reason) and only say an escalation request was recorded if that tool returns recorded=true. If uncertain, do not invent facts; draft a handoff for human review."""


class LocalTriageModel:
    """Offline stand-in for an LLM; returns the same tool-call/final-answer shape."""

    def respond(self, ticket, tool_results):
        names_called = [result["name"] for result in tool_results]
        order_id = ticket.get("order_id")
        text = ticket["subject"] + " " + ticket["body"]
        asks_refund = any(word in text.lower() for word in ("refund", "money back", "reimburse"))

        # This sample deliberately repeats a tool request to exercise the safety cap.
        if "CONFUSE_LOOP_TEST" in text:
            return {"tool_calls": [{"name": "classify_ticket_category", "arguments": {"text": text}}]}

        if "classify_ticket_category" not in names_called:
            return {"tool_calls": [{"name": "classify_ticket_category", "arguments": {"text": text}}]}

        category_result = next(
            result["result"] for result in reversed(tool_results)
            if result["name"] == "classify_ticket_category"
        )

        if order_id and category_result == "shipping" and "lookup_order" not in names_called:
            return {"tool_calls": [{"name": "lookup_order", "arguments": {"order_id": order_id}}]}

        if order_id and category_result == "billing":
            if "lookup_order" not in names_called:
                return {"tool_calls": [{"name": "lookup_order", "arguments": {"order_id": order_id}}]}
            if asks_refund and "check_refund_eligibility" not in names_called:
                return {"tool_calls": [{"name": "check_refund_eligibility", "arguments": {"order_id": order_id}}]}

            order = next(
                (result["result"] for result in reversed(tool_results)
                 if result["name"] == "lookup_order"),
                {},
            )
            if order.get("payment_review") and "escalate_to_human" not in names_called:
                reason = "Failed payment with bank deduction; order {} requires payment review.".format(order_id)
                return {"tool_calls": [{"name": "escalate_to_human", "arguments": {"reason": reason}}]}

        return {"content": draft_reply(ticket, category_result, tool_results)}


def draft_reply(ticket, category, tool_results):
    order = next((item["result"] for item in tool_results if item["name"] == "lookup_order"), None)
    refund = next((item["result"] for item in tool_results if item["name"] == "check_refund_eligibility"), None)
    escalation = next(
        (item["result"] for item in tool_results if item["name"] == "escalate_to_human"),
        None,
    )

    if refund:
        if refund.get("eligible"):
            return "Thanks for reaching out. Your order appears eligible for a refund. Our support team will review and process your refund request; this does not mean a refund has been issued."
        return "Thanks for reaching out. I checked your order, and it is not currently eligible for a refund: {} Please contact us if you would like to discuss the available options.".format(refund.get("reason", "eligibility could not be confirmed."))
    if category == "shipping" and order and not order.get("error"):
        return "Thanks for checking in. Your order is {}{}".format(
            order["status"],
            " and is estimated to arrive by {}.".format(order["estimated_delivery"]) if order.get("estimated_delivery") else ".",
        )
    if category == "bug":
        return "Sorry you're running into this technical issue. We understand something isn't working as expected. Please share the steps to reproduce it and, if relevant, your device, browser, or app/website version so we can investigate."
    if category == "billing":
        text = (ticket["subject"] + " " + ticket["body"]).lower()
        if "invoice" in text:
            return "Thanks for your invoice request. Our support team will help locate and send the invoice for your recent payment."
        if "failed" in text and ("deducted" in text or "bank" in text) and order and not order.get("error"):
            review_note = order.get("payment_review")
            if review_note and escalation and escalation.get("recorded"):
                return "Thanks for flagging this. I’ve noted your report of a failed payment with a bank deduction. The order record shows the order is {} and requires review. A request for a support specialist to review this issue was recorded, and they should follow up with you.".format(order["status"])
            if review_note:
                return "Thanks for flagging this. I’ve noted your report of a failed payment with a bank deduction. The order record shows the order is {} and notes that review is required. A support specialist should review this and follow up with you.".format(order["status"])
            return "Thanks for flagging this. I’ve noted your report of a failed payment with a bank deduction. Our order record shows the order is {}; a support specialist should review this and follow up with you.".format(order["status"])
        return "Thanks for flagging this billing issue. We’ll review the charge and follow up with you."
    if category == "shipping":
        return "Thanks for contacting us about your delivery. Please share your order ID so we can check its status."
    return "Thanks for contacting us. We’ve received your message and will follow up with you shortly."


def run_agent(ticket, model, max_tool_rounds=MAX_TOOL_ROUNDS):
    tool_results = []
    seen_calls = set()
    duplicate_calls_skipped = 0
    tool_errors = 0
    for _ in range(max_tool_rounds):
        response = model.respond(ticket, tool_results)
        if "tool_calls" not in response:
            category = next(
                (item["result"] for item in reversed(tool_results)
                 if item["name"] == "classify_ticket_category"),
                "other",
            )
            return {
                "category": category,
                "tools_called": [item["name"] for item in tool_results],
                "reply": response["content"],
                "cap_hit": False,
                "duplicate_calls_skipped": duplicate_calls_skipped,
                "tool_errors": tool_errors,
                "escalation_recorded": any(
                    item["name"] == "escalate_to_human"
                    and isinstance(item["result"], dict)
                    and item["result"].get("recorded") is True
                    for item in tool_results
                ),
            }

        for call in response["tool_calls"]:
            name = call["name"]
            arguments = call["arguments"]
            if name == "escalate_to_human" and any(
                item["name"] == "escalate_to_human" for item in tool_results
            ):
                duplicate_calls_skipped += 1
                log("  SKIP duplicate escalation request {}({})".format(
                    name, json.dumps(arguments)
                ))
                continue
            call_key = (name, json.dumps(arguments, sort_keys=True))
            if call_key in seen_calls:
                duplicate_calls_skipped += 1
                log("  SKIP duplicate tool call {}({})".format(name, json.dumps(arguments)))
                continue
            seen_calls.add(call_key)
            if name not in TOOLS:
                result = {"error": "Unknown tool."}
            else:
                try:
                    result = TOOLS[name](**arguments)
                except TypeError as error:
                    result = {"error": str(error)}
            if isinstance(result, dict) and "error" in result:
                tool_errors += 1
            log("  TOOL {}({}) -> {}".format(name, json.dumps(arguments), json.dumps(result)))
            tool_results.append({"name": name, "arguments": arguments, "result": result})

    log("  WARNING: tool-call round cap ({}) reached for ticket {}.".format(max_tool_rounds, ticket["id"]))
    category = next(
        (item["result"] for item in reversed(tool_results)
         if item["name"] == "classify_ticket_category"),
        "other",
    )
    return {
        "category": category,
        "tools_called": [item["name"] for item in tool_results],
        "reply": "Thanks for contacting us. A support specialist will review your message and follow up.",
        "cap_hit": True,
        "duplicate_calls_skipped": duplicate_calls_skipped,
        "tool_errors": tool_errors,
        "escalation_recorded": any(
            item["name"] == "escalate_to_human"
            and isinstance(item["result"], dict)
            and item["result"].get("recorded") is True
            for item in tool_results
        ),
    }


CATEGORY_COLOURS = {"billing": MAGENTA, "shipping": BLUE, "bug": RED, "other": GREEN}
TOOL_SHORT = {
    "classify_ticket_category": "classify",
    "lookup_order": "lookup",
    "check_refund_eligibility": "refund",
    "escalate_to_human": "escalate",
}


def pad(text, width, colour=""):
    """Pad on the plain text so ANSI codes do not break column widths."""
    shown = text if len(text) <= width else text[: width - 1] + "…"
    return colour + shown + RESET + " " * (width - len(shown))


def print_table(rows):
    widths = [3, 9, 10, 36, 52]
    heads = ["#", "Ticket ID", "Category", "Tools Used", "Draft Reply"]
    line = lambda l, m, r: l + m.join("─" * (w + 2) for w in widths) + r
    print("\n" + CYAN + line("┌", "┬", "┐") + RESET)
    print(CYAN + "│" + RESET + "│".join(" " + pad(h, w, BOLD + CYAN) + " " for h, w in zip(heads, widths)) + CYAN + "│" + RESET)
    print(CYAN + line("├", "┼", "┤") + RESET)
    for n, (ticket, result) in enumerate(rows, 1):
        tools = " → ".join(TOOL_SHORT.get(t, t) for t in result["tools_called"]) or "none"
        reply = " ".join(result["reply"].split())
        cells = [
            pad(str(n), widths[0], DIM),
            pad(ticket["id"], widths[1], BOLD),
            pad(result["category"], widths[2], BOLD + CATEGORY_COLOURS.get(result["category"], "")),
            pad(tools, widths[3], YELLOW if result["cap_hit"] else ""),
            pad(reply, widths[4], DIM),
        ]
        print(CYAN + "│" + RESET + (CYAN + "│" + RESET).join(" " + c + " " for c in cells) + CYAN + "│" + RESET)
    print(CYAN + line("└", "┴", "┘") + RESET)


def main():
    ticket_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickets.json")
    with open(ticket_path, "r") as ticket_file:
        tickets = json.load(ticket_file)

    model = LocalTriageModel()
    results = []
    for ticket in tickets:
        if VERBOSE:
            print("\n{}{}Ticket {}:{} {}".format(BOLD, CYAN, ticket["id"], RESET, ticket["subject"]))
        results.append((ticket, run_agent(ticket, model)))

    print_table(results)

    cap_hits = [t["id"] for t, r in results if r["cap_hit"]]
    escalated = [t["id"] for t, r in results if r["escalation_recorded"]]
    multi = [t["id"] for t, r in results if len({x for x in r["tools_called"] if x != "classify_ticket_category"}) >= 2]
    errors = sum(r["tool_errors"] for _, r in results)
    skipped = sum(r["duplicate_calls_skipped"] for _, r in results)
    counts = {}
    for _, r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    print("\n{}{}Run Summary{}".format(BOLD, CYAN, RESET))
    print("  Tickets processed : {}{}{}".format(BOLD, len(results), RESET))
    print("  Categories        : " + "  ".join("{}{}{}: {}".format(CATEGORY_COLOURS.get(c, ""), c, RESET, n) for c, n in sorted(counts.items())))
    print("  Multi-tool tickets: {}".format(", ".join(multi) or "none"))
    print("  Safety cap hit    : {}{}{} ({} duplicate call(s) skipped)".format(YELLOW, ", ".join(cap_hits) or "none", RESET, skipped))
    print("  Escalations       : {}".format(", ".join(escalated) or "none"))
    print("  Tool errors       : {}{}{}".format(GREEN if errors == 0 else RED, errors, RESET))
    if len(results) != 15:
        print("{}WARNING: expected 15 tickets, found {}.{}".format(YELLOW, len(results), RESET))


if __name__ == "__main__":
    main()
