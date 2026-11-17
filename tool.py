"""Tools for the TicketPilot triage agent: mock data + four callable tools."""
import datetime

def date_offset(days):
    return (datetime.date.today() + datetime.timedelta(days=days)).isoformat()


ORDERS = {
    "ORD-1001": {"status": "shipped", "estimated_delivery": date_offset(3)},
    "ORD-1002": {"status": "delivered", "estimated_delivery": date_offset(-8)},
    "ORD-1003": {"status": "processing", "estimated_delivery": date_offset(5)},
    "ORD-1004": {"status": "cancelled", "estimated_delivery": None},
    "ORD-1005": {"status": "shipped", "estimated_delivery": date_offset(2)},
    "ORD-1006": {"status": "delivered", "estimated_delivery": date_offset(-6)},
    "ORD-1007": {
        "status": "processing",
        "estimated_delivery": date_offset(6),
        "payment_review": "Customer reports a failed payment with a bank deduction; review required.",
    },
}

REFUND_ELIGIBILITY = {
    "ORD-1001": {"eligible": False, "reason": "The order has shipped."},
    "ORD-1002": {"eligible": True, "reason": "Delivered within the 30-day return window."},
    "ORD-1003": {"eligible": True, "reason": "The order has not shipped yet."},
    "ORD-1004": {"eligible": True, "reason": "The order was cancelled."},
    "ORD-1005": {"eligible": False, "reason": "The order has shipped."},
    "ORD-1006": {"eligible": True, "reason": "Delivered within the 30-day return window."},
    "ORD-1007": {"eligible": True, "reason": "The order has not shipped yet."},
}

def classify_ticket_category(text: str) -> str:
    """Classify a ticket using small, transparent keyword rules."""
    text = text.lower()
    if any(word in text for word in ("bug", "error", "crash", "crashes", "broken", "doesn't work", "not working", "sign in", "sign-in", "login", "log in")):
        return "bug"
    if any(word in text for word in ("refund", "charged", "charge", "payment", "invoice", "billing")):
        return "billing"
    if any(word in text for word in ("shipping", "delivery", "delivered", "tracking", "track", "shipped", "package", "order is late")):
        return "shipping"
    return "other"


def lookup_order(order_id: str):
    return ORDERS.get(order_id, {"error": "Order not found."})


def check_refund_eligibility(order_id: str):
    return REFUND_ELIGIBILITY.get(order_id, {"eligible": False, "reason": "Order not found; eligibility cannot be confirmed."})


ESCALATION_REQUESTS = []


def escalate_to_human(reason: str):
    """Record a request for a support specialist to review the issue."""
    ESCALATION_REQUESTS.append(reason)
    return {"recorded": True, "reason": reason}


TOOLS = {
    "classify_ticket_category": classify_ticket_category,
    "lookup_order": lookup_order,
    "check_refund_eligibility": check_refund_eligibility,
    "escalate_to_human": escalate_to_human,
}
