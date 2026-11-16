"""Tools that the AI agent is allowed to call."""

ORDERS = {
    "A123": "Shipped",
    "A124": "Delivered",
    "A125": "Processing",
}


def lookup_order_status(order_id: str) -> dict:
    """Look up the current status of a customer order.

    Args:
        order_id: The order ID given by the customer, for example "A123".

    Returns:
        A dictionary with the order_id and its status.
        The status is "Not Found" if the order ID does not exist.
    """
    clean_id = str(order_id).strip().upper()
    status = ORDERS.get(clean_id, "Not Found")
    return {"order_id": clean_id, "status": status}


TOOLS = {"lookup_order_status": lookup_order_status}
