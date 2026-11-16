"""Runs the 4 required test cases through the real agent (needs API key)."""

from agent import ask_agent

CASES = [
    ("Known order (Shipped)", "What's the status of order A123?"),
    ("Known order (Delivered)", "Where is my order A124?"),
    ("Known order (Processing)", "Check order A125 please"),
    ("Unknown order", "What's the status of order Z999?"),
]

for title, question in CASES:
    print(f"\n=== {title} ===\nCustomer: {question}")
    print(f"TicketPilot: {ask_agent(question)}")
