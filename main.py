"""Run TicketPilot.

Usage:
    python main.py                       # interactive chat
    python main.py "Status of order A123?"   # single question
"""

import sys

from agent import ask_agent

LINE = "-" * 56


def single(question: str) -> None:
    print(f"\nCustomer: {question}")
    print(f"TicketPilot: {ask_agent(question)}\n")


def chat() -> None:
    print(f"{LINE}\n  TicketPilot  |  type 'exit' to quit\n{LINE}")
    while True:
        question = input("\nCustomer: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            print(f"TicketPilot: {ask_agent(question)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        single(" ".join(sys.argv[1:]))
    else:
        chat()
