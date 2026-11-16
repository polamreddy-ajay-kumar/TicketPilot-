"""TicketPilot - a tiny tool-calling support agent powered by Google Gemini."""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import TOOLS, lookup_order_status

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are TicketPilot, a polite customer-support agent. "
    "Never guess an order status. Always call the lookup_order_status tool "
    "to get the real status. If the tool returns 'Not Found', tell the "
    "customer the order ID could not be found and ask them to re-check it."
)


def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return genai.Client(api_key=api_key)


def ask_agent(question: str, verbose: bool = True) -> str:
    """Send a customer question to Gemini and let it call our Python tool."""
    client = _client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[lookup_order_status],
        # We run the tool loop ourselves so every tool call is visible.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    response = client.models.generate_content(model=MODEL, contents=contents, config=config)

    for _ in range(5):  # safety limit on tool-call rounds
        calls = response.function_calls
        if not calls:
            break
        contents.append(response.candidates[0].content)
        result_parts = []
        for call in calls:
            args = dict(call.args or {})
            if verbose:
                print(f"  [tool call]   {call.name}({args})")
            result = TOOLS[call.name](**args)
            if verbose:
                print(f"  [tool result] {result}")
            result_parts.append(
                types.Part.from_function_response(name=call.name, response=result)
            )
        contents.append(types.Content(role="user", parts=result_parts))
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )

    return response.text or "Sorry, I could not generate a reply."
