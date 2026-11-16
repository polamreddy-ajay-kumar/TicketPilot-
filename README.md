# ✈️ TicketPilot — Tool-Calling LLM

A small Python AI agent that **never guesses** an order status. When a customer asks
*"What's the status of order A123?"*, the Gemini model decides to call a real Python
function, `lookup_order_status()`, and then answers using the real result.

## How it works

```
Customer question
      │
      ▼
 Gemini model ──(decides to call a tool)──► lookup_order_status("A123")
      ▲                                              │
      └────────────(tool result: Shipped)────────────┘
      │
      ▼
 Final natural-language answer
```

| Order ID | Status      |
|----------|-------------|
| A123     | Shipped     |
| A124     | Delivered   |
| A125     | Processing  |
| anything else | Not Found |

## Project structure

```
TicketPilot/
├── main.py            # run the agent (chat or single question)
├── agent.py           # Gemini tool-calling loop
├── tools.py           # lookup_order_status() function
├── run_tests.py       # runs 4 test questions through the real agent
├── tests/test_tools.py# offline unit tests for the tool
├── requirements.txt
├── .env.example
└── docs/screenshots/  # output screenshots
```

## Setup

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Install and configure:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

3. Open `.env` and paste your key: `GEMINI_API_KEY=your_key`

## Run

```bash
python main.py                              # interactive chat
python main.py "What's the status of order A123?"
python run_tests.py                         # known + unknown order tests
python -m unittest discover -s tests -v     # offline tool tests (no API key)
```

## Sample output

```
Customer: What's the status of order A123?
  [tool call]   lookup_order_status({'order_id': 'A123'})
  [tool result] {'order_id': 'A123', 'status': 'Shipped'}
TicketPilot: Your order A123 has been shipped.

Customer: What's the status of order Z999?
  [tool call]   lookup_order_status({'order_id': 'Z999'})
  [tool result] {'order_id': 'Z999', 'status': 'Not Found'}
TicketPilot: I couldn't find order Z999. Could you please double-check the ID?
```

## Screenshots

![Known 
![Screenshots](

## Tech stack

Python · Google Gemini API (`google-genai`) · python-dotenv

## What I learned

- LLMs should call tools for facts instead of guessing.
- The model returns a *function call*; my code runs it and sends back the result.
- Always test both the happy path and the "Not Found" path.

