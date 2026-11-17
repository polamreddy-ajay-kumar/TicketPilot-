# TicketPilot — Week 3: RAG for Policy-Grounded Replies

TicketPilot's triage agent now has a company **policy knowledge base**. Refund, shipping,
cancellation and escalation answers are grounded in retrieved policy text and cite the chunk they
came from. If no document supports the answer, the agent says it **cannot confirm** instead of
inventing a rule.

Retrieval is exposed as a normal tool, `search_policy_docs`, that runs inside the same
tool-calling loop as the Week 2 triage tools.

## Output screenshots

**1. Fresh run: embeddings are created (cache MISS); refund and shipping replies cite real chunks**

![Fresh run, cache MISS, refund and shipping tickets](screenshots/01_fresh_run_cache_miss.png)

**2. Second run: embeddings are reused (cache HIT); an uncovered policy gets "cannot confirm"**

![Cache HIT and cannot-confirm reply for price matching](screenshots/02_cache_hit_and_cannot_confirm.png)

**3. Summary: retrieval was called before the reply on 7 of 8 tickets; 27 tests pass**

![Summary table and unittest result](screenshots/03_summary_and_tests.png)

These are rendered captures of the real output of `python main.py` and
`python -m unittest discover -s tests` from this repo.

## How it works

```
ticket ──> agent loop ──> model turn ──> tool_call(s) ──> tool result ──> ... ──> final reply
                              │                │
                              │                ├─ classify_ticket      (Week 2 style tool)
                              │                ├─ lookup_order         (Week 2 style tool)
                              │                └─ search_policy_docs   (NEW: RAG as a tool)
                              └─ SYSTEM_PROMPT: must search before any policy reply
```

1. **Knowledge base**: `policy_docs/` holds 4 short `.txt` policies (refund, shipping,
   cancellation, escalation) with concrete rules.
2. **Chunking and embedding**: `chunk_text()` splits each document into overlapping chunks
   (300 words, 40 overlap). Each chunk is embedded once. Vectors are cached in
   `embedding_cache.json`, keyed by the SHA-256 hash of the chunk text.
3. **Retrieval tool**: `search_policy_docs(query, top_k=3)` ranks chunks by cosine similarity
   combined with keyword overlap. Chunks below `MIN_RELEVANCE` (0.18) are dropped, so an
   uncovered topic returns nothing.
4. **Tool-calling loop**: `run_agent()` keeps a chat-style message list
   (user, assistant `tool_calls`, tool results). It dispatches each call through `TOOL_REGISTRY`
   and repeats until the model returns a final reply (max 5 steps).
5. **System prompt**: `SYSTEM_PROMPT` tells the agent to call `search_policy_docs` before drafting
   any reply that mentions a refund, deadline or policy rule, to use only retrieved text, to cite the
   chunk id, and to say it cannot confirm when nothing supports the answer. Retrieval is not forced
   for non-policy tickets (T-108 skips it).
6. **Traceable output**: every reply is printed with `Policy rule` and `Policy chunks used`, so each
   policy claim maps to a chunk id such as `refund_policy.txt#chunk-1`.

### About the "model"

No API key or network is needed. The loop is driven by `LocalPolicyModel`, a deterministic stand-in
that speaks the same protocol as a hosted LLM (`{"content", "tool_calls"}`), follows the system-prompt
rule, and drafts only from retrieved chunks. To use a real LLM, write a callable that sends
`messages`, `TOOL_SCHEMAS` and `SYSTEM_PROMPT` to your provider and returns that same dictionary;
`run_agent(..., model=your_model)` needs no other change.
`classify_ticket` and `lookup_order` are simple placeholders for the Week 2 tools; swap in your own.

## Run it

```bash
python main.py                  # all 8 sample tickets + summary table
python main.py --topk-test      # stretch goal: top_k=1 vs top_k=3 on 3 tickets
python -m unittest discover -s tests
```

Python 3.6+, standard library only (`requirements.txt` lists no packages).

## Sample tickets and what they prove

| Ticket | Scenario | Result |
|---|---|---|
| T-101 | Refund deadline | Cites `refund_policy.txt#chunk-1` (30 calendar days) |
| T-102 | Delivery time | Cites `shipping_policy.txt#chunk-1` (3 to 5 business days) |
| T-103 | **Price matching (not in any doc)** | **Cannot confirm**, recommends contacting support, `Source: None` |
| T-104 | Cancel while processing | Cites `cancellation_policy.txt#chunk-1` |
| T-105 | Package late after 7 business days | Cites `shipping_policy.txt#chunk-1` |
| T-106 | Defective item, shipping-charge refund (`ORD-1042`) | `lookup_order` + `search_policy_docs`, cites `refund_policy.txt#chunk-1` |
| T-107 | Safety issue | Cites `escalation_policy.txt#chunk-1` |
| T-108 | "Thanks, sounds great!" | No policy question, so no retrieval |

## Checklist (Week 3 "Done When")

- [x] Refund ticket produces a reply citing a real rule from `refund_policy.txt` (T-101, T-106).
- [x] Uncovered policy gets an honest "cannot confirm" instead of an invented rule (T-103).
- [x] Re-running does not re-embed the documents: 1st run `0 HIT, 4 MISS`, 2nd run `4 HIT, 0 MISS`
      (also asserted by an automated test).
- [x] At least 3 tickets show retrieval before the reply (7 do; see the Summary screenshot).
- [x] Stretch: `--topk-test` compares `top_k=1` and `top_k=3` on T-102, T-105 and T-106. With this
      small knowledge base, `top_k=1` already returns the right chunk and the replies are identical.
      `top_k=3` mainly adds loosely related chunks, such as a refund chunk for a shipping question.

## Project structure

```text
.
├── main.py                    # chunking, embeddings + cache, tools, system prompt, agent loop
├── README.md
├── requirements.txt
├── .gitignore                 # ignores the generated embedding_cache.json
├── screenshots/               # the 3 output screenshots used above
├── policy_docs/               # refund, shipping, cancellation, escalation policies
└── tests/test_policy_agent.py # 27 unittest cases
```

## Tests

27 tests, all passing (Python 3.13). They cover:

- each policy rule;
- query and `top_k` validation;
- cache MISS then HIT, plus a two-process re-run check;
- missing or corrupted cache;
- a missing policy folder;
- the unsupported price-matching query;
- citations;
- tool dispatch;
- retrieval before reply;
- no forced retrieval for non-policy tickets;
- the Week 2 tools running alongside retrieval.

## Limitations

- Embeddings are hashed bag-of-words vectors, which are lexical rather than semantic. Paraphrases
  with no shared terms may not be retrieved. Swap `embed_text` for a real embedding model to
  improve this; the cache and search code stay the same.
- The `0.18` relevance threshold and the alias list are heuristics tuned to these four documents.
- The model is a deterministic stand-in. See "About the model" above to plug in a real LLM.
