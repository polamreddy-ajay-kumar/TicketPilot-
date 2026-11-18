import hashlib
import json
import math
import os
import re
import sys
import tempfile


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(BASE_DIR, "policy_docs")
CACHE_FILE = os.path.join(BASE_DIR, "embedding_cache.json")
VECTOR_SIZE = 512
MIN_RELEVANCE = 0.18
MAX_TOP_K = 20
MAX_LEXICAL_TERMS = 6
MIN_FALLBACK_OVERLAP = 2
MATCH_GENERIC_WORDS = set(("item order day business policy customer product company" ).split())

STOP_WORDS = set((
    "a an and are as at be by can do for from has have how i if in is it me my "
    "of on or our please the this to we what when where who with would you your "
    "does did was were am been being about after before within into should"
).split())
WORD_ALIASES = {
    "refunds": "refund", "refunded": "refund", "refunding": "refund",
    "returns": "return", "returned": "return", "cancellation": "cancel",
    "cancellations": "cancel", "cancelled": "cancel", "canceled": "cancel",
    "shipping": "ship", "shipped": "ship", "shipment": "ship",
    "shipments": "ship", "dispatched": "dispatch", "dispatching": "dispatch",
    "days": "day", "businesses": "business", "items": "item",
    "processing": "process", "processed": "process", "inspection": "inspect",
    "inspected": "inspect",
    "products": "product", "charges": "charge", "updates": "update",
    "updated": "update", "arrived": "arrive", "arrives": "arrive",
    "delayed": "delay", "delaying": "delay",
    "fulfillment": "fulfill", "fulfilled": "fulfill", "eligible": "eligibility",
    "matching": "match", "cheaper": "cheap", "alleges": "allege", "alleging": "allege",
    "alleged": "allege",
    # synonyms of words the escalation policy uses ("safety issue", "legal action")
    "fire": "safety", "overheated": "safety", "overheating": "safety", "sparked": "safety",
    "dangerous": "safety", "sue": "legal", "sued": "legal", "lawsuit": "legal",
    "lawyer": "legal", "attorney": "legal",
    "damaged": "defective", "broken": "defective", "faulty": "defective",
    "threatens": "threaten", "requests": "request", "requested": "request",
}


def chunk_text(text, max_words=300, overlap=40):
    """Split text into overlapping chunks, validating chunk settings."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(max_words, int) or isinstance(max_words, bool) or max_words <= 0:
        raise ValueError("max_words must be a positive integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def tokens(text):
    if not isinstance(text, str):
        return []
    result = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        word = WORD_ALIASES.get(word, word)
        if word not in STOP_WORDS:
            result.append(word)
    return result


def embed_text(text):
    """Create a deterministic, sparse hashed bag-of-words vector."""
    vector = [0.0] * VECTOR_SIZE
    for word in tokens(text):
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % VECTOR_SIZE
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return vector


def cosine_similarity(left, right):
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def load_policy_chunks(policy_dir=POLICY_DIR):
    """Load non-empty .txt policy files; missing/unreadable files are skipped safely."""
    chunks = []
    seen = set()
    try:
        filenames = sorted(os.listdir(policy_dir))
    except OSError as error:
        print("Policy documents unavailable at {}: {}".format(policy_dir, error))
        return chunks

    for filename in filenames:
        if not filename.lower().endswith(".txt"):
            continue
        path = os.path.join(policy_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as policy_file:
                content = policy_file.read()
        except (IOError, UnicodeError) as error:
            print("Skipping policy file {}: {}".format(path, error))
            continue
        if not content.strip():
            print("Skipping empty policy file: {}".format(path))
            continue
        for text in chunk_text(content):
            fingerprint = " ".join(text.lower().split())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            chunk_id = "{}#chunk-{}".format(filename, len(
                [item for item in chunks if item["source"] == filename]) + 1)
            chunks.append({"source": filename, "chunk_id": chunk_id, "text": text})
    return chunks


def _valid_vector(vector):
    return (isinstance(vector, list) and len(vector) == VECTOR_SIZE and
            all(isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(value) for value in vector))


def load_embedding_cache(cache_file=CACHE_FILE):
    try:
        with open(cache_file, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        if not isinstance(data, dict):
            raise ValueError("cache root must be a JSON object")
        valid = {key: value for key, value in data.items() if _valid_vector(value)}
        if len(valid) != len(data):
            print("Embedding cache contains invalid vectors; invalid entries ignored.")
        return valid
    except (IOError, ValueError, TypeError) as error:
        if os.path.exists(cache_file):
            print("Embedding cache could not be read; starting with an empty cache: {}".format(error))
        else:
            print("Embedding cache not found; it will be created as needed.")
        return {}


def save_embedding_cache(cache, cache_file=CACHE_FILE):
    """Atomically replace the cache file so interruption cannot leave partial JSON."""
    directory = os.path.dirname(os.path.abspath(cache_file))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=".embedding_cache_", suffix=".tmp", delete=False) as file_handle:
            temporary_path = file_handle.name
            json.dump(cache, file_handle, indent=2, sort_keys=True)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, cache_file)
        return True
    except (IOError, OSError, TypeError, ValueError) as error:
        print("Warning: could not save embedding cache {}: {}".format(cache_file, error))
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        return False


POLICY_CHUNKS = load_policy_chunks()
EMBEDDING_CACHE = load_embedding_cache()
CACHE_STATS = {"HIT": 0, "MISS": 0}
print("Embedding cache file: {} ({} saved vectors loaded)".format(
    os.path.basename(CACHE_FILE), len(EMBEDDING_CACHE)))


def cached_embedding(text, label):
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    vector = EMBEDDING_CACHE.get(key)
    if _valid_vector(vector):
        CACHE_STATS["HIT"] += 1
        print("  Embedding cache HIT: {} (reused saved vector)".format(label))
        return vector
    vector = embed_text(text)
    EMBEDDING_CACHE[key] = vector
    CACHE_STATS["MISS"] += 1
    print("  Embedding cache MISS: {} (generated and saved vector)".format(label))
    save_embedding_cache(EMBEDDING_CACHE)
    return vector


for _policy_chunk in POLICY_CHUNKS:
    _policy_chunk["embedding"] = cached_embedding(
        _policy_chunk["text"], _policy_chunk["chunk_id"])
print("Embedding cache summary: {} HIT, {} MISS".format(
    CACHE_STATS["HIT"], CACHE_STATS["MISS"]))


def _validate_query(query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not tokens(query):
        raise ValueError("query must contain searchable words")


def _validate_top_k(top_k):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError("top_k must be an integer from 1 to {}".format(MAX_TOP_K))


def search_policy_docs(query, top_k=3):
    """Return threshold-qualified chunks with both lexical and vector relevance."""
    _validate_query(query)
    _validate_top_k(top_k)
    query_terms = set(tokens(query))
    match_terms = query_terms.difference(MATCH_GENERIC_WORDS)
    query_vector = embed_text(query)
    ranked = []
    for policy_chunk in POLICY_CHUNKS:
        document_terms = set(tokens(policy_chunk["text"]))
        overlap = match_terms.intersection(document_terms)
        # A vector hash collision alone is not evidence that a policy is relevant.
        if not overlap:
            continue
        cosine = max(0.0, cosine_similarity(query_vector, policy_chunk["embedding"]))
        # Cap the denominator so a long, chatty ticket doesn't dilute a clear keyword match.
        lexical = len(overlap) / float(min(len(match_terms), MAX_LEXICAL_TERMS) or 1)
        score = 0.65 * cosine + 0.35 * lexical
        if score >= MIN_RELEVANCE:
            ranked.append({
                "source": policy_chunk["source"],
                "chunk_id": policy_chunk["chunk_id"],
                "text": policy_chunk["text"],
                "score": round(score, 4),
            })
    ranked.sort(key=lambda result: result["score"], reverse=True)
    return ranked[:top_k]


ORDER_DB = {
    "ORD-1042": {"status": "delivered", "item": "Wireless Earbuds", "final_sale": False},
    "ORD-2087": {"status": "in transit", "item": "Desk Lamp", "final_sale": False},
}
HIGH_PRIORITY_TERMS = set(("safety", "safe", "fraud", "legal", "lawsuit", "overheated", "fire"))
# Escalation triggers copied from escalation_policy.txt: safety, fraud, legal action, exception.
ESCALATION_TERMS = set(("safety", "safe", "fraud", "legal", "lawsuit", "overheated", "fire",
                        "exception", "supervisor"))
CATEGORY_TERMS = [
    ("escalation", ESCALATION_TERMS),
    ("cancellation", set(("cancel",))),
    ("refund", set(("refund", "return", "defective"))),
    ("shipping", set(("ship", "delivery", "dispatch", "tracking", "arrive", "package", "delay"))),
]


SMALL_TALK_TERMS = set(("thank", "thanks", "great", "love", "appreciate", "awesome", "perfect"))
# Words that state what the customer WANTS (intent) count double; words that only describe
# the situation (dispatch, package, ...) count once. "cancel ... dispatched" is a cancellation.
INTENT_TERMS = set(("cancel", "refund", "return", "defective", "safety", "safe", "fraud",
                    "legal", "lawsuit", "overheated", "fire", "exception"))
NON_DELIVERY = re.compile(r"\b(never|not|n't|didn't|hasn't|haven't)\b\W+(?:\w+\W+){0,2}?arriv", re.I)


def classify_ticket(ticket):
    """Week 2 style triage tool: category, priority and a confidence score.

    Confidence is the winning category's weighted share of all keyword hits, scaled by how
    many distinct keywords matched. Ties between categories (two competing intents) and
    tickets with no keyword at all (that are not small talk) get LOW confidence.
    "never/not arrived" counts as a shipping signal so a missing package is not mistaken
    for an ordinary refund request.
    """
    if not isinstance(ticket, str) or not ticket.strip():
        raise ValueError("ticket must be a non-empty string")
    words = set(tokens(ticket))
    hits = [(name, words & terms) for name, terms in CATEGORY_TERMS]
    weights = [sum(2 if word in INTENT_TERMS else 1 for word in matched) for _, matched in hits]
    distinct = [len(matched) for _, matched in hits]
    matched_extra = {}
    if NON_DELIVERY.search(ticket):
        shipping = [name for name, _ in hits].index("shipping")
        weights[shipping] += 2
        distinct[shipping] += 1
        matched_extra[shipping] = ["non_delivery"]
    top = max(weights)
    priority = "high" if words & HIGH_PRIORITY_TERMS else "normal"
    if top == 0:
        confidence = 0.8 if words & SMALL_TALK_TERMS else 0.4
        return {"category": "general", "priority": priority,
                "confidence": confidence, "matched_terms": []}
    best = weights.index(top)
    share = top / float(sum(weights))
    confidence = round(share * min(1.0, 0.6 + 0.2 * distinct[best]), 2)
    return {"category": hits[best][0], "priority": priority, "confidence": confidence,
            "matched_terms": sorted(hits[best][1]) + matched_extra.get(best, [])}


def lookup_order(order_id):
    """Week 2 style tool: mock order lookup (replace with your real Week 2 tool)."""
    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError("order_id must be a non-empty string")
    order = ORDER_DB.get(order_id.strip().upper())
    return dict(order, order_id=order_id.strip().upper()) if order else {
        "order_id": order_id.strip().upper(), "status": "not found"}


TOOL_REGISTRY = {
    "classify_ticket": classify_ticket,
    "lookup_order": lookup_order,
    "search_policy_docs": search_policy_docs,
}

TOOL_SCHEMAS = [
    {"name": "classify_ticket",
     "description": "Classify a support ticket into a category and priority.",
     "parameters": {"type": "object", "properties": {"ticket": {"type": "string"}},
                    "required": ["ticket"]}},
    {"name": "lookup_order",
     "description": "Look up an order's status by order id (e.g. ORD-1042).",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]}},
    {"name": "search_policy_docs",
     "description": "Search the company policy knowledge base (refund, shipping, "
                    "cancellation, escalation). Returns the top_k most relevant chunks "
                    "with chunk ids that must be cited. Put the policy's stage words "
                    "(processing, fulfillment, dispatch) in the query, not only the "
                    "customer's wording.",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"},
                                   "top_k": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "required": ["query"]}},
]

SYSTEM_PROMPT = """You are TicketPilot, a customer-support triage agent.
You can call these tools: classify_ticket, lookup_order, search_policy_docs.

Rules:
1. Before drafting ANY reply that mentions a refund, deadline, shipping time,
   cancellation rule or any other company policy, you MUST call
   search_policy_docs first. Tickets with no policy question need no search.
2. State only rules that appear in the retrieved chunks. Never rely on your own
   knowledge of what a "typical" refund or shipping policy looks like.
3. Cite the chunk id (for example refund_policy.txt#chunk-1) for every policy claim.
4. If no retrieved chunk supports the answer, say you cannot confirm the policy
   and recommend contacting support. Never invent a rule.
5. Customers describe the order stage in their own words ("still in the warehouse",
   "hasn't left", "packed", "shipped"). Map that to the policy's stages (processing,
   fulfillment, dispatch) and search with the policy's words as well as theirs.
6. Never turn "not dispatched yet" into "you can cancel". Cancellation depends on whether
   the order has entered fulfillment. If the retrieved chunk says it depends on the order
   status, say exactly that and send the customer to support with the order number.
"""


def call_tool(name, arguments):
    if not isinstance(name, str) or name not in TOOL_REGISTRY:
        raise ValueError("Unknown tool: {}".format(name))
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a dictionary")
    try:
        return TOOL_REGISTRY[name](**arguments)
    except TypeError as error:
        raise ValueError("Invalid arguments for tool {}: {}".format(name, error))


def execute_tool_calls(tool_calls):
    """Dispatch validated model-style calls and preserve call order."""
    if not isinstance(tool_calls, (list, tuple)):
        raise ValueError("tool_calls must be a list or tuple")
    results = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict) or "name" not in tool_call or "arguments" not in tool_call:
            raise ValueError("Tool call at index {} must contain name and arguments".format(index))
        results.append(call_tool(tool_call["name"], tool_call["arguments"]))
    return results


POLICY_TERMS = set((
    "refund return ship delivery dispatch deadline late delay cancel order eligible "
    "day business price match final sale defective wrong charge safety safe fraud "
    "legal lawsuit exception supervisor tracking track processing fulfill policy "
    "support arrive package"
).split())


def needs_policy_lookup(ticket):
    if not isinstance(ticket, str) or not ticket.strip():
        return False
    category_words = set().union(*[terms for _, terms in CATEGORY_TERMS])
    return bool(set(tokens(ticket)).intersection(POLICY_TERMS | category_words))


def _sentences(policy_text):
    result = re.split(r"(?<=[.!?])\s+", policy_text)
    result = [re.sub(
        r"^(?:Refund and Returns|Shipping and Delivery|Order Cancellation|Escalation) Policy\s*",
        "", sentence).strip() for sentence in result]
    return [sentence for sentence in result if sentence]


def select_policy_rule(ticket, policy_text):
    """Select only source sentences that directly answer the ticket."""
    query_words = set(tokens(ticket))
    sentences = _sentences(policy_text)
    if not sentences:
        return ""

    rules = [
        (("refund", "deadline"), "30 calendar days"),
        (("refund", "process"), "5 to 10 business days"),
        (("final", "sale"), "Final-sale items"),
        (("ship", "charge"), "Original shipping charges"),
        (("defective",), "Original shipping charges"),
        (("wrong", "item"), "Original shipping charges"),
        (("ship", "process"), "1 to 2 business days"),
        (("ship", "delivery"), "3 to 5 business days"),
        (("ship", "delay"), "7 business days after dispatch"),
        (("tracking", "update"), "7 business days after dispatch"),
        (("cancel", "dispatch"), "Once an order has entered fulfillment"),
        (("cancel", "fulfill"), "Once an order has entered fulfillment"),
        (("cancel", "process"), "A customer may request cancellation"),
        (("safety",), "Escalate a ticket"),
        (("safe",), "Escalate a ticket"),
        (("fraud",), "Escalate a ticket"),
        (("legal",), "Escalate a ticket"),
        (("exception",), "Escalate a ticket"),
        (("cancel", "warehouse"), "can be cancelled only if it has not entered fulfillment"),
        (("cancel", "packed"), "can be cancelled only if it has not entered fulfillment"),
        (("return", "day"), "30 calendar days"),
        (("arrive",), "7 business days after dispatch"),
    ]
    for required_terms, phrase in rules:
        if set(required_terms).issubset(query_words):
            matched = [sentence for sentence in sentences if phrase.lower() in sentence.lower()]
            if matched:
                if phrase == "3 to 5 business days":
                    support = next((sentence for sentence in sentences
                                    if "7 business days after dispatch" in sentence), None)
                    return "{} {}".format(matched[0], support) if support else matched[0]
                return matched[0]

    # Fallback: need >= MIN_FALLBACK_OVERLAP shared specific words. A lone topic word such as
    # "shipping" must not make an unrelated sentence look like an answer (ungrounded guess).
    specific = query_words.difference(MATCH_GENERIC_WORDS)
    candidates = [(len(specific.intersection(tokens(sentence))), sentence) for sentence in sentences]
    best = max(candidates, key=lambda candidate: candidate[0])
    return best[1] if best[0] >= MIN_FALLBACK_OVERLAP else ""


CATEGORY_POLICY = {"refund": "refund_policy.txt", "shipping": "shipping_policy.txt",
                   "cancellation": "cancellation_policy.txt", "escalation": "escalation_policy.txt"}


def relevant_policy_chunk(ticket, retrieved_chunks, preferred_source=None):
    """Pick the chunk that answers the ticket; prefer the policy file matching its category."""
    if not retrieved_chunks:
        return None
    if preferred_source:
        retrieved_chunks = [c for c in retrieved_chunks if c["source"] == preferred_source] or retrieved_chunks
    query_terms = set(tokens(ticket))
    preferred_rules = []
    if "refund" in query_terms and "deadline" in query_terms:
        preferred_rules.append("30 calendar days")
    if "refund" in query_terms and query_terms.intersection(set(("process", "inspect"))):
        preferred_rules.append("5 to 10 business days")
    if "final" in query_terms and "sale" in query_terms:
        preferred_rules.append("Final-sale items")
    if "ship" in query_terms and "process" in query_terms:
        preferred_rules.append("1 to 2 business days")
    if "ship" in query_terms and "delivery" in query_terms:
        preferred_rules.append("3 to 5 business days")
    if "cancel" in query_terms and query_terms.intersection(set(("dispatch", "fulfill"))):
        preferred_rules.append("Once an order has entered fulfillment")
    if "cancel" in query_terms and "process" in query_terms:
        preferred_rules.append("A customer may request cancellation")
    if query_terms.intersection(set(("safety", "safe", "fraud", "legal", "exception"))):
        preferred_rules.append("Escalate a ticket")
    for phrase in preferred_rules:
        for chunk in retrieved_chunks:
            if phrase.lower() in chunk["text"].lower():
                return chunk
    return max(retrieved_chunks, key=lambda chunk: len(
        query_terms.intersection(tokens(chunk["text"]))))


def draft_reply(ticket, retrieved_chunks, preferred_source=None):
    if not retrieved_chunks:
        return ("I couldn't confirm that policy from the company documents available to me. "
                "Please contact support for confirmation. Source: None.")
    chunk = relevant_policy_chunk(ticket, retrieved_chunks, preferred_source)
    rule = select_policy_rule(ticket, chunk["text"])
    if not rule:
        return ("I couldn't confirm that policy from the company documents available to me. "
                "Please contact support for confirmation. Source: None.")
    return "Our policy states: {} (Source: {}).".format(rule, chunk["chunk_id"])

# --------------------------------------------------------------------------
# Tool-calling agent loop (RAG is "just another tool" the model chooses to call)
# --------------------------------------------------------------------------
STAGE_WORDS = {"warehouse": "fulfillment", "packed": "fulfillment", "packing": "fulfillment",
               "picked": "fulfillment", "shipped": "dispatch", "left": "dispatch"}


def stage_terms(ticket):
    """Policy stage words implied by the customer's wording (prompt rule 5)."""
    words = set(re.findall(r"[a-z]+", ticket.lower()))
    return sorted(set(STAGE_WORDS[w] for w in words if w in STAGE_WORDS))


DISPATCHED_STATUSES = ("in transit", "shipped", "out for delivery", "delivered")


def ticket_with_context(ticket, order):
    """Fold lookup_order facts into the ticket text so retrieval and rule selection see them."""
    if isinstance(order, dict) and order.get("status") in DISPATCHED_STATUSES:
        return ticket + " (order already dispatched)"
    return ticket


ORDER_ID_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
MAX_STEPS = 5
CANNOT_CONFIRM = ("I couldn't confirm that policy from the company documents available "
                  "to me. Please contact support for confirmation. Source: None.")


class LocalPolicyModel(object):
    """Deterministic stand-in for an LLM.

    It speaks the same chat protocol a hosted model would (system prompt, tool_calls,
    tool results), obeys SYSTEM_PROMPT rule 1 (search before drafting a policy reply)
    and drafts only from retrieved chunks. Swap it for a real model client that
    returns the same {"content", "tool_calls"} dictionary.
    """
    name = "local-deterministic-planner"

    def __call__(self, messages, tools, system_prompt):
        ticket = next(m["content"] for m in messages if m["role"] == "user")
        done = {m["name"]: json.loads(m["content"]) for m in messages if m["role"] == "tool"}
        calls = []
        if "classify_ticket" not in done:
            calls.append(("classify_ticket", {"ticket": ticket}))
        order = ORDER_ID_PATTERN.search(ticket)
        if order and "lookup_order" not in done:
            calls.append(("lookup_order", {"order_id": order.group(0)}))
        if needs_policy_lookup(ticket) and "search_policy_docs" not in done:
            # Query expansion: add the classifier's category word so the right policy file ranks.
            category = done.get("classify_ticket", {}).get("category")
            context = ticket_with_context(ticket, done.get("lookup_order"))
            query = "{} {}".format(context, category) if category in CATEGORY_POLICY else context
            if category == "cancellation":
                query = " ".join([query] + stage_terms(ticket))
            calls.append(("search_policy_docs", {"query": query, "top_k": 3}))
        if calls:
            name, args = calls[0]  # one tool call per step, like a typical model turn
            return {"content": None, "tool_calls": [
                {"id": "call_{}".format(len(messages)), "name": name, "arguments": args}]}
        if "search_policy_docs" in done:
            source = CATEGORY_POLICY.get(done.get("classify_ticket", {}).get("category"))
            reply = draft_reply(ticket_with_context(ticket, done.get("lookup_order")),
                                done["search_policy_docs"], source)
        else:
            reply = "Thanks for reaching out! No policy question was detected, so a support agent will follow up."
        if "lookup_order" in done and done["lookup_order"].get("status") != "not found":
            reply = "Re: order {} (status: {}). {}".format(
                done["lookup_order"]["order_id"], done["lookup_order"]["status"], reply)
        return {"content": reply, "tool_calls": []}


def run_agent(ticket_id, ticket, model=None, verbose=True):
    """Run the tool-calling loop and return a dict with reply, chunks used and trace."""
    if not isinstance(ticket, str) or not ticket.strip():
        raise ValueError("ticket must be a non-empty string")
    model = model or LocalPolicyModel()
    messages = [{"role": "user", "content": ticket}]
    trace, retrieved, retrieval_step, reply = [], [], None, None
    if verbose:
        print("\n{}: {}".format(ticket_id, ticket))
    for step in range(1, MAX_STEPS + 1):
        response = model(messages, TOOL_SCHEMAS, SYSTEM_PROMPT)
        if not response.get("tool_calls"):
            reply = response["content"]
            break
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": response["tool_calls"]})
        for call in response["tool_calls"]:
            try:
                result = execute_tool_calls([call])[0]
            except ValueError as error:
                result = {"error": str(error)}
            if call["name"] == "search_policy_docs" and isinstance(result, list):
                retrieved, retrieval_step = result, step
            trace.append((step, call["name"], call["arguments"], result))
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "name": call["name"], "content": json.dumps(result)})
            if verbose:
                _print_tool_step(step, call, result)
    if reply is None:
        reply = CANNOT_CONFIRM
    category = next((r.get("category") for _, name, _, r in trace
                     if name == "classify_ticket" and isinstance(r, dict)), None)
    order = next((r for _, name, _, r in trace if name == "lookup_order"), None)
    context = ticket_with_context(ticket, order)
    chunk = relevant_policy_chunk(context, retrieved, CATEGORY_POLICY.get(category))
    rule = select_policy_rule(context, chunk["text"]) if chunk else ""
    chunks_used = [chunk["chunk_id"]] if chunk and rule else []
    outcome = {
        "ticket_id": ticket_id, "reply": reply, "policy_rule": rule or None,
        "chunks_used": chunks_used, "retrieval_called": retrieval_step is not None,
        "retrieved": [r["chunk_id"] for r in retrieved], "trace": trace,
    }
    if verbose:
        print("  Drafted reply: {}".format(reply))
        print("  Policy rule: {}".format(rule or "None confirmed from the available documents."))
        print("  Policy chunks used: {}".format(", ".join(chunks_used) or "None"))
    return outcome


def _print_tool_step(step, call, result):
    arguments = dict(call["arguments"])
    if "ticket" in arguments:
        arguments["ticket"] = "<ticket text>"
    if "query" in arguments:
        arguments["query"] = "<ticket text>"
    print("  Step {}: tool_call {}({})".format(step, call["name"], json.dumps(arguments)))
    if call["name"] == "search_policy_docs" and isinstance(result, list):
        if not result:
            print("          -> 0 chunks above relevance threshold {:.2f}".format(MIN_RELEVANCE))
        for item in result:
            snippet = item["text"] if len(item["text"]) <= 150 else item["text"][:147] + "..."
            print("          -> {} (score {}): {}".format(item["chunk_id"], item["score"], snippet))
    else:
        print("          -> {}".format(json.dumps(result)))


SAMPLE_TICKETS = [
    ("T-101", "I received my order last week and want a refund. What is the refund deadline?"),
    ("T-102", "My package was dispatched four business days ago. What is the shipping delivery timeframe?"),
    ("T-103", "Do you offer price matching if I find the same item cheaper elsewhere?"),
    ("T-104", "Can I cancel my order while it is still processing?"),
    ("T-105", "My package has not arrived after seven business days; should I contact support?"),
    ("T-106", "Order ORD-1042 arrived defective. Will you refund the original shipping charges?"),
    ("T-107", "The charger I received overheated and sparked. This is a safety issue, who handles it?"),
    ("T-108", "Thanks so much, the earbuds sound great!"),
]


def print_summary(outcomes):
    print("\n" + "=" * 78)
    print("SUMMARY: retrieval called before reply / policy chunk cited")
    print("=" * 78)
    print("{:<8}{:<11}{:<32}{}".format("Ticket", "Retrieval", "Chunk cited", "Answer"))
    for item in outcomes:
        answer = "grounded" if item["chunks_used"] else (
            "cannot confirm" if item["retrieval_called"] else "no policy needed")
        print("{:<8}{:<11}{:<32}{}".format(
            item["ticket_id"], "YES" if item["retrieval_called"] else "no",
            ", ".join(item["chunks_used"]) or "-", answer))
    grounded = sum(1 for item in outcomes if item["chunks_used"])
    print("\n{} of {} tickets called search_policy_docs before drafting; "
          "{} replies cite a real chunk.".format(
              sum(1 for i in outcomes if i["retrieval_called"]), len(outcomes), grounded))


def topk_sensitivity_test(tickets=None):
    """Stretch goal: compare retrieved chunks and reply at top_k=1 vs top_k=3."""
    tickets = tickets or [t for t in SAMPLE_TICKETS if t[0] in ("T-102", "T-105", "T-106")]
    print("\nTOP_K SENSITIVITY (top_k=1 vs top_k=3)")
    for ticket_id, ticket in tickets:
        print("\n{}: {}".format(ticket_id, ticket))
        for top_k in (1, 3):
            results = search_policy_docs(ticket, top_k=top_k)
            reply = draft_reply(ticket, results)
            print("  top_k={} chunks={}".format(top_k, [r["chunk_id"] for r in results]))
            print("          reply: {}".format(reply))


if __name__ == "__main__":
    if "--topk-test" in sys.argv[1:]:
        topk_sensitivity_test()
    else:
        _outcomes = [run_agent(_ticket_id, _ticket) for _ticket_id, _ticket in SAMPLE_TICKETS]
        print_summary(_outcomes)
