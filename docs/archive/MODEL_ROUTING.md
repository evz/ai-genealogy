## 1. Overall lineup

### 1) FAST INTERACTIVE CHAT (default for most queries)

**Model:** `llama3.1:8b` ([Ollama][1])

Use this for:

* Normal family-history questions
* Light reasoning over a few records
* Everyday interactive chat with your RAG stack

**Why it fits your 5090:**

* 4.7–4.9 GB Q4 on disk; fully in VRAM with tons of headroom ([Ollama][1])
* Very strong general chat model for its size.

**Pull & quick test:**

```bash
ollama pull llama3.1:8b
ollama run llama3.1:8b "In one paragraph, explain what a marriage record usually contains."
```

---

### 2) STRONGER GENERAL REASONER (multilingual, longer context)

**Model:** `qwen2.5:14b-instruct-q5_K_M` ([Ollama][2])

Use this for:

* Harder questions involving multiple documents
* Anything with **Dutch/Scandinavian** text or multi-language sources
* “What’s going on across these 5 records?” type summaries

This can be your **main “serious” genealogy model**, with Llama 3.1 8B as the “super snappy” option.

**Pull & test:**

```bash
ollama pull qwen2.5:14b-instruct-q5_K_M
ollama run qwen2.5:14b-instruct-q5_K_M "Explain the difference between a civil registration and a church record in genealogy."
```

Notes:

* ~11 GB Q5_K_M quant, still fine on a 32 GB card. ([Ollama][3])
* Good tradeoff between quality and speed.

---

### 3) HEAVY REASONING MODE (for identity/merge problems)

**Model:** `deepseek-r1:14b` (thinking mode; or `:8b` if you want it lighter) ([Ollama][4])

Use this only when:

* You’re reconciling **conflicting records**
* Trying to decide if **several entries are the same person**
* Doing complex reasoning over timelines, migrations, etc.

**Pull & test:**

```bash
ollama pull deepseek-r1:14b
ollama run deepseek-r1:14b "Think step by step: Given these two baptisms 5 years apart with similar parents' names, what are the possible explanations?"
```

DeepSeek-R1 will emit its inner “thinking”; you can strip or keep that in your app as you like.

---

## 2. Example Modelfiles tailored to your genealogy RAG

You can wrap each base model in a Modelfile so you bake in:

* System prompt tuned for genealogy
* Default parameters (temperature, etc.)

### A. `gene-chat-fast` (Llama 3.1 8B)

```Dockerfile
# Modelfile: gene-chat-fast
FROM llama3.1:8b

PARAMETER temperature 0.4
PARAMETER num_ctx 16384
PARAMETER num_predict 512
PARAMETER top_p 0.9

SYSTEM """
You are a genealogy research assistant embedded in a family history app.

Core rules:
- You work WITH retrieved context from tools (RAG) and must not invent records.
- If you are unsure, say so and list what additional records would be needed.
- Prefer concise, structured answers (lists, timelines, tables).
- Explain your reasoning briefly when comparing or reconciling records.

Domain details:
- You frequently see Dutch, Scandinavian and English records.
- Be careful with patronymics (e.g., Jansdatter / Jansdr, Jansen / Janszoon).
- Dates may be ambiguous; always specify the exact text and your interpretation.
- Do NOT assume two people are the same unless there is clear evidence.
"""
```

Load:

```bash
ollama create gene-chat-fast -f gene-chat-fast.Modelfile
```

Use:

```bash
ollama run gene-chat-fast
```

In your app, this is your **default** for most user queries.

---

### B. `gene-chat-main` (Qwen2.5 14B Instruct)

```Dockerfile
# Modelfile: gene-chat-main
FROM qwen2.5:14b-instruct-q5_K_M

PARAMETER temperature 0.35
PARAMETER num_ctx 32768
PARAMETER num_predict 768
PARAMETER top_p 0.9

SYSTEM """
You are the primary genealogy analysis model for a family history research tool.

- Focus on accurate reasoning over people, places, and dates.
- Always base conclusions on the provided context and tools, not on guesses.
- When merging or comparing individuals:
  - List each piece of evidence FOR and AGAINST a merge.
  - End with a clear judgement: 'same person', 'probably same person',
    'uncertain', or 'different people', and explain why.
- Handle Dutch and Scandinavian names/patronymics with care.
- Summaries should be short but precise, then add a bullet list of key facts.

If a user provides multiple records, consider:
- Name similarity (including spelling variants).
- Ages, occupations, addresses, parish names, and witnesses.
- Consistency of family members (spouses, children, parents).
"""
```

Create:

```bash
ollama create gene-chat-main -f gene-chat-main.Modelfile
```

Use this model from your backend for “serious” queries, or when your router thinks the question is complex.

---

### C. `gene-reasoner` (DeepSeek-R1 14B)

```Dockerfile
# Modelfile: gene-reasoner
FROM deepseek-r1:14b

PARAMETER temperature 0.3
PARAMETER num_ctx 32768
PARAMETER num_predict 1024
PARAMETER top_p 0.9

SYSTEM """
You are a specialized reasoning assistant for complex genealogy problems.

- Think step by step, but try to keep your reasoning as compact as possible.
- Your main tasks:
  1) Evaluate whether multiple records refer to the same person.
  2) Reconstruct plausible life timelines from fragmentary evidence.
  3) Identify contradictions and propose explanations (e.g., clerical error,
     second marriage, two cousins with similar names, etc.).
- You may propose multiple hypotheses with estimated likelihoods.

VERY IMPORTANT:
- Base every step of reasoning on specific details from the input.
- Do not invent records or events that weren’t mentioned.
- If evidence is insufficient, say so and suggest what additional records
  would most help (e.g., 'search civil marriage in Haarlem 1885–1895').
"""
```

Create:

```bash
ollama create gene-reasoner -f gene-reasoner.Modelfile
```

Use sparingly via routing in your app, e.g. only when the query involves:

* “Are these the same person?”
* “Which of these scenarios is more likely?”
* Merging clusters of extracted entities from your book.

---

## 3. How to wire this into your app (high-level)

In your genealogy backend, you can do something like:

* **Route to `gene-chat-fast`** if:

  * Short, straightforward question
  * Only 1–2 retrieved docs
* **Route to `gene-chat-main`** if:

  * Multiple docs / languages / longer answer needed
* **Route to `gene-reasoner`** if:

  * Your query classifier sees words like “same person”, “reconcile”, “merge”, “which is more likely”, etc., or
  * You’re running a background “merge suggestions” process

All three are just different `model` strings on the Ollama HTTP API, so it’s mostly configuration on your side.

## 1. Python routing function

Assumptions:

* You’re calling Ollama via its **OpenAI-compatible** HTTP API (`/v1/chat/completions`), but that doesn’t matter much here – routing is just returning a `model_name` string.
* You already have a list of retrieved docs like:

```python
@dataclass
class RetrievedDoc:
    id: str
    score: float
    text: str
    source: str | None = None  # e.g. "Haarlem civil births", "Family book p.32"
```

Here’s a simple router:

```python
import re
from dataclasses import dataclass
from typing import List

MERGE_KEYWORDS = [
    r"\bsame person\b",
    r"\bsame man\b",
    r"\bsame woman\b",
    r"\bduplicate(s)?\b",
    r"\breconcile\b",
    r"\bmerge\b",
    r"\bwhich of these\b",
    r"\bmore likely\b",
    r"\bconflicting\b",
    r"\bcontradict(?:ion|ory)\b",
]

@dataclass
class RetrievedDoc:
    id: str
    score: float
    text: str
    source: str | None = None


def is_merge_or_conflict_query(query: str) -> bool:
    q = query.lower()
    return any(re.search(pat, q) for pat in MERGE_KEYWORDS)


def estimate_complexity(query: str, docs: List[RetrievedDoc]) -> int:
    """
    Very crude 'complexity' score: length of question + number/size of docs.
    You can replace this later with something smarter.
    """
    q_len = len(query.split())
    total_doc_chars = sum(len(d.text) for d in docs)
    n_docs = len(docs)

    score = 0
    if q_len > 40:
        score += 1
    if n_docs >= 5:
        score += 1
    if total_doc_chars > 8000:
        score += 1
    return score


def choose_genealogy_model(
    query: str,
    docs: List[RetrievedDoc],
) -> str:
    """
    Returns the Ollama model name:
      - 'gene-reasoner'     (deepseek-r1 wrapper)
      - 'gene-chat-main'    (qwen2.5 14b wrapper)
      - 'gene-chat-fast'    (llama 3.1 8b wrapper)
    """

    # 1) Identity / merge / conflict resolution → heavy reasoning
    if is_merge_or_conflict_query(query):
        return "gene-reasoner"

    complexity = estimate_complexity(query, docs)

    # 2) Moderately complex query or big context → main model
    if complexity >= 2:
        return "gene-chat-main"

    # 3) Everything else → fast model
    return "gene-chat-fast"
```

You’d use it roughly like:

```python
docs = retrieve_docs(query)  # your RAG+RRF output
model_name = choose_genealogy_model(query, docs)
prompt = build_general_rag_prompt(query, docs)  # see below
response = call_ollama_chat(model_name, prompt)
```

---

## 2. RAG prompt shapes (templates)

### A. General genealogy Q&A (for `gene-chat-fast` / `gene-chat-main`)

Goal: keep it **structured**, avoid hallucinations, and encourage clear evidence-based answers.

Here’s a Python helper to build a single **user message** from query + docs:

```python
from textwrap import dedent
from typing import List

def format_docs_for_prompt(docs: List[RetrievedDoc], max_chars: int = 8000) -> str:
    """
    Format retrieved docs as numbered context blocks.
    Truncates roughly to max_chars total.
    """
    chunks = []
    used = 0
    for i, d in enumerate(docs, start=1):
        header_parts = []
        if d.source:
            header_parts.append(f"Source: {d.source}")
        header_parts.append(f"Doc ID: {d.id}")
        header = " | ".join(header_parts)

        body = d.text.strip()
        block = f"[Document {i}]\n{header}\n\n{body}\n"
        if used + len(block) > max_chars:
            break
        chunks.append(block)
        used += len(block)

    if not chunks:
        return "No external documents were retrieved for this query."

    return "\n\n".join(chunks)


def build_general_rag_prompt(query: str, docs: List[RetrievedDoc]) -> list:
    """
    Returns a list of messages suitable for the OpenAI-style /v1/chat/completions API.
    System message can also live in your Modelfile; here we keep it in-code for clarity.
    """
    context_block = format_docs_for_prompt(docs)

    system_msg = dedent("""
        You are a genealogy research assistant embedded in a family history app.

        Use ONLY the information in the provided documents and the user question.
        If something is not supported by the documents, say you do not know.
        Prefer short, precise answers followed by bullet-pointed evidence.

        When relevant:
        - Mention people with full name and approximate life dates if known.
        - Clearly distinguish between facts (directly stated) and inferences.
        - If you infer something (e.g., likely same person), explain why.
        - If evidence is weak or ambiguous, say so explicitly.
    """).strip()

    user_msg = dedent(f"""
        User question:
        {query.strip()}

        Retrieved documents:
        {context_block}

        Instructions:
        1. First, summarize briefly what the question is asking.
        2. Then, answer using ONLY the information from the documents.
        3. Cite document numbers when you refer to specific evidence, e.g. (Doc 2).
        4. If the answer is uncertain or incomplete, explain what additional records
           would help clarify the situation.
    """).strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
```

Use with any of your models:

```python
messages = build_general_rag_prompt(query, docs)
response = call_ollama_chat("gene-chat-main", messages)
```

(where `call_ollama_chat` is your wrapper around the HTTP call).

---

### B. Complex identity / merge reasoning (for `gene-reasoner`)

For DeepSeek-R1 / `gene-reasoner`, you want to **encourage stepwise reasoning**, but also keep it constrained to your evidence.

Prompt builder:

```python
def build_merge_reasoning_prompt(query: str, docs: List[RetrievedDoc]) -> list:
    context_block = format_docs_for_prompt(docs, max_chars=10000)

    system_msg = dedent("""
        You are a specialist in genealogical reasoning.

        Your primary job in this mode is to:
        - Decide whether records describe the same person or different people.
        - Reconstruct plausible life histories from fragmentary evidence.
        - Identify contradictions and propose possible explanations.

        Rules:
        - Base every step of reasoning on details from the documents.
        - Do NOT invent additional people, events, or dates.
        - If the data is insufficient for a firm conclusion, say so and give
          one or more plausible hypotheses with your confidence.

        Output structure:
        1. Brief restatement of the problem.
        2. Step-by-step reasoning referencing document numbers.
        3. Conclusion:
           - same person
           - probably same person
           - uncertain
           - different people
        4. Suggestions for further research (specific record types, places, and years).
    """).strip()

    user_msg = dedent(f"""
        User question:
        {query.strip()}

        Retrieved documents (these may be different records that could refer to one or more people):
        {context_block}

        Please follow the output structure exactly.
    """).strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
```

Routing + usage:

```python
def build_messages_for_model(model_name: str, query: str, docs: List[RetrievedDoc]) -> list:
    if model_name == "gene-reasoner":
        return build_merge_reasoning_prompt(query, docs)
    else:
        return build_general_rag_prompt(query, docs)


def answer_query(query: str, retrieved_docs: List[RetrievedDoc]) -> str:
    model_name = choose_genealogy_model(query, retrieved_docs)
    messages = build_messages_for_model(model_name, query, retrieved_docs)
    return call_ollama_chat(model_name, messages)
```

---

### C. Minimal `call_ollama_chat` example (for completeness)

If you’re using the OpenAI-compatible API:

```python
import requests

def call_ollama_chat(model: str, messages: list) -> str:
    url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
```

You can of course override temperature / num_ctx etc. in the Modelfile instead of here.
