## 1. What you *want* from the workhorse model

* **Multilingual bidirectionality (NL↔EN)** — good embeddings of Dutch text, and ability to answer in either language.
* **Instruction following** — reliable JSON / table outputs.
* **Context size** — ideally enough to hold your whole book (~128k tokens).

On those axes today:

* **Aya-35B**: good Dutch, solid instruction-following, but only **8k context**.
* **Llama 3.1 8B/70B, Mistral NeMo 12B, Qwen2.5 7B/14B**: **128k context**, better instruction tuning than Aya, decent multilingual (though Aya has an edge on Dutch specifically).

---

## 2. Why *not* just stuff the whole corpus in

Even if the model can take 128k tokens:

* **Recall is not guaranteed**: long-context models *degrade* with distance — they “pay less attention” to far-back tokens. Relevant facts may be overlooked if buried 80k tokens deep.
* **Latency & memory**: context cost scales roughly linearly with size. 128k context → much slower answers, much more VRAM used for KV cache (not the weights, but the running memory). On a 10GB card, that caps you around 32k–64k tokens practical.
* **Disambiguation**: genealogy isn’t just about recall — it’s about *which Jan van Santen* is which. Raw text doesn’t mark boundaries; retrieval/anchoring *does*.

---

## 3. Why RAG + RRF is still valuable

* **Focus**: with RAG, you give the model just the 2–4 chunks that mention the right Aart/Leendert, *plus their neighbors*. No attention dilution.
* **Anchors**: you attach structured keys (`II.1.a`), parent IDs, dates — things the model can’t infer from 100k raw tokens.
* **Control**: you can mix retrieval signals (vector, phonetic, trigram) to be robust to OCR spelling variation. That’s something long context alone doesn’t solve.
* **Re-use**: once you’ve built your index, you can apply it to multiple corpora (interviews, registers) without re-feeding huge contexts.

So: even with a 128k model, RAG gives **precision and interpretability**.

---

## 4. A practical compromise

* **Daily Q&A / extraction:** keep using **RAG + anchors**. It’s snappier, uses less VRAM, and gives you disambiguation.
* **Occasional “big sweep” jobs:** if you want to, e.g., “summarize the whole third generation,” then load 40k–60k tokens into a long-context model and let it reason over the big chunk.
* **Best of both worlds**: RAG for precision, long context for narrative/global questions.

---

## 5. Choosing the model

Given your priorities:

* **Aya-35B**: still best Dutch comprehension locally, but limited context.
* **Llama 3.1 8B**: 128k context, strong instruction following, reasonable NL handling. Light enough to fit fully in VRAM → very responsive.
* **Qwen2.5 7B**: also 128k, *excellent instruction-following*, surprisingly good cross-lingual ability, smaller VRAM footprint.
* **Mistral NeMo 12B**: strong reasoning, 128k, but a little more English-centric.

If your corpus really does fit under 128k, you might experiment with **Llama 3.1 8B or Qwen2.5 7B** as the “workhorse” for long-context tasks. But don’t throw away RAG + RRF: you’ll still want it for the sharp, name-focused queries where precision matters.

Awesome — here’s a clean, practical “**RAG-first, long-context fallback**” controller you can drop into your project. It keeps your fast, anchor-aware retrieval as the default, and only switches to a long-context model when the retriever can’t assemble enough evidence.

---

# Controller design (succinct)

1. **Try RAG** (hybrid: vectors + trigram/phonetic, RRF).
2. If you get **≥ N good passages**, expand to neighbor chunks and answer with **Aya-35B** (best NL↔EN + instruction following).
3. Else **fallback**: pack a **long context** (up to a token budget) and answer with a **128k model** (e.g., `llama3.1:8b` or `qwen2.5:7b`).

## Practical tips

* **Language control (NL↔EN):** prepend a short instruction to the user question like
  `"Answer in English. Quote Dutch names verbatim."`
  or detect with a tiny language classifier (small model) and set the rule automatically.
* **Structured output:** when you need JSON, set `options={"format":"json","temperature":0}` and validate; if invalid, reprompt *once* with the same context.
* **Observability:** log which path was used (`"rag"` vs `"long"`) and the lengths (chars/tokens) of contexts so you can tune thresholds later.

This gives you the best of both worlds: **snappy, accurate answers** most of the time, and a **safety net** for questions that truly need a large slice of the corpus.

# Suggested settings (Ollama `options`)

Use these as **starting points** and tune upward.

| Model                               | Quality role                         |    Quant to pull | `num_ctx` (start)                                     | Other useful knobs                                                         |
| ----------------------------------- | ------------------------------------ | ---------------: | ----------------------------------------------------- | -------------------------------------------------------------------------- |
| **Qwen 2.5 7B** or **Llama 3.1 8B** | long-context workhorse, EN↔NL decent | `q4_0` or `q5_0` | **64 k** (try 128 k for special runs)                 | `"num_batch": 256`, `"num_gpu": 1`                                         |
| **Mistral-NeMo 12B**                | long-context reasoning               |           `q4_0` | **64 k**                                              | `"num_batch": 128–192`                                                     |
| **Aya 35B (23)**                    | Dutch accuracy / extraction          |           `q4_0` | **24–32 k** (8 k is default; you can push higher now) | `"num_batch": 64–128`                                                      |
| **Llama 3.1 70B**                   | occasional heavy reasoning           |           `q4_0` | **8–16 k** (keep modest)                              | `"num_gpu_layers": 60–100` (let Ollama auto-tune; offload as much as fits) |

Example per-call:

```json
{
  "model": "llama3.1:8b",
  "prompt": "...",
  "options": { "num_ctx": 65536, "num_batch": 256, "temperature": 0.2 }
}
```

For Aya 35B:

```json
{
  "model": "aya:35b-23",
  "prompt": "...",
  "options": { "num_ctx": 32768, "num_batch": 96, "temperature": 0 }
}
```

For a 70B test:

```json
{
  "model": "llama3.1:70b",
  "prompt": "...",
  "options": { "num_ctx": 12000, "num_gpu_layers": 80, "num_batch": 64 }
}
```

# How to think about VRAM usage

* **Model weights** (quantized): fixed cost (e.g., ~5–8 GB for 7–12B Q4; ~18–22 GB for 35B Q4; ~35–40 GB for 70B Q4, but you’ll offload the remainder to CPU when VRAM is insufficient).
* **KV-cache** (context): grows with `num_ctx` and model size. With 32 GB you can comfortably run:

  * **7–12B** at **64k** contexts for interactive use.
  * **35B** at **24–32k** without stutters.
  * **70B** at **8–16k**; higher is possible but will start to hit RAM/PCIe limits.

# What I’d run day-to-day on your box

* **Workhorse (long context):** **Llama 3.1 8B** or **Qwen 2.5 7B** at 64k context for your “big slice” fallback mode.
* **Accuracy pass (Dutch extraction/Q&A):** **Aya 35B** with `num_ctx` 24–32k for structured outputs and bilingual answers.
* Keep a **small helper** (Qwen-7B or Llama-3.2-3B) around for paraphrasing/routing; with 32 GB you can run one small + one big concurrently.

# Quick tuning loop

1. Start with the table values.
2. Watch GPU use: `nvidia-smi` during a run (look at **memory** and **utilization**).
3. If memory headroom > 4 GB while generating, **bump `num_ctx`** or **`num_batch`** a notch.
4. If you see paging/slowdowns, drop `num_ctx` a step or reduce `num_batch`.

---

Bottom line: **32 GB VRAM lets you keep RAG for precision** while giving you a very capable **long-context fallback** (64k on 7–12B, 24–32k on 35B). You can also dabble with **70B** when you want, but expect slower throughput.

totally—let’s make a **“bring-your-own needle & haystack”** NIAH probe. you give real text (the haystack) and the exact needle(s); the script plants them at specific depths and checks recall (and an optional multi-hop).

below is a small, self-contained tool you can run against **ollama**.

---

## what it does

* reads one or more **haystack** files you provide (real OCR text, transcripts, etc.), concatenates them in order.
* **inserts your needle(s)** at exact positions (by **percent of total length** or **absolute character index**).
* builds a single prompt: `BRON BEGIN … BRON EINDE` with your content + needles.
* asks your model to **retrieve the value** (single-fact) or **infer the relationship** (two-hop) and reports pass/fail for each `num_ctx` you test.

---

## usage examples

```bash
# 1) single-fact recall at two depths using aya
python niah_custom.py \
  --model aya:35b-23 \
  --haystack files/page_001.txt files/page_002.txt files/page_003.txt \
  --needle "AART_BAPTISM_YEAR=1733@20%" \
  --needle "PLACE_ONE=HAAFTEN@85%" \
  --ctx 8192 12000 16000

# 2) two-hop relation, facts far apart, test a long-context model
python niah_custom.py \
  --model llama3.1:8b \
  --haystack docs/book_part1.txt docs/book_part2.txt \
  --needle "LINK1_CHILD_OF=AART→LEENDERT@10%" \
  --needle "LINK2_PARENT_OF=LEENDERT→DIRK@90%" \
  --ask "Using the FACTs, what is AART's relationship to DIRK?" \
  --expect GRANDCHILD \
  --ctx 16000 32000 64000
```

---

## the script: `niah_custom.py`

```python
#!/usr/bin/env python3
import argparse, json, math, re, sys, time
from pathlib import Path
from urllib import request

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_ollama(model, prompt, num_ctx, temperature=0.0):
    data = {"model": model, "prompt": prompt,
            "options": {"num_ctx": num_ctx, "temperature": temperature},
            "stream": False}
    req = request.Request(OLLAMA_URL, data=json.dumps(data).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read().decode("utf-8"))
        return out.get("response", "").strip()

def read_haystack(files):
    parts = []
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"[!] missing file: {f}", file=sys.stderr); sys.exit(1)
        parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)

def parse_needle(spec):
    # format: KEY=VALUE@POS   where POS like "20%" or "12345" (char index)
    if "@" not in spec or "=" not in spec:
        raise ValueError("needle must be KEY=VALUE@POSITION")
    kv, pos = spec.split("@", 1)
    key, val = kv.split("=", 1)
    return key.strip(), val.strip(), pos.strip()

def position_to_index(total_chars, pos):
    if pos.endswith("%"):
        pct = float(pos[:-1]) / 100.0
        return max(0, min(total_chars-1, int(total_chars * pct)))
    # absolute char index
    return max(0, min(total_chars-1, int(pos)))

def insert_needles(hay, needles):
    """
    needles: list[(key,val,pos_string)]
    returns (text_with_facts, placements[list of dict]])
    """
    total = len(hay)
    placements = []
    inserts = []
    for k, v, pos in needles:
        idx = position_to_index(total, pos)
        fact = f" FACT {k}={v} ENDFACT "
        inserts.append((idx, fact, k, v))
    inserts.sort(key=lambda x: x[0])

    out = []
    cursor = 0
    for idx, fact, k, v in inserts:
        out.append(hay[cursor:idx])
        out.append(fact)
        cursor = idx
        placements.append({"key": k, "value": v, "char_index": idx})
    out.append(hay[cursor:])
    context = "BRON BEGIN\n" + "".join(out) + "\nBRON EINDE\n"
    return context, placements

SYSTEM_SINGLE = ("You will be asked to retrieve exact values from BRON. "
                 "Answer EXACTLY the value; if unknown, reply UNKNOWN. No extra words.")
SYSTEM_FREE = ("Answer concisely and exactly according to BRON. If unsure, reply UNKNOWN.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--haystack", nargs="+", required=True, help="one or more text files")
    ap.add_argument("--needle", nargs="+", required=True,
                    help='KEY=VALUE@POSITION (POSITION like "20%%" or "12345")')
    ap.add_argument("--ctx", nargs="+", type=int, required=True, help="num_ctx values to test")
    ap.add_argument("--ask", default="", help="optional custom question (freeform)")
    ap.add_argument("--expect", default="", help="expected exact answer for --ask")
    args = ap.parse_args()

    hay = read_haystack(args.haystack)
    context, placements = insert_needles(hay, [parse_needle(n) for n in args.needle])

    # report where we inserted (approx tokens by chars/4 heuristic)
    print(f"\nBuilt haystack length: {len(context)} chars (~{max(1,len(context)//4)} tokens)")
    for p in placements:
        depth_pct = 100 * p["char_index"] / max(1, len(hay))
        print(f"  Inserted {p['key']} at char {p['char_index']} (~{depth_pct:.1f}% of haystack)")

    # build single-fact questions for each KEY unless user provided --ask
    single_qs = []
    if not args.ask:
        for p in placements:
            single_qs.append((p["key"], p["value"],
                              f"{SYSTEM_SINGLE}\n{context}\nWhat is the exact value for {p['key']}?"))

    # run probes
    results = []
    for nc in args.ctx:
        if single_qs:
            for key, exp, prompt in single_qs:
                got = ask_ollama(args.model, prompt, num_ctx=nc, temperature=0.0)
                ok = (got.strip().upper() == exp.upper())
                results.append(("single", nc, key, exp, got, ok))
                time.sleep(0.1)
        if args.ask:
            prompt = f"{SYSTEM_FREE}\n{context}\n{args.ask.strip()}"
            got = ask_ollama(args.model, prompt, num_ctx=nc, temperature=0.0)
            ok = (args.expect.strip() != "" and got.strip().upper() == args.expect.strip().upper())
            results.append(("free", nc, "QUESTION", args.expect, got, ok))
            time.sleep(0.1)

    # print compact report
    print("\nRESULTS:")
    by_kind = {}
    for kind, nc, key, exp, got, ok in results:
        by_kind.setdefault(kind, []).append((nc, key, exp, got, ok))
    for kind, rows in by_kind.items():
        print(f"\n[{kind.upper()}]")
        for nc, key, exp, got, ok in sorted(rows, key=lambda r: r[0]):
            mark = "✓" if ok else "·"
            print(f"  num_ctx={nc:>6} | {key:<16} | expected={exp:<20} | got={got[:40]:<40} | {mark}")

if __name__ == "__main__":
    main()
```

---

### tips

* **Positions:** use `@20%`, `@50%`, `@85%` to place needles early/middle/late. If you pass a big haystack, those positions are realistic.
* **Freeform checks:** with `--ask/--expect` you can test multi-hop (“What is AART’s relationship to DIRK?” expect `GRANDCHILD`) or any exact string you want.
* **Language:** the boilerplate is bilingual enough; tweak prompts if you’d like fully Dutch instructions.
* **Models:** run the same haystack/needles across `aya:35b-23`, `llama3.1:8b`, `qwen2.5:7b` and compare which `num_ctx` holds up.

this keeps the test **true to your data** (your haystack), **controlled** (you pick the needles & depths), and **repeatable** so you can choose the right workhorse settings with real evidence.


Totally fair. The version I gave still **inserts** `FACT … ENDFACT`, so it’s closer to the synthetic test than you want.

Here are two **truly different** ways to test, no artificial markers:

## A) “In-place” NIAH (no insertion at all)

You use your real haystack as-is. We pick **real facts already in the text**, compute their character positions, and ask questions whose answers must be those facts. We grade by exact match.

**How it differs:** zero markup; the model must find the answer in your own prose.

**Sketch:**

1. Concatenate your pages into one string `HAY`.
2. Provide a list of `(question, expected_answer)` pairs (e.g., “What is Aart’s baptism date?” → `1733-02-15`).
3. The script finds the **first occurrence index** of `expected_answer` inside `HAY` to get the *depth*.
4. Build the prompt as just:

   ```
   BRON BEGIN
   ...your real text only...
   BRON EINDE

   Q: <question>  (Answer with the exact value only.)
   ```
5. Run for various `num_ctx`, record correct/incorrect vs. the depth where that answer occurs.

> If a value appears multiple times, the script can take the **deepest** occurrence to make it harder.

## B) “Natural window” probe (low bias, minimal hint)

Still no insertion. We pick a **short quote near** the answer (e.g., a sentence before it), and include that one-line **lead-in** at the end of the prompt as a locating hint:

```
Hint (verbatim line from BRON): "Ondertrouw alhier 6.11.1761…"
```

Then ask: “What is the marriage date mentioned in BRON?”
This simulates realistic use where your retrieval brings the model *close* to the answer, but not on the exact line.

---

### Tiny code delta (conceptual)

If you liked the previous script’s structure, swap the “insert needles” part with:

```python
# Build HAY from your files (no insert)
HAY = "\n".join(Path(f).read_text() for f in haystack_files)

# Your gold questions/answers
GOLD = [
  ("What is Aart van Santen's baptism date?", "1733-02-15"),
  ("What place is listed for his death?", "Culemborg"),
]

# Measure depth by where the expected answer actually occurs in HAY
def answer_depth(hay, val):
    idx = hay.upper().rfind(val.upper())  # take the deepest occurrence
    return idx, 100.0 * idx / max(1, len(hay))

context = f"BRON BEGIN\n{HAY}\nBRON EINDE\n"
for num_ctx in [8192, 12000, 16000, 32000]:
    for (q, exp) in GOLD:
        prompt = f"You must answer with the exact value only.\n{context}\n{q}"
        got = call_ollama(model, prompt, num_ctx)
        ok = got.strip().upper() == exp.upper()
        depth_char, depth_pct = answer_depth(HAY, exp)
        print(num_ctx, depth_pct, exp, got, "OK" if ok else "MISS")
```

That’s it—no `FACT` tags, no filler.

---

### When to use which

* **A) In-place** is the purist test of long-context recall on *your prose*.
* **B) Natural window** mimics real usage with retrieval (a small hint), and is great for testing if a long context model actually helps over your RAG.

If you want, tell me one concrete Q→A from your corpus (just the exact string to expect), and I’ll hand you a ready-to-run snippet wired to that example.
