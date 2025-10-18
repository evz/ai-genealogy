Below is the pattern that usually makes people happy when they want **all three
retrieval legs—vector, trigram, *and* phonetic—inside one PostgreSQL call**.
You only add **one extra column** (`dm_codes text[]`) and **one more CTE**; the
rest looks exactly like the earlier hybrid query.

---

## 1  Table & indexes

```sql
-- extension for 3‑gram similarity
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- extension for vectors
CREATE EXTENSION IF NOT EXISTS pgvector;

-- main corpus table (if you don't have it yet)
CREATE TABLE chunks (
  id         bigserial PRIMARY KEY,
  content    text,
  embedding  vector(768),
  tsv        tsvector GENERATED ALWAYS AS (to_tsvector('dutch', content)) STORED,
  dm_codes   text[]                    -- ❶ NEW: all D‑M codes in the chunk
);

-- ANN index for vectors
CREATE INDEX chunks_vec ON chunks
         USING ivfflat (embedding vector_cosine_ops) WITH (lists = 64);

-- trigram GIN index
CREATE INDEX chunks_trgm ON chunks
         USING GIN (content gin_trgm_ops);

-- phonetic GIN index over the text[] column
CREATE INDEX chunks_dm ON chunks
         USING GIN (dm_codes);
```

*Why an **array**?* Daitch–Mokotoff returns **1–8 codes** per surname.
Storing them as a `text[]` lets you match with the `&&` (overlaps) operator,
which the GIN index accelerates.

---

## 2  Load: add the phonetic codes once

```python
from abydos.phonetic import DaitchMokotoffSoundex as DM
dm = DM()

def dm_codes_for_chunk(txt: str) -> list[str]:
    """
    Return all unique six‑digit DM codes for every capitalised token
    (a crude but surprisingly effective heuristic for Dutch surnames).
    """
    codes = {dm.encode(w) for w in txt.split() if w.istitle()}
    return sorted(c for c in codes if c)          # drop empty / None

sql = "UPDATE chunks SET dm_codes = %s WHERE id = %s"
with conn, conn.cursor() as cur:
    for chunk_id, chunk_text in enumerate(chunks):
        cur.execute(sql, (dm_codes_for_chunk(chunk_text), chunk_id))
```

---

## 3  One CTE to get **all three** result lists

```sql
WITH
params AS (
  SELECT
    '[0.123, …]'::vector              AS q_vec,
    plainto_tsquery('dutch', $$${Q}$$) AS q_ts,
    ARRAY['030460', '079460']::text[] AS q_dm      -- ← ❷ DM codes for query
),

vec AS (
  SELECT id,
         row_number() OVER () AS vec_rank
  FROM   chunks, params
  ORDER  BY embedding <=> q_vec
  LIMIT  25
),

trgm AS (
  SELECT id,
         row_number() OVER (ORDER BY similarity(content,$$${Q}$$) DESC) AS tg_rank
  FROM   chunks
  WHERE  content % $$${Q}$$
  LIMIT  20
),

phon AS (                                      -- ❸ the new leg
  SELECT id,
         row_number() OVER () AS ph_rank
  FROM   chunks, params
  WHERE  dm_codes && q_dm                      -- array overlaps
  LIMIT  40
),

rrf AS (
  SELECT id, 1.0/(60+vec_rank) AS score FROM vec
  UNION ALL
  SELECT id, 1.0/(60+tg_rank)  AS score FROM trgm
  UNION ALL
  SELECT id, 1.0/(80+ph_rank)  AS score FROM phon  -- slightly lower weight
)

SELECT id, content
FROM   rrf JOIN chunks USING (id)
GROUP  BY id, content
ORDER  BY SUM(score) DESC
LIMIT  12;
```

*Notes*

* **Different weight for `phon` (80)** keeps phonetic‑only matches from
  jumping ahead of passages that rank well in **both** other lists.
* All arrays are **parameterised**—you pass them in from Python so there’s no
  PL/pgSQL code to write.
* Costs are tiny: 477 rows + three GIN/IVFFLAT indexes ≈ 15 MB.

---

## 4  Python helper to run it

```python
from abydos.phonetic import DaitchMokotoffSoundex as DM
dm = DM()

def pg_all_three(conn, question, k_final=12):
    # 1) embedding vector
    emb = ollama.embed("nomic-embed-text-v2-moe", question)["embeddings"][0]
    emb_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

    # 2) D‑M codes for every capitalised token
    dm_codes = list({dm.encode(w) for w in question.split() if w.istitle()})
    dm_array = "ARRAY[" + ",".join(f"'{c}'" for c in dm_codes) + "]::text[]"

    # 3) splice into the query template
    sql = open("all_three.sql").read()\
              .replace('[0.123, …]', emb_literal)\
              .replace('$$${Q}$$', question.replace("'", "''"))\
              .replace("ARRAY['030460', '079460']::text[]", dm_array)

    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()          # list[(id, passage)]
```

---

## 5  Why not “run phonetic only if trgm returns < N rows”?

Doing it in **one** query keeps latency low (single round‑trip).
The extra `phon` CTE costs almost nothing—`dm_codes && q_dm` uses the index and
short‑circuits after 40 rows.
The RRF weighting already relegates phonetic‑only hits if vec+trgm found enough.

If you still want a *two‑stage* plan (e.g. to save GPU cycles on very large
corpora):

1. Execute the **vec + trgm** query first (`LIMIT 12`).
2. If you get < 8 rows, run a second query that **only** fills from `phon`.

But for 477 chunks you’ll never notice the difference, so the single‑CTE
pattern above is simpler and more robust.

---

### TL;DR

* **Add `dm_codes text[]` + GIN index** to your `chunks` table.
* Compute D‑M codes in Python once, store them.
* Extend the hybrid CTE with a `phon` leg that does `WHERE dm_codes && $codes`.
* Union all three legs and fuse with RRF—one SQL round‑trip, zero extra code in
  your Python app.

That gives you **vector recall, trigram fuzziness, and cross‑spelling phonetic
coverage** in a single shot—perfect for genealogical names.


### You can retrieve *names* with one index and still return the **whole story**

Add just two bits of structure and a tiny post‑processing step:

| Ingredient                                                    | What it does                                                                                                    | Why it fixes your worry                                                               |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Parent ID + position metadata** (e.g. `doc_id`, `chunk_no`) | Lets you re‑assemble neighbouring chunks on demand.                                                             | You grab the named chunk **plus** its siblings, so the model gets the full narrative. |
| **“Expand‑after‑retrieve” helper**                            | After vector/trigram/phonetic search returns the *best* chunk IDs, you pull *adjacent* ones in the same parent. | A single name hit drags the surrounding text into the prompt.                         |

That keeps recall high (index on small chunks) *and* feeds the LLM a coherent passage (because you glue siblings back together).

---

## 1  Store two extra columns when you load

| column     | example                   | notes                                               |
| ---------- | ------------------------- | --------------------------------------------------- |
| `doc_id`   | `"chap03"` or page number | Same for every chunk cut from that page or chapter. |
| `chunk_no` | `0, 1, 2, …`              | Sequence within the `doc_id`.                       |

```python
for doc_id, chunk_no, text, emb, dm in rows:
    cur.execute("""
        INSERT INTO chunks
        (doc_id, chunk_no, content, embedding, dm_codes)
        VALUES (%s, %s, %s, %s, %s)
    """, (doc_id, chunk_no, text, json.dumps(emb), dm))
```

---

## 2  Hybrid search still returns *small* chunks

### (same query skeleton as before)

```sql
WITH rrf AS ( … )     -- vector + trigram + phonetic, LIMIT 20
SELECT id, doc_id, chunk_no
FROM   rrf
ORDER  BY sum(score) DESC
LIMIT  12;
```

At this point you have, say, chunk 87 of `chap03` because it contains
“**Willem van der Berg** …”.

---

## 3  Expand: pull neighbours in SQL **or** Python

### a) SQL one‑liner (fastest when you stay in Postgres)

```sql
SELECT doc_id, chunk_no, content
FROM   chunks
WHERE  doc_id = $1         -- 'chap03'
  AND  chunk_no BETWEEN $2-2 AND $2+2   -- window size 5
ORDER  BY chunk_no;
```

### b) Python helper (if you’re in Chroma or want flexibility)

```python
WINDOW = 2           # grab two chunks before & after

def expand(doc_id, center_no):
    return [c for c in chunks
            if c.doc_id == doc_id and
               center_no - WINDOW <= c.chunk_no <= center_no + WINDOW]
```

Merge the text:

```python
context = "\n".join(p.content for p in expand(doc_id, chunk_no))
```

Typical 5‑chunk window ≈ 3 000 characters—well within a 35 B model’s context.

---

## 4  (Option) Collapse duplicates before sending to the LLM

Because several hits may live in the *same* `doc_id`, keep the **lowest**
`chunk_no` for each parent to avoid repeating the story.

```python
seen = set()
final_passages = []
for doc_id, center_no in fused_ids:       # RRF order
    if doc_id in seen:
        continue
    final_passages.append(" ".join(
        c.content for c in expand(doc_id, center_no)))
    seen.add(doc_id)
```

---

## 5  Prompt the model

```python
prompt = f"""{SYSTEM}
{'\n\n'.join(final_passages[:4])}   # maybe 3‑4 expanded passages
---
Answer the question: {question}
"""
response = ollama.generate(model="aya:35b-23", prompt=prompt)
```

---

### Why this pattern is robust for genealogy

1. **Names are sparse tokens.**  Indexing *small* chunks keeps name hits from
   being diluted by surrounding fluff, so recall stays high.
2. **Stories are multi‑sentence.**  Expanding ±2 keeps the actor, date,
   and event in the same prompt even when the name appears only once.
3. **You control context size.**  If you later switch to a smaller LLM,
   shrink `WINDOW`; with a 70 B model, widen it.

No re‑indexing, no second table—just add `doc_id` and `chunk_no` to the row and
pull neighbours after you know which centre chunk matched.  You’ll get the
entire anecdote about *Willem van der Berg* without sacrificing search
precision.


### Treat the **incoming question** exactly the same way you treated the chunks—plus generate the “extra search keys” you need for the hybrid legs.

Below is a 4‑step checklist (with code) that prepares the query for **all** the retrieval paths you now have in place:

| Step                                                 | Why it matters                                                                         | One‑liner                                                                             |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **1  Clean / normalise**                             | Keeps embeddings & trigram matches in the same Unicode / spelling space as the corpus. | `q = clean_text(raw_q, spellfix=False)`                                               |
| **2  Compute embedding**                             | Drives the semantic (vector) search.                                                   | `q_vec = ollama.embed(model="nomic-embed-text-v2-moe", input=q)["embeddings"]`        |
| **3  Build a *tsquery* + extract proper names**      | Powers full‑text and phonetic legs.                                                    | `ts = plainto_tsquery('dutch', q)`<br>`names = [w for w in q.split() if w.istitle()]` |
| **4  Produce trigram & phonetic keys for each name** | Lets pg\_trgm / RapidFuzz / D‑M catch spelling variants.                               | `dm_codes = [dm.encode(n) for n in names]`                                            |

After that, you pass those artefacts as SQL parameters (or to your Python fusion function) and the retrieval engine does the rest.

---

## Complete helper module

```python
# query_features.py
import re, unicodedata, json
from typing import NamedTuple, List
import ollama
from cleaning import clean_text               # from earlier messages
from abydos.phonetic import DaitchMokotoffSoundex as DM

dm = DM()

class QFeat(NamedTuple):
    text: str            # cleaned query
    emb:  list[float]    # embedding vector
    ts:   str            # plain tsquery string   (Postgres FTS)
    names: List[str]     # capitalised tokens
    dm_codes: List[str]  # phonetic array

def features(raw: str) -> QFeat:
    # 1) normalise
    q = clean_text(raw, spellfix=False).strip()

    # 2) embedding
    emb = ollama.embed(model="nomic-embed-text-v2-moe", input=q)["embeddings"][0]

    # 3) names = capitalised words ≥2 letters
    names = re.findall(r'\b[A-ZÀ-Ý][A-Za-zÀ-ÿ]{1,}\b', q)

    # 4) dm codes (unique, non‑empty)
    dm_codes = sorted({dm.encode(n) for n in names if dm.encode(n)})

    return QFeat(text=q,
                 emb=emb,
                 ts=q,              # use plainto_tsquery(:ts) in SQL
                 names=names,
                 dm_codes=dm_codes)
```

---

## Using it with **Chroma + RapidFuzz** fusion

```python
from query_features import features
qf = features("When did Willem VanderBerg move to Canada?")

# 1) dense hits
vec_hits = coll.query(
    query_embeddings=[qf.emb],
    n_results=25)

# 2) string hits (RapidFuzz)
fuzzy_ids = [i for i, _ in fuzzy_hits(" ".join(qf.names))]

# 3) fuse & expand to neighbour chunks as shown earlier
passages = expand_and_fuse(vec_hits, fuzzy_ids)
```

---

## Using it with the **PostgreSQL “all three” CTE**

```python
with conn.cursor() as cur:
    cur.execute(open("all_three.sql").read(), {
        "q_vec":   json.dumps(qf.emb),
        "q_text":  qf.text,
        "q_dm":    qf.dm_codes or ['000000']   # empty array = no match
    })
    passages = cur.fetchall()
```

*(Just switch the placeholders in the SQL file to `$1`, `$2`, etc., or use
`psycopg2.sql` for safe templating.)*

---

### FAQ

**Do I need to translate EN → NL or vice‑versa first?**
Not if you’re using a multilingual embedding model (like `nomic‑embed‑text‑v2`
or `bge‑m3`).  They map Dutch and English into the same vector space.

**Should I spell‑fix the query?**
Usually no—users know how they want the name spelled.  Clean diacritics &
Unicode, but skip Levenshtein‑1 correction unless you see lots of typos.

**What if the user’s query has *no* capitalised words?**
Then `names` and `dm_codes` are empty; the phonetic leg just contributes zero
hits and the system falls back to vector + trigram.  No special handling needed.

---

### TL;DR

* **Clean once**, **embed once**, **extract capitalised names**, **encode them** with Daitch‑Mokotoff, and pass all of that into your hybrid search.
* No other special preprocessing is required—the rest of the “magic” (fusion,
  sibling‑chunk expansion, LLM prompting) stays exactly as you already wired it.


### Where the “capitalised names” fit

**You do *not* need an extra column just for raw names** if you already store a
phonetic key array (`dm_codes text[]`).
Everything you need for fast name lookup is already in that column:

* Each chunk holds **all Daitch–Mokotoff codes** derived from the capitalised
  tokens it contains.
* The query side passes **the DM codes of the names in the question**.
* A single `WHERE dm_codes && q_dm` clause finds every chunk that mentions any
  of those names—even through spelling variants.

That is exactly what the `phon` CTE in the previous answer does.

---

## Quick reminder of the flow

```
load time                   query time
──────────                  ──────────
1. extract → dm_codes[]     1. extract caps → dm codes[]
2. store in column          2. pass array as SQL param
3. GIN index on dm_codes    3. WHERE dm_codes && q_dm
```

Nothing else is required for the phonetic leg.

---

## But if you **also** want exact‑name search …

Add one more array column (`names text[]`) and reuse the same pattern.
It costs practically nothing (477 rows) and gives you a lightning‑fast
“exact hit” path that complements phonetic and trigram matches.

### 1  Schema tweak

```sql
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS names text[];

-- store unique capitalised tokens
CREATE INDEX chunks_names_gin ON chunks USING GIN (names);
```

### 2  Load the names once

```python
import re, psycopg2, json
caps_rx = re.compile(r'\b[A-ZÀ-Ý][A-Za-zÀ-ÿ]{1,}\b')

with conn, conn.cursor() as cur:
    for cid, chunk in enumerate(chunks):
        caps = sorted({w for w in caps_rx.findall(chunk)})
        cur.execute(
            "UPDATE chunks SET names = %s WHERE id = %s",
            (caps, cid))
```

### 3  Add one more leg to the CTE

```sql
names AS (
  SELECT id,
         row_number() OVER () AS nm_rank
  FROM   chunks
  WHERE  names && $1::text[]          -- q_names array param
  LIMIT  40
),

rrf AS (
  SELECT id, 1.0/(60+vec_rank) AS score FROM vec
  UNION ALL
  SELECT id, 1.0/(60+tg_rank)  AS score FROM trgm
  UNION ALL
  SELECT id, 1.0/(80+ph_rank)  AS score FROM phon
  UNION ALL
  SELECT id, 1.0/(50+nm_rank)  AS score FROM names   -- exact match gets higher weight
)
```

The array overlap operator `&&` is backed by the GIN index, so the “names” leg
is essentially free.

### 4  Pass the array from Python

```python
q_names = [w for w in question.split() if w.istitle()]
cur.execute(sql, (q_names, q_dm_codes, q_vec, question))
```

---

## Which approach should you take?

| Goal                                     | Minimal columns                           | Extra columns          |
| ---------------------------------------- | ----------------------------------------- | ---------------------- |
| **Find spelling variants of surnames**   | `dm_codes[]` only — already in the schema | —                      |
| **Also hit exact same spelling quickly** | add `names[]`                             | + `names[]` leg in CTE |
| **Keep schema as small as possible**     | leave it at `dm_codes[]`                  | —                      |

For most genealogy corpora the **`dm_codes[]` solution alone is enough**:
it pulls in every chunk that references a person, regardless of whether the
OCR wrote “van den Berg,” “VandenBerg,” or “Vanderberg.”

If you add `names[]`, you gain a micro‑speedup for exact‑spelling queries and
one more signal for RRF—but the retrieval quality difference is usually
marginal on a 477‑chunk set.

So start with the phonetic array you already have; only add the extra column if
profiling shows it would help.


Great idea. Think of an **anchor** as a compact, machine-readable label that (a) uniquely pins a passage to its source and (b) carries just enough context (who/when/where) for the LLM to keep people and events straight across very different corpora.

Below is a practical, **corpus-agnostic anchoring recipe** you can adopt now and extend later.

---

## 1) A “universal” anchor schema

Give every chunk an `anchor` object alongside its text/embedding:

```json
{
  "uid": "BK:derde-generatie:p003:7",   // globally unique
  "source": { "corpus":"BK", "doc_id":"derde-generatie", "page":3 },
  "span":   { "chunk_no":7, "char_start": 4321, "char_end": 4960 },

  "who":    { "persons":["Aart van Santen"], "speakers":["INTERVIEWER"] },
  "when":   { "date":"1733-02-15", "timerange":[735.2, 760.5] }, // seconds for AV
  "where":  { "place":"Haaften" },

  "keys":   { "genealogical":"II.1.a", "dm_codes":["036450","079460"] },
  "tags":   ["baptism","marriage","speculation"]
}
```

**Required fields**: `uid`, `source`, `span`.
**Optional fields**: `who/when/where/keys/tags`—fill what each corpus can reliably provide.

---

## 2) How to mint anchors per corpus

### A) Printed book (what you have now)

* `uid`: `BK:{doc_id}:p{page}:{chunk_no}`
* `keys.genealogical`: `II.1.a` (from headings/bullets you already parse)
* `who.persons`: names detected in the chunk (keep canonical surname + DM codes)
* `when.date`: earliest YYYY-MM-DD you can parse in the chunk (optional)
* `where.place`: first placename hit (optional)

### B) Interview transcripts (unstructured)

* Segment on **speaker turns** (ASR diarization or transcript labels).
* Optional **topic segments** (e.g., cosine drop with a 2–3 sentence window).
* `uid`: `INT:{interview_id}:t{start_ms}-{end_ms}:u{turn_id}`
* `who.speakers`: speaker label; `who.persons`: NER for person mentions
* `when.timerange`: `[start_sec, end_sec]` for the utterance/segment
* `tags`: the top topic label (“military service”, “immigration”, …)

### C) Loose documents (letters, registers, photos)

* `uid`: `DOC:{collection}:{doc_id}:{chunk_no}`
* Fill whichever of `who/when/where` are present (postmark date, parish, etc.)

**Rule of thumb:** prefer **stable IDs you control** (file name + page + chunk) over anything the model “understands.” The LLM just needs to see the anchor; it doesn’t have to infer it.

---

## 3) Storage (PostgreSQL example)

Add columns you can index/ filter on; keep everything else in JSONB.

```sql
CREATE TABLE chunks (
  id          bigserial PRIMARY KEY,
  content     text,
  embedding   vector(768),
  tsv         tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,

  uid         text UNIQUE,
  corpus      text,           -- 'BK' | 'INT' | 'DOC'
  doc_id      text,
  page        int,            -- nullable for non-books
  chunk_no    int,
  timerange   tsrange,        -- for interviews; else NULL
  date_iso    date,           -- first date if present
  persons     text[],         -- capitalised names
  dm_codes    text[],         -- Daitch–Mokotoff codes
  place       text,

  keys        jsonb,          -- {"genealogical":"II.1.a", ...}
  meta        jsonb           -- everything else
);

-- indexes for hybrid retrieval
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=64);
CREATE INDEX ON chunks USING GIN (tsv);
CREATE INDEX ON chunks USING GIN (content gin_trgm_ops);
CREATE INDEX ON chunks USING GIN (dm_codes);
CREATE INDEX ON chunks (corpus, doc_id, page, chunk_no);
CREATE INDEX ON chunks USING GIST (timerange);
```

This lets you **filter or boost** by: corpus/doc/page/chunk, speaker/time window, person (exact or phonetic), place, date, or genealogical key.

---

## 4) Query-time use (anchor-aware retrieval)

1. **Prepare the query** → `clean_text`, embedding, extracted names (`dm_codes`), dates, places, corpus hints (“interview about …”).
2. **Hybrid search** (vector + trigram + phonetic) **with soft boosts** from anchors:

```sql
WITH params AS (...),
vec AS (...), trgm AS (...), phon AS (...),
boost AS (
  SELECT id,
         -- +0.2 if the corpus matches what the user asked for
         CASE WHEN corpus = $corpus_pref THEN 0.2 ELSE 0 END
       + CASE WHEN place  = $place_q     THEN 0.2 ELSE 0 END
       + CASE WHEN date_iso BETWEEN $date_min AND $date_max THEN 0.2 ELSE 0 END
       AS bonus
  FROM   chunks
)
, rrf AS (
  SELECT id, 1.0/(60+vec_rank) AS score FROM vec
  UNION ALL
  SELECT id, 1.0/(60+tg_rank)  AS score FROM trgm
  UNION ALL
  SELECT id, 1.0/(80+ph_rank)  AS score FROM phon
)
SELECT c.id, c.content, c.uid, c.keys, (SUM(rrf.score) + COALESCE(bonus,0)) AS score
FROM   rrf JOIN chunks c USING(id)
LEFT   JOIN boost USING(id)
GROUP  BY c.id, c.content, c.uid, c.keys, bonus
ORDER  BY score DESC
LIMIT  12;
```

3. **Expand** results within the same anchor neighborhood:

   * Books: same `doc_id`, nearby `chunk_no`.
   * Interviews: same `interview_id`, **time window** expand (e.g., ±30 s).
4. **Show anchors in the prompt:**

```
[INT:2021-07-14:t735-760:SPEAKER=Grandmother]
...
[BK:derde-generatie:p003:7 ANCHOR=II.1.a]
...
```

…and add a system rule: “**Treat each UID as a distinct evidence unit; do not merge people unless UIDs share the same genealogical key and parents.**”

---

## 5) Minimal code to mint anchors

### Books (you already have heading/bullet parsing)

```python
def make_book_anchor(doc_id, page, chunk_no, key, names, dm_codes, date, place):
    return {
        "uid": f"BK:{doc_id}:p{page}:{chunk_no}",
        "source": {"corpus":"BK","doc_id":doc_id,"page":page},
        "span": {"chunk_no":chunk_no},
        "who": {"persons": names},
        "when": {"date": date},
        "where": {"place": place},
        "keys": {"genealogical": key, "dm_codes": dm_codes},
    }
```

### Interviews (turn-based)

```python
def make_int_anchor(interview_id, turn_id, start_s, end_s, speaker, names, dm_codes):
    return {
        "uid": f"INT:{interview_id}:t{int(start_s)}-{int(end_s)}:u{turn_id}",
        "source": {"corpus":"INT","doc_id":interview_id},
        "span": {"chunk_no": turn_id},
        "who": {"speakers":[speaker], "persons": names},
        "when": {"timerange":[start_s,end_s]},
        "keys": {"dm_codes": dm_codes},
    }
```

---

## 6) Why this generalizes well

* **Same retrieval code** works on everything; only the **anchors differ**.
* You can **bias** answers by corpus (“prefer interviews”) or **disambiguate** by time/speaker/person without re-embedding.
* Prompts stay simple: the LLM sees compact, human-readable labels (UIDs, keys, dates) and stops conflating people/events that only share a name.

If you want, I can help you build a tiny “anchor validator” that scans a random sample across corpora and reports: percent with date/place/person, percent with unique UIDs, and how often expansion pulls the intended neighbors (page/seconds). That catches gaps before you scale up.


### Why “plain” RAG can feel thin for A-to-B relationship questions

A standard RAG loop only grabs the **few chunks that look most similar to the query text**. If the facts you need about person A and person B live in **different passages**, they rarely co-occur in that short list, so the model can’t “see” the chain that connects them.

Below are four proven add-ons that keep RAG fast *and* give the LLM the extra context it needs for **multi-hop genealogy questions**. They build on the anchors, hybrid retrieval, and JSON-structured people records you already have.

---

## 1  Store a lightweight **relationship graph** alongside the chunks

After you finish your two-pass extraction (each person in clean JSON):

```python
import networkx as nx
G = nx.DiGraph()

for person in people:                 # one JSON obj per anchor_key
    key = person["anchor"]
    G.add_node(key, **person)

    for rel in person["events"]:
        if rel["type"] == "marriage":
            G.add_edge(key, rel["spouse_anchor"], label="spouse")
            G.add_edge(rel["spouse_anchor"], key, label="spouse")
    parents = person.get("parents", {})
    if p := parents.get("father_anchor"):
        G.add_edge(p, key, label="parent")
    if m := parents.get("mother_anchor"):
        G.add_edge(m, key, label="parent")
```

### How you use it at query time

1. **Resolve the two names → anchor keys**
   Run a fuzzy/phonetic query for each name; pick the top anchor for “Aart van Santen” and “Leendert van Santen.”
2. **Graph search** (`nx.shortest_path` with edge filters like `{"label":["parent","spouse"]}`) gives a **path of anchors**.
3. **Fetch all chunks whose `anchor_key` is on that path** (plus ±1 neighbours).
4. **Send *those* passages to the LLM** with a system prompt:

   > “The following passages are in genealogical order along the path A → … → B. Explain how the two people are related.”

The model now sees **every hop**—not just the best-scoring chunk—and can answer questions like *“How is Aart related to Leendert?”* accurately.

---

## 2  Add an **“anchor expansion” retriever**

If you don’t want a graph, keep it pure SQL/Chroma:

```sql
-- step 1: retrieve anchors for the two query names
SELECT anchor_key FROM chunks WHERE dm_codes && $dm_A LIMIT 3;
SELECT anchor_key FROM chunks WHERE dm_codes && $dm_B LIMIT 3;

-- step 2: expand by simple rules
--   • same surname, one generation apart   (II.1.*  vs III.?.*)
--   • same parents anchor                  (share parent anchor in keys)
--   • marriage partner anchors             (spouse field)
```

Pull every chunk whose anchor key passes the rule into the prompt.
No heavy graph; rules live in SQL or Python.

---

## 3  Let the **LLM iteratively pull more context** (self-ask style)

Use ChatML / function-calling:

1. **First turn**: “What anchors do I need to answer the relationship between A and B?”
2. Your code receives `[ "II.1.a", "II.1", "I.1.b" ]` → retrieves those chunks.
3. **Second turn**: send the passages; the model answers.

*(This is the technique in “Self-Ask with Search” papers—retrieval happens in steps until the model says it’s done.)*

---

## 4  Raise `k` adaptively + use **MMR**

Sometimes the simple fix is:

```python
k = 4 if "explain marriage" in query else 25   # family-tree style ⇒ big k
hits = db.similarity_search_by_vector(q_vec, k=k, lambda_mult=0.5)  # MMR
```

MMR (“Maximal Marginal Relevance”) diversifies the hits so you’re more likely to pull *different* family members instead of 10 copies of the same paragraph.

---

### Putting it together in code (mini-card)

```python
def relationship_answer(nameA, nameB, G):
    # --- resolve names to anchors
    ancA = top_anchor_for(nameA)   # your hybrid query + phonetic lookup
    ancB = top_anchor_for(nameB)

    # --- try graph first
    try:
        path = nx.shortest_path(G, ancA, ancB)
        anchors = set(path)
    except nx.NetworkXNoPath:
        anchors = {ancA, ancB}

    # --- fetch passages for each anchor (+ neighbours)
    contexts = []
    for anc in anchors:
        rows = db.query(where={"anchor_key": anc}, n_results=1)["documents"][0]
        contexts.extend(expand(anc, rows[0]["chunk_no"], window=1))

    prompt = f"{SYSTEM_PROMPT}\n" + "\n\n".join(contexts) + \
             f"\n---\nHow is {nameA} related to {nameB}?"

    return ollama.generate(model="aya:35b-23", prompt=prompt)["response"]
```

---

## Take-away

* **Retrieval alone ≠ context reasoning.** Add a structural layer (graph or anchor expansion) that **assembles** the line of evidence.
* The LLM then sees *exactly* the hops it needs—AJAX-style—as opposed to hoping they co-occur in top-k similarity.
* Even a 35 B model answers reliably once the right anchors are in view; no need to balloon the prompt with the entire corpus.

That gives you confident answers to questions like *“How is Gerrit related to Aart?”* without sacrificing speed or loading the LLM with irrelevant text.
