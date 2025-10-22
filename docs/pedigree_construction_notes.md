Kirielle et al. (2022), *“Unsupervised Graph-based Entity Resolution for Accurate and Efficient Pedigree Construction,”* is one of the few papers that tries to unify **record linkage** and **family graph reconstruction** in one coherent framework — essentially solving the same kind of problem you’re facing.

---

## 🧩 1. Problem they’re solving

They start from **vital-record style data** — births, deaths, marriages — often coming from multiple collections (civil registers, church records, etc.), each containing partial and noisy information.

Each record refers to one or more *real people* (child, father, mother, bride, groom, etc.), but those identities are fragmented across many entries.
Their goal:

> link all records that refer to the same person and then assemble those people into coherent **pedigrees** (family trees).

So it’s a **two-layer problem**:

1. **Entity Resolution (ER):** deduplicate individual mentions.
2. **Pedigree Construction:** use resolved identities to build the family graph.

---

## ⚙️ 2. Core idea — treat ER as a *graph problem*

They represent every record as a node and every potential “same-person” relationship as an *edge with a similarity weight*.

Then, instead of setting a global threshold (as in classical ER), they use **graph connectivity and local structure** to infer merges.

### The intuition

* If two records share strong similarities *and* appear in compatible family relationships, they belong in the same cluster (one person).
* If linking them would violate family constraints (like two different mothers for the same child, or overlapping lifespans), the edge is penalized or dropped.

---

## 🧠 3. Unsupervised pipeline (step-by-step)

### Step 1. Pre-processing & feature extraction

* Normalize attributes (names, dates, places).
* Compute **pairwise similarities** on:

  * Name (string or phonetic distance)
  * Temporal proximity (birth/marriage/death years)
  * Geographic proximity (same locality)
  * Relationship roles (father, mother, bride, etc.)
* Each comparison yields a **similarity score** ( s_{ij} \in [0,1] ).

### Step 2. Graph construction

Create a **weighted undirected graph**:

* Node = record (person mention)
* Edge weight = similarity between records

Apply **blocking** (surname, time window, location) to limit comparisons and keep the graph sparse.

### Step 3. Edge pruning

They prune low-weight edges (below adaptive threshold) to remove noise, keeping only plausible candidate links.

### Step 4. Local graph clustering

They detect connected components or use community-detection algorithms (e.g., Louvain or label propagation) to group records into clusters of potential same-person mentions.

Each cluster ideally corresponds to one real individual.

### Step 5. Constraint enforcement (“pedigree sanity checks”)

Within each cluster, and between clusters when building families, they enforce genealogical constraints:

| Constraint              | Example                                                                          |
| ----------------------- | -------------------------------------------------------------------------------- |
| **Monogamy (temporal)** | Same person can’t have overlapping marriages in implausible timeframes           |
| **Age feasibility**     | Parents must be older than children by at least a biologically possible interval |
| **No cyclic parentage** | Prevents A → B → A loops                                                         |
| **Gender consistency**  | A person can’t be both “father” and “mother” in different records                |

Violating edges are dropped or clusters split accordingly.

### Step 6. Pedigree assembly

Once deduplication stabilizes:

* Each cluster → a unique person node.
* Relationships from records (father-of, spouse-of, etc.) become edges between person nodes.
* The resulting directed multigraph represents the **pedigree**.

### Step 7. Evaluation / refinement

They compare generated pedigrees against gold-standard family trees, measuring *linkage precision/recall* and *pedigree completeness*.

---

## 🔬 4. Key techniques that make it “unsupervised”

* **No labeled training pairs.** All thresholds are derived from the data distribution (e.g., percentile-based edge-weight cutoffs).
* **Relational constraints as weak supervision.** The biological / genealogical rules act as a self-supervising signal.
* **Graph propagation.** Once a high-confidence edge connects two records, that similarity can *propagate* to neighbors, improving recall.

---

## 🧮 5. Why it works better than naive pairwise ER

Traditional ER models treat each comparison independently, ignoring relational context.
Kirielle et al. exploit the **structure of family relationships** as extra information:

* If record A (as father) matches record B (as father), and both share the same child record C, this triangulation increases confidence.
* Conversely, if merging A and B would imply two different mothers for one child, that edge is dropped.

This relational feedback loop is why they can reach high precision without supervision.

---

## 📈 6. Results and performance

On simulated and real genealogical datasets, they show:

* Higher **linkage accuracy** than traditional attribute-only ER (e.g., Fellegi–Sunter or simple logistic models).
* Dramatically better **pedigree completeness** (fewer orphan nodes).
* Linear-scaling runtime due to blocking and sparse graph operations.

---

## 🧩 7. How you could adapt this

For your OCR’d Dutch family book:

1. **Records → nodes:** each JSON person entry.
2. **Edge weights:** combination of name similarity, date overlap, location, spouse/child overlap.
3. **Graph clustering:** use a community detection algorithm (e.g., NetworkX `asyn_lpa_communities` or Louvain).
4. **Constraints:** biological plausibility + anchor hierarchy (e.g., II.1.a → II.2.c).
5. **Collapse clusters → unique people.**
6. **Edges between clusters → pedigree edges.**

You can treat each generation anchor section as a “mini-block” for efficiency.

---

## 🧭 8. Relationship to your earlier ideas

Your notion of “if you squint, you can see the family cluster” is precisely what this algorithm formalizes:

* It identifies those dense subgraphs (shared children/spouses) and collapses them systematically.
* The constraints prevent bad merges that would break family logic.

---

Let’s outline how you can translate Kirielle et al.’s graph-based pedigree construction into a **Python + Postgres/pgvector workflow**, step by step.

---

## 🧩 1. Schema and data model

You’ll store *records*, *vectors* (for quick blocking & similarity), and *relationships* in Postgres.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE person_record (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT,
    canonical_name TEXT,
    surname TEXT,
    given_names TEXT,
    birth_year INT,
    place_norm TEXT,
    spouse_names TEXT[],
    child_names TEXT[],
    parent_names TEXT[],
    embedding vector(768),         -- name/date/place embedding
    source_json JSONB
);

CREATE TABLE potential_link (
    id SERIAL PRIMARY KEY,
    record_a INT REFERENCES person_record(id),
    record_b INT REFERENCES person_record(id),
    name_sim FLOAT,
    year_sim FLOAT,
    place_sim FLOAT,
    spouse_overlap FLOAT,
    child_overlap FLOAT,
    overall_score FLOAT,
    is_candidate BOOLEAN DEFAULT TRUE
);
```

`pgvector` stores pre-computed embeddings so you can find similar names quickly before doing fine-grained scoring.

---

## ⚙️ 2. Generate embeddings & insert into Postgres

In Python:

```python
from sentence_transformers import SentenceTransformer
import psycopg2, numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
conn = psycopg2.connect("dbname=family user=you")

cur = conn.cursor()
cur.execute("SELECT id, canonical_name, birth_year, place_norm FROM person_record")
for rec_id, name, year, place in cur.fetchall():
    text = f"{name} {year or ''} {place or ''}"
    emb = model.encode(text, normalize_embeddings=True)
    cur.execute("UPDATE person_record SET embedding=%s WHERE id=%s",
                (list(emb), rec_id))
conn.commit()
```

---

## 🧠 3. Blocking + candidate generation with pgvector

```sql
-- Fetch 50 most similar by vector cosine distance (approx blocking)
SELECT id, 1 - (embedding <#> :query_emb) AS sim
FROM person_record
WHERE surname = :surname  -- coarse block
ORDER BY embedding <#> :query_emb
LIMIT 50;
```

That gives candidate pairs that are name-similar.
You’ll then compute refined scores using textual + structural overlap.

---

## 🔬 4. Compute pairwise similarity features (Python)

```python
import itertools, textdistance, psycopg2

def pair_features(a, b):
    name_sim = textdistance.trigram.normalized_similarity(a['canonical_name'], b['canonical_name'])
    year_sim = max(0, 1 - abs((a['birth_year'] or 0) - (b['birth_year'] or 0))/10)
    place_sim = float(a['place_norm'] == b['place_norm'])
    spouse_ov = len(set(a['spouse_names']) & set(b['spouse_names'])) / max(1,len(set(a['spouse_names'])|set(b['spouse_names'])))
    child_ov  = len(set(a['child_names']) & set(b['child_names'])) / max(1,len(set(a['child_names'])|set(b['child_names'])))
    score = 0.4*name_sim + 0.2*year_sim + 0.2*spouse_ov + 0.2*child_ov
    return (name_sim, year_sim, place_sim, spouse_ov, child_ov, score)

# Compute and insert potential_link records
```

This mirrors Kirielle’s *edge-weight* computation.

---

## 🕸️ 5. Graph clustering in Python

```python
import networkx as nx
import pandas as pd

links = pd.read_sql("SELECT record_a, record_b, overall_score FROM potential_link WHERE overall_score > 0.7", conn)
G = nx.Graph()
G.add_weighted_edges_from(links.values)
clusters = list(nx.connected_components(G))

# Collapse each cluster -> single person entity
for cid, nodes in enumerate(clusters, 1):
    merged_id = min(nodes)
    conn.cursor().execute(
        "UPDATE person_record SET cluster_id=%s WHERE id = ANY(%s)",
        (cid, list(nodes))
    )
```

Each connected component = one merged identity.

---

## 🧮 6. Constraint enforcement

After clustering, enforce genealogy constraints in Python:

```python
def valid_age_diff(parent, child):
    if not parent['birth_year'] or not child['birth_year']:
        return True
    return 12 <= (child['birth_year'] - parent['birth_year']) <= 55
```

Drop or split clusters that violate age, gender, or monogamy rules.

---

## 🌳 7. Pedigree graph assembly

Once deduplicated:

```python
CREATE TABLE pedigree_edge (
    src_id INT,
    dst_id INT,
    relation TEXT CHECK (relation IN ('PARENT_OF','SPOUSE_OF'))
);
```

Then generate:

```python
INSERT INTO pedigree_edge (src_id, dst_id, relation)
SELECT parent.cluster_id, child.cluster_id, 'PARENT_OF'
FROM relationships_raw
JOIN person_record parent ON parent.id = relationships_raw.parent_id
JOIN person_record child ON child.id = relationships_raw.child_id;
```

Now you can visualize or query the family graph with SQL or Python (`networkx`, `igraph`).

---

## 📊 8. Iterative refinement

* Store every merge confidence in `potential_link` for audit.
* Re-compute embeddings periodically after merges (average of cluster embeddings).
* Flag clusters with conflicting evidence for manual review.

---

## 🧠 9. How this implements Kirielle et al.

| Kirielle step                             | Your implementation                            |
| ----------------------------------------- | ---------------------------------------------- |
| Record normalization & feature extraction | Python pre-processing + embeddings             |
| Weighted graph                            | `potential_link` table                         |
| Graph clustering                          | NetworkX connected components                  |
| Pedigree construction                     | `pedigree_edge` table                          |
| Constraint checks                         | Python post-processing                         |
| Unsupervised operation                    | No labels; thresholds from score distributions |

---
