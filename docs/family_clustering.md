---

## 🧩 The Problem Restated

You have a bipartite-like graph of `Person` and `Relationship` nodes that came from per-chunk extractions:

```
(ParentA) -[:PARENT_OF]-> (ChildX)
(ParentA') -[:PARENT_OF]-> (ChildX)
(ParentA'') -[:PARENT_OF]-> (ChildY)
(ParentA)-[:SPOUSE_OF]->(ParentB)
(ParentA'')-[:SPOUSE_OF]->(ParentB')
...
```

The “same” person appears multiple times: once per relationship mention.
Your goal is to **collapse** those near-duplicate nodes into single identities, using *graph topology* in addition to textual similarity.

---

## 🧠 Conceptual Approach

Think in two layers:

### **1. Attribute Similarity (textual features)**

You already have canonicalized name/date/place similarities → good start.

### **2. Structural Similarity (graph features)**

Now add relational cues:

| Feature                   | Intuition                                                                           |
| ------------------------- | ----------------------------------------------------------------------------------- |
| **Child set overlap**     | Two candidates sharing many identical or overlapping children → likely same parent. |
| **Spouse overlap**        | Same spouse(s) → very strong duplicate signal.                                      |
| **Parent overlap**        | Same parents → same generation identity.                                            |
| **Neighborhood shape**    | Same degree pattern (e.g., both are parents of 11 children born in same decades).   |
| **Co-occurrence context** | Appear in same chunk/anchor region → higher likelihood.                             |

If you compute a **similarity score** combining text + structural cues, you can cluster nodes with high mutual similarity into one entity.

---

## ⚙️ Step-by-Step: Collapsing via Graph Signatures

### **Step 1. Build a basic graph**

In NetworkX, Neo4j, or whatever backend you use:

```python
G.add_edge("Jan_1681_A", "Pieter_1703", relation="parent")
G.add_edge("Jan_1681_B", "Pieter_1703", relation="parent")
G.add_edge("Jan_1681_A", "Maria_1679", relation="spouse")
...
```

### **Step 2. Derive neighborhood signatures**

For each person node `p`, define:

```python
signature(p) = {
  "children": sorted(list(child surnames or IDs)),
  "spouses": sorted(list(spouse surnames or IDs)),
  "parents": sorted(list(parent surnames or IDs)),
  "birth_decade": 1680,
  "places": ["Delft"]
}
```

Then compute *overlap ratios* between signatures:

```python
child_overlap = |C1 ∩ C2| / |C1 ∪ C2|
spouse_overlap = |S1 ∩ S2| / |S1 ∪ S2|
parent_overlap = |P1 ∩ P2| / |P1 ∪ P2|
```

Combine with textual similarity:

```
score = 0.4*name_similarity + 0.3*spouse_overlap + 0.2*child_overlap + 0.1*place_similarity
```

If `score > 0.75`, consider them the same person.

### **Step 3. Graph clustering**

Treat high-scoring edges as “same-person” edges and compute connected components:

```python
H = nx.Graph()
H.add_nodes_from(person_ids)
H.add_weighted_edges_from(similarity_edges)
clusters = list(nx.connected_components(H.subgraph([e for e in H.edges if weight>0.75])))
```

Each connected component = one merged identity.

### **Step 4. Collapse**

For each cluster:

* Choose a representative ID (earliest anchor or lowest hash).
* Merge all attributes and life events (union of events, citations, spouses, etc.).
* Re-wire all relationships to the representative.

That’s your deduplicated graph.

---

## 🔬 Structural Heuristics That Work Surprisingly Well

| Heuristic                   | Why it works                                                          | Implementation Tip                          |                     |                                 |
| --------------------------- | --------------------------------------------------------------------- | ------------------------------------------- | ------------------- | ------------------------------- |
| **Spouse consistency rule** | Two nodes married to the same person are very likely the same.        | If `                                        | spousesA ∩ spousesB | ≥ 1`and`name_sim > 0.6`, merge. |
| **Child set rule**          | Parent-of edges define identity strongly.                             | Merge if ≥ 50 % child overlap.              |                     |                                 |
| **Temporal sanity**         | Births clustered within plausible fertility span (~15–45 yrs).        | Filter merges violating this.               |                     |                                 |
| **Triangulation rule**      | If A≈B and B≈C, merge A,B,C together (transitive closure).            | Use connected components.                   |                     |                                 |
| **Family cluster rule**     | If two groups share ≥ 2 individuals in relationships, merge families. | Helps collapse duplicated nuclear families. |                     |                                 |

---

## 🧮 Example in Pseudocode

```python
def merge_candidates(person_nodes):
    merges = []
    for a, b in itertools.combinations(person_nodes, 2):
        sim = name_sim(a,b)
        spouse_ov = overlap(a.spouses, b.spouses)
        child_ov  = overlap(a.children, b.children)
        score = 0.4*sim + 0.3*spouse_ov + 0.2*child_ov
        if score > 0.8:
            merges.append((a.id,b.id))
    clusters = union_find(merges)
    return clusters
```

Then rebuild your graph with one node per cluster.

---

## 🧩 Optional: Graph-based Machine Learning

If you ever want to go fancy:

* Represent people as **node embeddings** using their textual + structural attributes (e.g., Node2Vec or GraphSAGE on your relationship graph).
* Cluster embeddings with DBSCAN or HDBSCAN to find duplicates automatically.
* Validate on a few manually labeled pairs.

This gets close to what professional record-linkage systems do but still transparent.

---

## 🧠 Sanity: keep provenance!

When merging:

* Keep a list of all source node IDs and their original chunk IDs.
* Keep merged confidence score and rule rationale.
* Don’t delete anything; create `MERGED_INTO` edges for traceability.

That way you can always undo or audit merges later.

---

### ✅ TL;DR

| Stage | What to do                                        |
| ----- | ------------------------------------------------- |
| 1     | Build person graph with parent/spouse/child edges |
| 2     | Compute name/date/place similarity                |
| 3     | Add structural overlap (child/spouse sets)        |
| 4     | Connect nodes with high combined score            |
| 5     | Collapse connected components into unified people |
| 6     | Keep provenance and confidence for review         |

---
