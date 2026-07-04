# Active Learning Implementation Plan

**Goal**: Add supervised learning on top of the unsupervised graph-based clustering algorithm, using manual labels from PotentialDuplicate review to improve clustering quality over time.

**Inspiration**: dedupe.io's active learning loop - train models from user labels, identify uncertain pairs for labeling, iterate.

**Status**: Planning phase
**Date**: 2025-10-23

---

## Overview

The current clustering system (based on Kirielle et al. 2022) uses **fixed, hand-coded weights** for similarity calculation:

```python
similarity = 0.4*name_sim + 0.2*date_sim + 0.3*relationship_sim + 0.1*other_sim
```

We'll extend this to **learn optimal weights** (or use a full classifier) from labeled data, while maintaining backward compatibility.

---

## Architecture

### Current Flow
```
PersonMention records
  → cluster_entities (unsupervised, fixed weights)
  → PotentialDuplicate records (review_status=PENDING)
  → Manual review in admin UI (mark CONFIRMED/REJECTED)
```

### New Flow
```
PersonMention records
  → cluster_entities (unsupervised, fixed weights) [FIRST RUN]
  → PotentialDuplicate records (review_status=PENDING)
  → Manual review in admin UI (label 50+ pairs)
  → train_clustering_model (learns from labels) [NEW]
  → cluster_entities (uses learned model) [SUBSEQUENT RUNS]
  → More PotentialDuplicate records
  → Label more pairs (prioritize uncertain ones)
  → Retrain, re-cluster, iterate...
```

---

## Implementation Phases

### Phase 1: Feature Extraction & Training (Core)

**Goal**: Train a model on existing labels

**Files to create/modify**:
- `genealogy/management/commands/train_clustering_model.py` (NEW)
- `genealogy/clustering/learned_model.py` (NEW)

**Key components**:

1. **Feature Extraction**
   - Extract atomic similarities from `RelationalNode.atomic_sims`
   - Convert to fixed-order feature vector for ML
   - Use same features the clustering algorithm already computes

2. **Model Training**
   - Start simple: `LogisticRegression` (interpretable, works with small data)
   - Later: try `XGBoost` or `RandomForest` for non-linear combinations
   - Input: feature vectors from labeled pairs
   - Output: probability of duplicate (0-1)

3. **Model Persistence**
   - Save trained model to `data/clustering_model.pkl`
   - Store metadata (feature names, training date, accuracy, etc.)

**Acceptance criteria**:
- Can extract features from any PersonMention pair
- Can train model on CONFIRMED/REJECTED labels
- Can save/load model from disk
- Reports training metrics (accuracy, precision, recall)

---

### Phase 2: Integration with Clustering

**Goal**: Use learned model in clustering algorithm

**Files to modify**:
- `genealogy/clustering/dependency_graph.py`
- `genealogy/clustering/person_record.py`

**Changes**:

1. **Load Model at Startup**
   ```python
   class DependencyGraph:
       def __init__(self):
           self.learned_model = self._load_learned_model()
   ```

2. **Use Model for Similarity Calculation**
   ```python
   def calculate_overall_similarity(self, p1_id, p2_id):
       # ... compute atomic_sims ...

       if self.learned_model:
           features = self._atomic_sims_to_vector(atomic_sims)
           similarity = self.learned_model.predict_proba([features])[0][1]
       else:
           # Fallback to hand-coded formula
           similarity = self._compute_similarity_formula(atomic_sims)

       return RelationalNode(similarity=similarity, atomic_sims=atomic_sims, ...)
   ```

3. **Backward Compatibility**
   - If no model exists, use original hand-coded weights
   - Never breaks existing functionality

**Acceptance criteria**:
- `cluster_entities` automatically uses learned model if available
- Falls back gracefully if no model exists
- Produces similar or better clustering quality

---

### Phase 3: Active Learning - Uncertainty Sampling

**Goal**: Prioritize which pairs users should label next

**Files to modify**:
- `genealogy/management/commands/train_clustering_model.py`
- `genealogy/admin/duplicate_clusters.py` (add sorting by uncertainty)

**Key concepts**:

1. **Uncertainty Calculation**
   - Pairs near decision boundary (probability ≈ 0.5) are most uncertain
   - These are most informative to label
   - `uncertainty = abs(predicted_probability - 0.5)`

2. **Suggest Pairs to Label**
   ```bash
   $ python manage.py train_clustering_model

   Trained on 87 labeled pairs
   Accuracy: 0.84, Precision: 0.88, Recall: 0.81

   Most uncertain pairs (suggest labeling these next):
   1. PersonMention #456 vs #789 (prob: 0.48, uncertainty: 0.02)
   2. PersonMention #123 vs #234 (prob: 0.52, uncertainty: 0.02)
   ...

   Saved model to data/clustering_model.pkl
   ```

3. **Admin UI Enhancement** (optional)
   - Sort PotentialDuplicate list by uncertainty
   - Show "Review These First" section
   - Badge/highlight uncertain pairs

**Acceptance criteria**:
- Command identifies and reports most uncertain pairs
- Uncertainty scores are meaningful (validated against holdout set)
- Can export uncertain pairs for targeted review

---

### Phase 4: Multi-Level Clustering Integration

**Goal**: Handle mention-to-identity and identity-to-identity comparisons

**Context**: Earlier discussion about clustering at multiple levels:
- Unlabeled mention ↔ unlabeled mention
- Unlabeled mention ↔ existing Identity cluster
- Identity ↔ Identity (suggest merging two clusters)

**Files to modify**:
- `genealogy/management/commands/cluster_entities.py`
- `genealogy/clustering/dependency_graph.py`

**Approach**:

1. **Mention vs Identity Comparison**
   ```python
   def compare_mention_to_identity(mention_id, identity_id, graph):
       # Get all mentions in the identity
       identity_mentions = get_mentions_for_identity(identity_id)

       # Compare mention to each, average similarity
       similarities = []
       for other_mention_id in identity_mentions:
           sim = graph.calculate_overall_similarity(mention_id, other_mention_id)
           similarities.append(sim)

       return np.mean(similarities)
   ```

2. **Identity vs Identity Comparison**
   ```python
   def compare_identity_to_identity(identity1_id, identity2_id, graph):
       mentions1 = get_mentions_for_identity(identity1_id)
       mentions2 = get_mentions_for_identity(identity2_id)

       # All pairs between the two identity clusters
       similarities = []
       for m1 in mentions1:
           for m2 in mentions2:
               sim = graph.calculate_overall_similarity(m1, m2)
               similarities.append(sim)

       return np.mean(similarities)
   ```

3. **Training Data Expansion**
   - Identity-level labels create multiple training examples
   - Each pairwise comparison becomes a training instance
   - More data from fewer manual labels

**Acceptance criteria**:
- Can compare entities at all three levels
- Learned model works for all comparison types
- Identity-level labels expand to pairwise training data

---

### Phase 5: Evaluation & Monitoring

**Goal**: Track model performance over time

**Files to create**:
- `genealogy/management/commands/evaluate_clustering_model.py` (NEW)

**Metrics to track**:

1. **Model Performance**
   - Cross-validation accuracy on labeled pairs
   - Precision/recall on holdout set
   - Feature importances (which features matter most?)

2. **Clustering Quality**
   - How many new clusters found after retraining?
   - How many existing Identity merges suggested?
   - Comparison to baseline (hand-coded weights)

3. **Active Learning Efficiency**
   - How quickly does accuracy improve with more labels?
   - Are uncertain pairs actually informative?

**Outputs**:
- Generate reports comparing model versions
- Track accuracy over time
- Identify when more training data is needed

**Acceptance criteria**:
- Can evaluate model on holdout data
- Can compare clustering results across model versions
- Can visualize learning curves (accuracy vs # labels)

---

## Data Requirements

### Minimum Viable Training Set
- **50-100 labeled pairs** for initial logistic regression
- Mix of CONFIRMED and REJECTED (ideally 50/50 balance)
- Diverse examples (different similarity scores, different relationship patterns)

### Optimal Training Set
- **200-500 labeled pairs** for good generalization
- Includes edge cases (siblings, common names, date uncertainties)
- Covers all relationship types (parents, spouses, children)

### Active Learning Target
- Label **20-50 uncertain pairs** per iteration
- Retrain after each batch
- Diminishing returns after ~500 total labels

---

## Technical Decisions

### Model Choice

**Option 1: Logistic Regression** (RECOMMENDED TO START)
- ✅ Interpretable (can see feature weights)
- ✅ Works with small training sets (50+ examples)
- ✅ Fast training and prediction
- ✅ Outputs calibrated probabilities
- ❌ Linear only (can't learn feature interactions)

**Option 2: Random Forest / XGBoost**
- ✅ Non-linear (learns feature interactions)
- ✅ Handles missing features well
- ✅ Feature importance scores
- ❌ Needs more training data (200+ examples)
- ❌ Less interpretable
- ⚠️ May overfit with small data

**Decision**: Start with LogisticRegression, switch to XGBoost when we have 200+ labels

### Feature Engineering

Use atomic similarities directly from clustering algorithm:

```python
feature_vector = [
    atomic_sims.get('given_names', 0.0),      # Name similarity
    atomic_sims.get('surname', 0.0),
    atomic_sims.get('birth_year', 0.0),       # Date similarities
    atomic_sims.get('death_year', 0.0),
    atomic_sims.get('parent_overlap', 0.0),   # Relationship overlaps
    atomic_sims.get('child_overlap', 0.0),
    atomic_sims.get('spouse_overlap', 0.0),
    atomic_sims.get('generation', 0.0),       # Generation match
    atomic_sims.get('birth_place', 0.0),      # Place similarities
    atomic_sims.get('death_place', 0.0),
]
```

**Rationale**:
- These are already computed by the clustering algorithm
- Domain experts understand these features
- Can compare learned weights to hand-coded weights

### Model Storage

**Location**: `data/clustering_model.pkl`

**Format**:
```python
{
    'model': trained_model_object,
    'feature_names': ['given_names', 'surname', ...],
    'training_date': '2025-10-23',
    'training_size': 87,
    'accuracy': 0.84,
    'precision': 0.88,
    'recall': 0.81,
}
```

**Versioning**: Keep model history
- `data/clustering_model_v1.pkl`
- `data/clustering_model_v2.pkl`
- `data/clustering_model_latest.pkl` (symlink)

---

## Modified `--clean` Flag Behavior

**Current behavior**:
```python
if options['clean']:
    PotentialDuplicate.objects.all().delete()  # Nukes everything!
```

**New behavior**:
```python
if options['clean']:
    # Only delete PENDING records, keep reviewed ones
    deleted_count = PotentialDuplicate.objects.filter(
        review_status='PENDING'
    ).delete()[0]

    kept_count = PotentialDuplicate.objects.exclude(
        review_status='PENDING'
    ).count()

    self.stdout.write(
        f"Deleted {deleted_count} pending duplicates, "
        f"kept {kept_count} reviewed pairs"
    )
```

**Rationale**:
- Preserve manual labeling work
- Allow iterative clustering without losing labels
- Training data accumulates over time

---

## Workflow Examples

### First-Time User (No Labels Yet)

```bash
# 1. Run initial clustering (hand-coded weights)
docker compose exec web python manage.py cluster_entities

# Output: "Found 150 potential duplicate clusters"

# 2. Review in admin UI, label 50 pairs as CONFIRMED/REJECTED

# 3. Not enough data yet - keep labeling
#    (train_clustering_model would warn: "Need at least 50 labels")

# 4. After 50+ labels, train first model
docker compose exec web python manage.py train_clustering_model

# Output:
# "Trained logistic regression on 52 pairs
#  Accuracy: 0.79 (cross-validation)
#  Model saved to data/clustering_model.pkl
#
#  Suggest labeling these uncertain pairs next:
#  1. PersonMention #456 vs #789 (prob: 0.48)
#  ..."

# 5. Re-run clustering with learned model
docker compose exec web python manage.py cluster_entities --clean

# Output:
# "Using learned model from data/clustering_model.pkl
#  Found 12 new potential duplicate clusters
#  Found 3 suggested identity merges"
```

### Experienced User (Iterating)

```bash
# Label 30 more uncertain pairs from previous run

# Retrain (now 82 total labels)
docker compose exec web python manage.py train_clustering_model

# Output:
# "Trained on 82 pairs (up from 52)
#  Accuracy improved: 0.79 → 0.84
#  Top feature: parent_overlap (weight: 0.42)
#  ..."

# Re-cluster
docker compose exec web python manage.py cluster_entities --clean

# Repeat...
```

---

## Success Metrics

### Short Term (Phase 1-2)
- ✅ Can train model on 50+ labeled pairs
- ✅ Learned model achieves ≥80% accuracy (cross-validation)
- ✅ cluster_entities uses learned model seamlessly
- ✅ Backward compatible (works without model)

### Medium Term (Phase 3-4)
- ✅ Active learning identifies informative pairs
- ✅ Labeling uncertain pairs improves accuracy faster than random
- ✅ Multi-level clustering (mention/identity) integrated
- ✅ Model quality improves with 200+ labels

### Long Term (Phase 5)
- ✅ Clustering quality measurably better than hand-coded weights
- ✅ Fewer false positives in PotentialDuplicate
- ✅ Model generalizes to new documents
- ✅ Can track model performance over time

---

## Risks & Mitigation

### Risk: Overfitting with Small Data
**Mitigation**:
- Start with logistic regression (high bias, low variance)
- Use cross-validation to detect overfitting
- Regularization (L2 penalty)

### Risk: Label Bias
**Problem**: User only labels easy/obvious pairs, model doesn't learn edge cases
**Mitigation**:
- Active learning (show uncertain pairs)
- Random sampling (include some random pairs)
- Monitor distribution of labels

### Risk: Feature Distribution Shift
**Problem**: New documents have different name/date distributions
**Mitigation**:
- Track feature distributions over time
- Retrain periodically on recent labels
- Alert if new data looks very different

### Risk: Model Breaks Clustering
**Problem**: Learned model produces worse results than hand-coded weights
**Mitigation**:
- Easy rollback (delete model file, restart cluster_entities)
- A/B comparison (run both, compare results)
- Gradual rollout (use model only above confidence threshold)

---

## Future Enhancements

### Beyond This Plan

1. **Ensemble Models**
   - Combine hand-coded + learned model
   - `final_similarity = 0.3*hand_coded + 0.7*learned`

2. **Deep Learning**
   - Learn embeddings for names/relationships
   - End-to-end similarity learning
   - Requires much more data (1000+ labels)

3. **Transfer Learning**
   - Pre-train on other genealogical datasets
   - Fine-tune on your specific data
   - Leverage external knowledge

4. **Constraint Learning**
   - Learn which constraints matter most
   - Adaptive temporal constraints
   - Data-driven sibling detection

5. **Interactive Labeling UI**
   - Real-time model updates
   - Show why model predicted duplicate/not
   - Explain feature contributions

---

## References

- **dedupe.io**: https://docs.dedupe.io/en/latest/how-it-works/How-it-works.html
  - Active learning for entity resolution
  - Uncertainty sampling
  - Blocking + supervised classification

- **Kirielle et al. (2022)**: "Unsupervised Graph-based Entity Resolution for Accurate and Efficient Family Pedigree Search"
  - Current unsupervised approach
  - Mentions active learning as future work (Section 11)

- **Settles (2009)**: "Active Learning Literature Survey"
  - Uncertainty sampling strategies
  - Query-by-committee
  - Expected model change

---

## Implementation Timeline

**Week 1**: Phase 1 - Feature extraction and model training
- Create `train_clustering_model` command
- Extract features from labeled pairs
- Train and save logistic regression model

**Week 2**: Phase 2 - Integration with clustering
- Modify `DependencyGraph` to load/use model
- Test backward compatibility
- Compare results to baseline

**Week 3**: Phase 3 - Active learning
- Implement uncertainty sampling
- Report uncertain pairs
- Label 50-100 uncertain pairs, measure improvement

**Week 4**: Phase 4 - Multi-level clustering
- Implement mention-to-identity comparison
- Implement identity-to-identity comparison
- Test end-to-end workflow

**Week 5**: Phase 5 - Evaluation
- Build evaluation framework
- Compare learned vs hand-coded weights
- Document results

---

## Next Steps

1. ✅ Review this plan
2. ⬜ Implement Phase 1 (feature extraction + training)
3. ⬜ Test with existing labels
4. ⬜ Evaluate and iterate
