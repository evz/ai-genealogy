# Clustering Issues Analysis - 2025-10-12

## Problem Summary

After implementing the fix for transitive closure (only recording computed pairs), we still have a **super-cluster** (Cluster 0 with 126 persons). However, the quality improved:
- **Before fix**: 31.1% average confidence, 57% of pairs <50%
- **After fix**: 62.7% average confidence, only 13.5% of pairs <50%

The transitive closure fix **worked**, but revealed underlying similarity calculation issues.

## Root Causes

### 1. Parent Overlap Treated as Positive Signal ❌

**Current behavior** (lines 375-379 in `cluster_entities.py`):
```python
parent_overlap = self.calculate_relationship_overlap(p1.parent_names, p2.parent_names)
if parent_overlap > 0:
    weighted_sim += weights['parent_overlap'] * parent_overlap  # +7.5% weight
    total_weight += weights['parent_overlap']
```

**Problem**: When two people share the same parents, they are **siblings**, not duplicates! Parent overlap should be a **negative signal** for entity resolution.

**Evidence**:
- Gerard van Zanten <-> Francien van Zanten (70% confidence)
- Match reason: `exact_parent_overlap`
- These are siblings, not duplicates

### 2. Merge Threshold Too Low

**Current setting**: `--merge-threshold 0.60` (60%)

**Problem**: Allows weak matches to cluster:
- Surname match (20% weight × 1.0 = 20%)
- Parent overlap (7.5% weight × 1.0 = 7.5%)
- Weak given name similarity (20% weight × 0.3 = 6%)
- **Total: 33.5% before normalization, ~60-70% after normalization**

**Evidence**:
- 47% of pairs in Cluster 0 have <60% confidence
- Many pairs match on surname + parent overlap alone

### 3. Weak Given Names

**Problem**: Placeholder names like "Daughter Van Zanten" match too easily

**Examples**:
- "Daughter Van Zanten" (Gen 9) <-> "Aartje Hendrika W. van Zanten" (Gen 8)
- Confidence: 70%
- Reason: `exact_surname` only

**Issue**: String similarity function gives partial credit for character overlap, so "Daughter" vs any real name gets some similarity.

### 4. Generation Constraint Too Loose

**Current setting**: Allows ±1 generation difference

**Problem**: Parent-child pairs can cluster together:
- "Gerard Van Zanten" (Gen 7) <-> "Gertrude van Zanten" (Gen 8)
- Same surname + weak given name similarity = 70%

**This is technically correct** for the use case (child mentioned with parent in family book), but creates noise.

## Cluster 0 Breakdown

### Composition
- **Total**: 126 persons
- **Surnames**: 101 "van Zanten", 8 "Van Zanten", 6 "Zanten", 11 parsing errors
- **Generations**: 6-12 (spans 7 generations!)

### Quality Metrics
- **Total pairs**: 3,900
- **Average confidence**: 62.7%
- **Median confidence**: 61.0%
- **Cross-generation pairs (>1 gen)**: 72 (1.8%)
- **Pairs <60%**: 1,834 (47.0%)

### Example Problematic Matches

| Person 1 | Gen | Person 2 | Gen | Conf | Reason |
|----------|-----|----------|-----|------|--------|
| Daughter Van Zanten | 9 | Aartje Hendrika W. van Zanten | 8 | 70% | exact_surname |
| Gerard van Zanten | 8 | Francien van Zanten | 8 | 70% | exact_surname, exact_parent_overlap |
| Petronella van Zanten | 7 | Piet van Zanten | 7 | 70% | exact_surname, exact_parent_overlap |
| Gerard Van Zanten | 7 | Gertrude van Zanten | 8 | 70% | exact_surname |

All of these are **false positives** - they're different people (siblings or different generations) with the same surname.

## Recommended Fixes

### Priority 1: Fix Parent Overlap Logic ⚠️

**Change parent overlap from positive to negative signal:**

```python
# Parent overlap - SIBLING DETECTION
parent_overlap = self.calculate_relationship_overlap(p1.parent_names, p2.parent_names)
if parent_overlap > 0.5:  # Significant parent overlap = siblings
    # Apply penalty for sibling matches
    # Only allow if names are VERY similar (actual duplicates)
    name_sim = self.calculate_atomic_similarity('given_names', p1.given_names, p2.given_names)
    if name_sim < 0.9:  # Names don't match strongly
        rel_node.constraints_valid = False
        rel_node.constraint_violations.append('Likely siblings (shared parents, different names)')
        rel_node.similarity = 0.0
        return rel_node
```

**Impact**: Will prevent most sibling clusters

### Priority 2: Increase Merge Threshold

**Change from 0.60 to 0.70:**

```bash
python manage.py cluster_entities --merge-threshold 0.70
```

**Impact**: Will split surname-only clusters

### Priority 3: Require Minimum Given Name Similarity

**Add constraint in calculate_overall_similarity:**

```python
# Require minimum given name similarity for any match
given_name_sim = self.calculate_atomic_similarity('given_names', p1.given_names, p2.given_names)
if given_name_sim < 0.3:  # Very weak given name match
    rel_node.similarity = 0.0
    rel_node.constraints_valid = False
    rel_node.constraint_violations.append('Given names too dissimilar')
    return rel_node
```

**Impact**: Will prevent "Daughter" matching "Gertrude"

### Priority 4: Tighten Generation Constraint

**Change from ±1 to exact match only (for bootstrap phase):**

```python
# In _bootstrap_clusters:
if p1.generation is not None and p2.generation is not None:
    if p1.generation != p2.generation:  # Exact match only
        continue
```

**Keep ±1 for iterative merge** to handle edge cases where generation is slightly off.

**Impact**: Will split cross-generation surname matches

## Testing Plan

1. **Re-run clustering with parent overlap fix**
   ```bash
   docker compose exec web python manage.py cluster_entities --clean --merge-threshold 0.70
   ```

2. **Check Cluster 0 size and composition**
   - Should be much smaller (<20 persons)
   - Should have higher average confidence (>75%)

3. **Verify van Zanten test case** (docs/test_case_van_zanten_family.md)
   - Aart van Zanten cluster should still exist
   - Should have 10-11 persons (not 16 with false positives)
   - Should NOT include siblings

4. **Spot-check other large clusters**
   - Clusters 1-5 should also improve

## Implementation Priority

**Must fix**:
1. Parent overlap logic (Priority 1)
2. Increase merge threshold to 0.70 (Priority 2)

**Should fix**:
3. Minimum given name similarity (Priority 3)

**Nice to have**:
4. Tighten generation constraint (Priority 4)

## Related Files

- `genealogy/management/commands/cluster_entities.py` - Main clustering algorithm
- `docs/test_case_van_zanten_family.md` - Test case tracking
- `docs/family_clustering.md` - Family-level clustering research notes
