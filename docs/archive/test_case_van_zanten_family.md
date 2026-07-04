# Test Case: van Zanten Family Clustering

## Overview

This document tracks the van Zanten family as a key test case for person clustering and family-level clustering algorithms.

## Family Background

- **Patriarch**: Aart van Zanten (Generation 5)
- **Matriarch**: Anna Antonia Kappers (Generation 5)
- **Children**: ~18 children (many died in infancy)
- **Relationship**: User's 3x great grandparents

## Current State (2025-10-12)

### Person Duplication

The family members appear multiple times in the database, with each appearance typically corresponding to a different child relationship mention:

- **Aart van Zanten**: 11 distinct Person records where he's married to Anna Antonia Kappers
  - 10 are Generation 5
  - 1 is labeled "Aart (Arie) van Zanten"
  - Each record typically has different children associated with it

- **Anna Antonia Kappers**: ~13 distinct Person records (estimated)

### Clustering Behavior

As of the latest clustering run:

**Current Cluster Status**: ✅ **FOUND** at Cluster #7

- **Total cluster size**: 16 persons
- **Average confidence**: 78.8%
- **Position in list**: 8th largest cluster (out of 228 total)

#### Cluster Composition

The cluster contains the correct Aart van Zanten records **plus some false positives**:

| Person Records | Count | Notes |
|---------------|-------|-------|
| Aart van Zanten married to Anna Antonia Kappers (Gen 5) | 10 | ✅ Correct - these are the target duplicates |
| Aart (Arie) van Zanten married to Anna Antonia Kappers (Gen 5) | 1 | ⚠️ Variant name - probably same person |
| Aart van Zanten (Gen 6) | 1 | ❌ False positive - this is a son |
| Aart van Zanten (Gen 7) | 1 | ❌ False positive - this is a grandson |
| Aart van **Santen** married to Johanna Cornelia Copier (Gen 5) | 3 | ❌ False positive - different family, similar surname |

**Total**: 16 persons

### Problems Identified

1. **Generation confusion**: Sons and grandsons (Gen 6, 7) are clustering with the patriarch (Gen 5)
2. **Surname variants**: "van Santen" vs "van Zanten" being treated as similar enough to cluster
3. **Spouse mismatch**: Aarts married to different people are clustering together
4. **Missing spouse clustering**: Anna Antonia Kappers duplicates are not in this cluster (should form a family-level cluster)

## What Good Clustering Should Look Like

### Person-Level Clustering (Current Goal)

**Ideal Aart van Zanten Cluster**:
- All 10-11 Aart van Zanten records married to Anna Antonia Kappers (Gen 5)
- Exclude: different generations, different surnames, different spouses
- Size: ~10-11 persons
- Confidence: High (>85%) due to:
  - Same name (Aart van Zanten)
  - Same generation (5)
  - Same spouse name (Anna Antonia Kappers)
  - Overlapping children across records

**Ideal Anna Antonia Kappers Cluster**:
- All ~13 Anna Antonia Kappers records married to Aart van Zanten (Gen 5)
- Size: ~13 persons
- Confidence: High due to same factors

### Family-Level Clustering (Future Goal)

**Ideal van Zanten Family Cluster**:
- One unified Aart van Zanten entity (merged from 10-11 records)
- One unified Anna Antonia Kappers entity (merged from ~13 records)
- Their ~18 children as separate entities
- Relationships properly linked
- Family unit recognized as a coherent cluster

## Testing Criteria

### Person Clustering Tests

- [ ] **Test 1**: Aart van Zanten records cluster together
- [ ] **Test 2**: Generation boundaries are respected (Gen 5 doesn't cluster with Gen 6/7)
- [ ] **Test 3**: Surname variants are handled appropriately (van Zanten vs van Santen)
- [ ] **Test 4**: Spouse names are used as strong signals
- [ ] **Test 5**: False positives are minimized

### Family Clustering Tests (Future)

- [ ] **Test 6**: Aart and Anna are recognized as a family unit
- [ ] **Test 7**: Their children are associated with the family unit
- [ ] **Test 8**: Multiple family instances (each child mention) can be aligned and merged

## Current Algorithm Strengths

✅ **Relationship overlap scoring** - Records sharing same spouse name get high similarity
✅ **Graph connectivity** - All 10 Aarts form a connected component
✅ **Found the cluster** - Cluster appears in the UI (position #7)

## Current Algorithm Weaknesses

❌ **Generation constraints** - Not enforcing strict generation boundaries
❌ **Surname normalization** - "van Santen" too similar to "van Zanten"
❌ **Spouse identity matching** - Not using spouse name as a hard constraint
❌ **No family-level grouping** - Aart and Anna clusters are separate

## Next Steps

1. **Add generation constraints** to clustering algorithm
   - Prevent clustering across generation boundaries
   - Exception: Handle edge cases where generation is missing/uncertain

2. **Improve surname matching**
   - Consider phonetic distance (Daitch-Mokotoff codes)
   - Set threshold for acceptable surname variation

3. **Strengthen spouse matching**
   - Use spouse name as high-weight signal (already done - 15%)
   - Consider spouse as quasi-constraint for same-generation records

4. **Implement family-level clustering**
   - Detect family units (parent pair + children)
   - Group person clusters into family clusters
   - Build family alignment UI

## References

- Research notes: `research/pedigree_construction_notes.md`
- Kirielle et al. paper on pedigree construction
- Current clustering command: `genealogy/management/commands/cluster_entities.py`
