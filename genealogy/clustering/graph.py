"""
DependencyGraph for graph-based entity resolution.

Implements the core graph structure and similarity calculations used in
entity resolution clustering.
"""
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

import Levenshtein

from .nodes import AtomicNode, RelationalNode
from .person_record import PersonRecord


class DependencyGraph:
    """
    Graph structure representing person records, their attributes, and potential matches.
    Implements constraint propagation and similarity calculations.
    """

    def __init__(self):
        self.persons: Dict[int, PersonRecord] = {}
        self.atomic_nodes: Dict[int, Set[AtomicNode]] = defaultdict(set)
        self.relational_nodes: Dict[Tuple[int, int], RelationalNode] = {}

        # Attribute frequency for disambiguation (AMB technique)
        self.attribute_frequency: Dict[str, Dict[any, int]] = defaultdict(lambda: defaultdict(int))

    def add_person(self, person_record: PersonRecord):
        """Add a person record to the graph"""
        self.persons[person_record.id] = person_record

        # Create atomic nodes for each attribute
        for attr_name, attr_value in person_record.get_attributes().items():
            atomic_node = AtomicNode(person_record.id, attr_name, attr_value)
            self.atomic_nodes[person_record.id].add(atomic_node)

            # Track frequency for disambiguation
            self.attribute_frequency[attr_name][attr_value] += 1

    def calculate_atomic_similarity(self, attr_name: str, val1: any, val2: any) -> float:
        """
        Calculate similarity for a single attribute pair.
        Returns a value between 0 and 1.
        """
        if val1 is None or val2 is None:
            return 0.0

        # Exact match
        if val1 == val2:
            return 1.0

        # String similarity for names and places
        if isinstance(val1, str) and isinstance(val2, str):
            return self._string_similarity(val1, val2)

        # Numeric similarity for years
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return self._numeric_similarity(val1, val2)

        return 0.0

    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity using Levenshtein distance.
        Returns value between 0 and 1.
        """
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()

        if s1 == s2:
            return 1.0

        # Check if one is a substring of the other (e.g., "John" vs "John William")
        if s1 in s2 or s2 in s1:
            return 0.85

        # Use python-Levenshtein library for similarity ratio (0-1 scale)
        return Levenshtein.ratio(s1, s2)

    def _numeric_similarity(self, n1: float, n2: float, tolerance: float = 5.0) -> float:
        """
        Calculate numeric similarity for years/ages.
        Returns 1.0 if within tolerance, decays to 0 as difference increases.
        """
        diff = abs(n1 - n2)

        if diff == 0:
            return 1.0
        elif diff <= tolerance:
            # At most 30% penalty within tolerance
            return 1.0 - (diff / tolerance) * 0.3
        else:
            # Exponential decay beyond tolerance
            return max(0.0, 0.7 * (tolerance / diff))

    def calculate_disambiguation_weight(self, attr_name: str, attr_value: any) -> float:
        """
        Calculate disambiguation weight (AMB technique from Paper 2).
        Rare attribute values get higher weight than common ones.
        """
        total_persons = len(self.persons)
        freq = self.attribute_frequency[attr_name][attr_value]

        if freq == 0 or total_persons == 0:
            return 1.0

        # Inverse frequency weighting
        # More common values → lower weight
        # Rarer values → higher weight
        return 1.0 / (1.0 + freq / total_persons)

    def calculate_relationship_overlap(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Calculate Jaccard similarity (overlap) between two sets of names.

        Returns:
            float: Jaccard coefficient (0.0 to 1.0)
        """
        if not set1 and not set2:
            return 0.0  # Both empty - no evidence either way

        if not set1 or not set2:
            return 0.0  # One empty - no overlap possible

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def calculate_relationship_overlap_by_cluster(self, ids1: Set[int], ids2: Set[int],
                                                   person_to_cluster: Dict[int, int]) -> float:
        """
        Calculate relationship overlap based on cluster membership (transitive matching).

        This implements the cluster-aware relationship matching from Kirielle et al.
        Two related persons are considered to "match" if they're in the same cluster,
        even if they're different Person objects. This handles cases like:
        - "Catharina Aarsen" and "Catharina Aarzen" (spelling variants)
        - Multiple extraction instances of the same person

        Args:
            ids1: Set of person IDs for first person's relationships (spouse/parent/child)
            ids2: Set of person IDs for second person's relationships
            person_to_cluster: Mapping from person_id to cluster_id

        Returns:
            float: Jaccard coefficient based on cluster overlap (0.0 to 1.0)
        """
        if not ids1 and not ids2:
            return 0.0

        if not ids1 or not ids2:
            return 0.0

        # Map person IDs to their cluster IDs
        clusters1 = set(person_to_cluster.get(pid, pid) for pid in ids1)
        clusters2 = set(person_to_cluster.get(pid, pid) for pid in ids2)

        # Calculate Jaccard similarity on cluster IDs
        intersection = len(clusters1 & clusters2)
        union = len(clusters1 | clusters2)

        return intersection / union if union > 0 else 0.0

    def calculate_overall_similarity(self, person1_id: int, person2_id: int,
                                     person_to_cluster: Optional[Dict[int, int]] = None) -> RelationalNode:
        """
        Calculate overall similarity between two person records.

        Uses weighted attribute similarity following Paper 2:
        - Must-match attributes (generation): Must be compatible or similarity = 0
        - Core attributes (names, birth year): High weight
        - Extra attributes (places, death info): Lower weight
        """
        p1 = self.persons[person1_id]
        p2 = self.persons[person2_id]

        rel_node = RelationalNode(person1_id, person2_id)

        # MUST-MATCH: Generation constraint
        # Require exact generation match (generation assignments are now reliable)
        if p1.generation is not None and p2.generation is not None:
            if p1.generation != p2.generation:
                rel_node.constraints_valid = False
                rel_node.constraint_violations.append(
                    f"Generation mismatch: {p1.generation} vs {p2.generation}"
                )
                rel_node.similarity = 0.0
                return rel_node

        # Define attribute weights
        # Based on Paper 2's findings + relationship overlap (Kirielle et al.)
        # Relationship overlap is VERY strong signal - two people sharing spouse/children
        # are very likely the same person
        # NOTE: Spouse matching is MORE important than name matching due to common names.
        # Many people in the family share given names (e.g., "Hendrik van Zanten").
        weights = {
            'given_names': 0.15,
            'surname': 0.15,
            'birth_year': 0.10,
            'birth_place': 0.05,
            'death_year': 0.05,
            'death_place': 0.05,
            'spouse_overlap': 0.30,
            'parent_overlap': 0.075,
            'child_overlap': 0.075,
        }

        total_weight = 0.0
        weighted_sim = 0.0

        attrs1 = p1.get_attributes()
        attrs2 = p2.get_attributes()

        # Calculate attribute similarities
        for attr_name, weight in weights.items():
            if attr_name.endswith('_overlap'):
                # Handle relationship overlaps separately
                continue

            val1 = attrs1.get(attr_name)
            val2 = attrs2.get(attr_name)

            # Skip if both are missing
            if val1 is None and val2 is None:
                continue

            # Calculate atomic similarity
            atomic_sim = self.calculate_atomic_similarity(attr_name, val1, val2)

            # Apply disambiguation weight (AMB)
            # Use the rarer of the two values for weighting
            amb_weight = 1.0
            if val1 is not None and val2 is not None and atomic_sim > 0:
                amb_weight = max(
                    self.calculate_disambiguation_weight(attr_name, val1),
                    self.calculate_disambiguation_weight(attr_name, val2)
                )

            # Combine
            final_weight = weight * amb_weight
            weighted_sim += final_weight * atomic_sim
            total_weight += final_weight

            # Store atomic similarity for debugging
            rel_node.atomic_sims[attr_name] = atomic_sim

        # Calculate relationship overlaps (Kirielle et al. - relational signal)
        # These are VERY strong indicators - if two person records share the same
        # spouse or children, they're very likely the same person
        # SPECIAL HANDLING: Parent overlap can indicate siblings OR recycled names
        # We use birth/death dates to distinguish

        # Spouse overlap - POSITIVE signal
        # Use cluster-aware matching if person_to_cluster mapping is available
        if person_to_cluster is not None:
            spouse_overlap = self.calculate_relationship_overlap_by_cluster(
                p1.spouse_ids, p2.spouse_ids, person_to_cluster
            )
        else:
            spouse_overlap = self.calculate_relationship_overlap(p1.spouse_names, p2.spouse_names)

        if spouse_overlap > 0:
            weighted_sim += weights['spouse_overlap'] * spouse_overlap
            total_weight += weights['spouse_overlap']
            rel_node.atomic_sims['spouse_overlap'] = spouse_overlap

        # Parent overlap - REQUIRES SPECIAL SIBLING/RECYCLED NAME DETECTION
        # Use cluster-aware matching if person_to_cluster mapping is available
        if person_to_cluster is not None:
            parent_overlap = self.calculate_relationship_overlap_by_cluster(
                p1.parent_ids, p2.parent_ids, person_to_cluster
            )
        else:
            parent_overlap = self.calculate_relationship_overlap(p1.parent_names, p2.parent_names)

        if parent_overlap > 0.5:  # Significant parent overlap
            rel_node.atomic_sims['parent_overlap'] = parent_overlap

            # Get name and temporal similarities
            given_name_sim = rel_node.atomic_sims.get('given_names', 0.0)

            # Check if they have birth/death date information
            have_temporal_data = (
                (p1.birth_year is not None or p1.death_year is not None) and
                (p2.birth_year is not None or p2.death_year is not None)
            )

            if given_name_sim < 0.7:
                # CASE 1: Different names + shared parents = SIBLINGS
                rel_node.constraints_valid = False
                rel_node.constraint_violations.append(
                    f'Siblings (shared parents, different names: "{p1.given_names}" vs "{p2.given_names}")'
                )
                rel_node.similarity = 0.0
                return rel_node

            elif have_temporal_data:
                # CASE 2: Same name + shared parents + temporal data available
                # Check if birth/death dates are incompatible (>5 year difference)
                temporal_mismatch = False

                if p1.birth_year and p2.birth_year:
                    if abs(p1.birth_year - p2.birth_year) > 5:
                        temporal_mismatch = True

                if p1.death_year and p2.death_year:
                    if abs(p1.death_year - p2.death_year) > 5:
                        temporal_mismatch = True

                if temporal_mismatch:
                    # Different birth/death dates = RECYCLED NAME (different people)
                    rel_node.constraints_valid = False
                    rel_node.constraint_violations.append(
                        f'Recycled name (shared parents, same name, different birth/death dates)'
                    )
                    rel_node.similarity = 0.0
                    return rel_node

            # CASE 3: Same name + shared parents + no temporal data OR compatible dates
            # This could be either:
            # - True duplicate (same person mentioned twice)
            # - Recycled name without enough data to distinguish
            # Let it through for human review, but DON'T add parent overlap as positive weight
            # (avoid boosting siblings)

        # Child overlap - POSITIVE signal
        # Use cluster-aware matching if person_to_cluster mapping is available
        if person_to_cluster is not None:
            child_overlap = self.calculate_relationship_overlap_by_cluster(
                p1.child_ids, p2.child_ids, person_to_cluster
            )
        else:
            child_overlap = self.calculate_relationship_overlap(p1.child_names, p2.child_names)

        if child_overlap > 0:
            weighted_sim += weights['child_overlap'] * child_overlap
            total_weight += weights['child_overlap']
            rel_node.atomic_sims['child_overlap'] = child_overlap

        # Calculate maximum possible weight based on available data
        # This prevents sparse records (with no relationships) from getting inflated scores
        max_possible_weight = 0.0

        # Basic attributes - always count these as "available" if either person has them
        for attr_name in ['given_names', 'surname', 'birth_year', 'birth_place', 'death_year', 'death_place']:
            val1 = attrs1.get(attr_name)
            val2 = attrs2.get(attr_name)
            if val1 is not None or val2 is not None:
                # At least one person has this attribute - count the weight
                weight = weights[attr_name]
                # Apply AMB if both have the attribute
                if val1 is not None and val2 is not None:
                    amb_weight = max(
                        self.calculate_disambiguation_weight(attr_name, val1),
                        self.calculate_disambiguation_weight(attr_name, val2)
                    )
                    max_possible_weight += weight * amb_weight
                else:
                    max_possible_weight += weight

        # Relationship overlaps - count as available if either person has the relationship
        if p1.spouse_names or p2.spouse_names:
            max_possible_weight += weights['spouse_overlap']
        if p1.parent_names or p2.parent_names:
            max_possible_weight += weights['parent_overlap']
        if p1.child_names or p2.child_names:
            max_possible_weight += weights['child_overlap']

        # Normalize by maximum possible weight (not just weight actually used)
        # This ensures sparse records get appropriately lower scores
        if max_possible_weight > 0:
            rel_node.similarity = weighted_sim / max_possible_weight
        else:
            rel_node.similarity = 0.0

        return rel_node

    def validate_constraints(self, person1_id: int, person2_id: int, rel_node: RelationalNode) -> bool:
        """
        Validate temporal and biological constraints (PROP-C technique).

        Returns False if merge would violate constraints.
        Updates rel_node.constraint_violations with details.
        """
        p1 = self.persons[person1_id]
        p2 = self.persons[person2_id]

        violations = []

        # Temporal constraint: Birth/death dates must be compatible
        if p1.birth_year and p2.birth_year:
            diff = abs(p1.birth_year - p2.birth_year)
            if diff > 5:  # Allow 5 year tolerance for recording errors
                violations.append(f"Birth year mismatch: {p1.birth_year} vs {p2.birth_year}")

        if p1.death_year and p2.death_year:
            diff = abs(p1.death_year - p2.death_year)
            if diff > 5:
                violations.append(f"Death year mismatch: {p1.death_year} vs {p2.death_year}")

        # Biological constraint: Death before birth is impossible
        if p1.birth_year and p1.death_year:
            if p1.death_year < p1.birth_year:
                violations.append(f"Person1 death before birth: {p1.birth_year} -> {p1.death_year}")

        if p2.birth_year and p2.death_year:
            if p2.death_year < p2.birth_year:
                violations.append(f"Person2 death before birth: {p2.birth_year} -> {p2.death_year}")

        # Combined birth/death constraint
        if p1.birth_year and p2.death_year:
            if abs(p1.birth_year - p2.death_year) > 120:  # Unlikely same person
                violations.append(f"Birth/death span too large: birth={p1.birth_year}, death={p2.death_year}")

        if violations:
            rel_node.constraint_violations.extend(violations)
            rel_node.constraints_valid = False
            return False

        return True

    def detect_partial_match_group(self, cluster: Set[int]) -> bool:
        """
        Detect if a cluster represents a partial match group (e.g., siblings).

        A partial match group occurs when:
        - Multiple people share parents (siblings)
        - Parents should be merged, but siblings should NOT be merged

        Returns True if this appears to be a partial match group.
        """
        if len(cluster) < 2:
            return False

        cluster_list = list(cluster)

        # Check if people in cluster have overlapping parents
        shared_parents = None
        for person_id in cluster_list:
            p = self.persons[person_id]
            if shared_parents is None:
                shared_parents = p.parent_ids.copy()
            else:
                shared_parents &= p.parent_ids

        # If they share parents, this might be siblings
        if shared_parents and len(shared_parents) > 0:
            # Check if they have different names (strong signal of different people)
            names = set()
            for person_id in cluster_list:
                p = self.persons[person_id]
                full_name = f"{p.given_names} {p.surname}".strip()
                names.add(full_name)

            # If names differ, likely siblings not duplicates
            if len(names) > 1:
                return True

        return False
