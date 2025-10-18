"""
Node classes for the dependency graph used in entity resolution.

These classes represent the atomic and relational nodes in the graph-based
clustering algorithm.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class AtomicNode:
    """
    Represents an atomic node in the dependency graph.
    An atomic node is an (attribute, value) pair for a person.
    """
    person_id: int
    attribute: str  # e.g., 'given_names', 'surname', 'birth_year', 'birth_place'
    value: any

    def __hash__(self):
        return hash((self.person_id, self.attribute, str(self.value)))

    def __eq__(self, other):
        return (self.person_id == other.person_id and
                self.attribute == other.attribute and
                self.value == other.value)


@dataclass
class RelationalNode:
    """
    Represents a relational node in the dependency graph.
    A relational node represents a potential match between two person records.
    """
    person1_id: int
    person2_id: int
    similarity: float = 0.0

    # Atomic similarities
    atomic_sims: Dict[str, float] = field(default_factory=dict)

    # Constraints
    constraints_valid: bool = True
    constraint_violations: List[str] = field(default_factory=list)

    # Dependencies
    supporting_matches: Set[Tuple[int, int]] = field(default_factory=set)

    def __hash__(self):
        # Ensure consistent ordering for undirected relationship
        p1, p2 = sorted([self.person1_id, self.person2_id])
        return hash((p1, p2))

    def __eq__(self, other):
        p1, p2 = sorted([self.person1_id, self.person2_id])
        o1, o2 = sorted([other.person1_id, other.person2_id])
        return p1 == o1 and p2 == o2
