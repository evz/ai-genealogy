"""
Graph-based entity resolution for genealogical records.

This package implements the clustering algorithms from:
- "In-context Clustering-based Entity Resolution with Large Language Models"
- "Unsupervised Graph-based Entity Resolution for Accurate and Efficient Family Pedigree Search"
"""

from .nodes import AtomicNode, RelationalNode
from .person_record import PersonRecord
from .graph import DependencyGraph

__all__ = [
    'AtomicNode',
    'RelationalNode',
    'PersonRecord',
    'DependencyGraph',
]
