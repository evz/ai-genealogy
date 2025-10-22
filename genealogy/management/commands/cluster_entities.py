"""
Management command to cluster Person entities using graph-based entity resolution.

Implements techniques from:
- "In-context Clustering-based Entity Resolution with Large Language Models"
- "Unsupervised Graph-based Entity Resolution for Accurate and Efficient Family Pedigree Search"

Uses dependency graphs, constraint propagation, and hierarchical clustering to:
1. Build a dependency graph of Person records and their relationships
2. Calculate similarity scores with constraint validation
3. Bootstrap high-confidence matches
4. Iteratively merge entities while handling partial match groups (siblings)
5. Refine clusters using graph topology
"""
import logging
from collections import defaultdict
from typing import Set, List

from django.core.management.base import BaseCommand
from django.db.models import Prefetch

from genealogy.clustering import DependencyGraph, PersonRecord
from genealogy.models import PersonMention, PartnershipMention, RelationshipMention, PotentialDuplicate

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Cluster Person entities using graph-based entity resolution"

    def add_arguments(self, parser):
        parser.add_argument(
            '--similarity-threshold',
            type=float,
            default=0.75,
            help='Minimum similarity score for bootstrapping high-confidence matches (default: 0.75)'
        )
        parser.add_argument(
            '--merge-threshold',
            type=float,
            default=0.60,
            help='Minimum similarity score for iterative merging (default: 0.60)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be clustered without making changes'
        )
        parser.add_argument(
            '--max-iterations',
            type=int,
            default=10,
            help='Maximum iterations for iterative merging (default: 10)'
        )
        parser.add_argument(
            '--clean',
            action='store_true',
            help='Delete all existing PotentialDuplicate records before clustering'
        )

    def handle(self, *args, **options):
        self.dry_run = options['dry_run']
        self.similarity_threshold = options['similarity_threshold']
        self.merge_threshold = options['merge_threshold']
        self.max_iterations = options['max_iterations']

        if self.dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN MODE - No changes will be saved"))

        # Clean existing duplicates if requested
        if options['clean']:
            if self.dry_run:
                count = PotentialDuplicate.objects.count()
                self.stdout.write(self.style.WARNING(f"Would delete {count} existing PotentialDuplicate records"))
            else:
                count = PotentialDuplicate.objects.count()
                PotentialDuplicate.objects.all().delete()
                self.stdout.write(self.style.SUCCESS(f"Deleted {count} existing PotentialDuplicate records"))

        self.stdout.write("Building dependency graph...")
        graph = self._build_dependency_graph()

        self.stdout.write(f"Graph built: {len(graph.persons)} persons")

        self.stdout.write("\nPhase 1: Bootstrap high-confidence matches...")
        clusters = self._bootstrap_clusters(graph)

        self.stdout.write(f"Bootstrap complete: {len(clusters)} initial clusters")

        self.stdout.write("\nPhase 2: Iterative merging with constraint validation...")
        clusters = self._iterative_merge(graph, clusters)

        self.stdout.write(f"Iterative merge complete: {len(clusters)} intermediate clusters")

        self.stdout.write("\nPhase 2.25: Infer missing partnerships from shared children...")
        partnerships_created = self._infer_partnerships_from_children(graph, clusters)
        self.stdout.write(f"Inferred {partnerships_created} partnerships from shared children")

        self.stdout.write("\nPhase 2.5: Cluster refinement with transitive relationship matching...")
        clusters = self._refine_clusters_with_transitive_relationships(graph, clusters)

        self.stdout.write(f"Refinement complete: {len(clusters)} final clusters")

        self.stdout.write("\nPhase 3: Apply clusters to database...")
        self._apply_clusters(graph, clusters)

        # Print summary
        self.stdout.write(self.style.SUCCESS("\nClustering Complete!"))

        # Count cluster statistics
        singleton_count = sum(1 for c in clusters if len(c) == 1)
        multi_count = len(clusters) - singleton_count
        total_persons = sum(len(c) for c in clusters)

        self.stdout.write(f"  Total persons: {total_persons}")
        self.stdout.write(f"  Singleton clusters: {singleton_count}")
        self.stdout.write(f"  Multi-person clusters: {multi_count}")

        if multi_count > 0:
            avg_cluster_size = sum(len(c) for c in clusters if len(c) > 1) / multi_count
            self.stdout.write(f"  Average cluster size (multi): {avg_cluster_size:.2f}")

    def _build_dependency_graph(self) -> DependencyGraph:
        """Build the dependency graph from Person records"""
        graph = DependencyGraph()

        # Prefetch related data for efficiency
        persons = PersonMention.objects.prefetch_related(
            'events',
            'events__place',
            'parent_relationships',
            'child_relationships',
            Prefetch('partnerships', queryset=PartnershipMention.objects.prefetch_related('partners'))
        ).all()

        for person in persons:
            person_record = PersonRecord(person)
            graph.add_person(person_record)

        return graph

    def _bootstrap_clusters(self, graph: DependencyGraph) -> List[Set[int]]:
        """
        Phase 1: Bootstrap high-confidence matches.

        Creates initial clusters by matching on high similarity (>= threshold).
        Uses Union-Find for efficient clustering.
        """
        # Union-Find data structure
        parent = {pid: pid for pid in graph.persons.keys()}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Calculate similarities for all pairs
        person_ids = list(graph.persons.keys())
        match_count = 0

        for i, p1_id in enumerate(person_ids):
            if i % 100 == 0:
                self.stdout.write(f"  Processing person {i}/{len(person_ids)}")

            p1 = graph.persons[p1_id]

            # Only compare within same generation (exact match required)
            for p2_id in person_ids[i+1:]:
                p2 = graph.persons[p2_id]

                # Quick generation filter - require exact match
                if p1.generation is not None and p2.generation is not None:
                    if p1.generation != p2.generation:
                        continue

                # Calculate similarity
                rel_node = graph.calculate_overall_similarity(p1_id, p2_id)

                # Validate constraints
                graph.validate_constraints(p1_id, p2_id, rel_node)

                # Store relational node
                key = tuple(sorted([p1_id, p2_id]))
                graph.relational_nodes[key] = rel_node

                # Bootstrap: merge if high confidence
                if rel_node.constraints_valid and rel_node.similarity >= self.similarity_threshold:
                    union(p1_id, p2_id)
                    match_count += 1

        self.stdout.write(f"  Found {match_count} high-confidence matches")

        # Convert Union-Find to clusters
        clusters_dict = defaultdict(set)
        for pid in graph.persons.keys():
            root = find(pid)
            clusters_dict[root].add(pid)

        return list(clusters_dict.values())

    def _iterative_merge(self, graph: DependencyGraph, clusters: List[Set[int]]) -> List[Set[int]]:
        """
        Phase 2: Iterative merging with constraint propagation.

        Iteratively merge clusters based on:
        - Similarity scores above merge_threshold
        - Constraint validation
        - Partial match group detection
        """
        for iteration in range(self.max_iterations):
            self.stdout.write(f"  Iteration {iteration + 1}/{self.max_iterations}")

            merged_any = False
            new_clusters = []
            merged_flags = [False] * len(clusters)

            for i, cluster1 in enumerate(clusters):
                if merged_flags[i]:
                    continue

                best_merge = None
                best_similarity = self.merge_threshold

                for j in range(i + 1, len(clusters)):
                    if merged_flags[j]:
                        continue

                    cluster2 = clusters[j]

                    # Calculate inter-cluster similarity (average of pairwise similarities)
                    sims = []
                    for p1_id in cluster1:
                        for p2_id in cluster2:
                            key = tuple(sorted([p1_id, p2_id]))

                            if key in graph.relational_nodes:
                                rel_node = graph.relational_nodes[key]
                            else:
                                # Calculate if not already done
                                rel_node = graph.calculate_overall_similarity(p1_id, p2_id)
                                graph.validate_constraints(p1_id, p2_id, rel_node)
                                graph.relational_nodes[key] = rel_node

                            if rel_node.constraints_valid:
                                sims.append(rel_node.similarity)

                    if not sims:
                        continue

                    avg_sim = sum(sims) / len(sims)

                    if avg_sim > best_similarity:
                        # Check for partial match group (siblings)
                        potential_merge = cluster1 | cluster2
                        if not graph.detect_partial_match_group(potential_merge):
                            best_similarity = avg_sim
                            best_merge = j

                if best_merge is not None:
                    # Merge clusters
                    merged_cluster = cluster1 | clusters[best_merge]
                    new_clusters.append(merged_cluster)
                    merged_flags[i] = True
                    merged_flags[best_merge] = True
                    merged_any = True
                else:
                    new_clusters.append(cluster1)
                    merged_flags[i] = True

            clusters = new_clusters

            if not merged_any:
                self.stdout.write("  No more merges possible, stopping early")
                break

        return clusters

    def _infer_partnerships_from_children(self, graph: DependencyGraph, clusters: List[Set[int]]) -> int:
        """
        Phase 2.25: Infer missing partnerships from shared children.

        If two people share 2+ children, they are almost certainly spouses, even if no
        explicit Partnership record was extracted. This handles cases where:
        - A person is mentioned as a parent in children's chunks but has no own biographical entry
        - The LLM failed to extract the partnership from the family group header

        This creates Partnership records in the database AND updates the graph's PersonRecords
        so that Phase 2.5 (refinement) can use these inferred partnerships for clustering.

        Returns:
            int: Number of partnerships created
        """
        partnerships_created = 0

        # Build mapping of child -> parents
        child_to_parents = defaultdict(set)
        for person_id, person_record in graph.persons.items():
            for child_id in person_record.child_ids:
                child_to_parents[child_id].add(person_id)

        # Find pairs of people who share 2+ children
        parent_pairs = defaultdict(set)  # (parent1, parent2) -> set of shared child_ids

        for child_id, parent_ids in child_to_parents.items():
            parent_list = list(parent_ids)
            # For each pair of parents of this child
            for i in range(len(parent_list)):
                for j in range(i + 1, len(parent_list)):
                    p1_id, p2_id = sorted([parent_list[i], parent_list[j]])
                    parent_pairs[(p1_id, p2_id)].add(child_id)

        # Create partnerships for pairs with 2+ shared children
        for (parent1_id, parent2_id), shared_children in parent_pairs.items():
            if len(shared_children) < 2:
                continue  # Need at least 2 shared children

            # Check if partnership already exists
            p1_record = graph.persons[parent1_id]
            p2_record = graph.persons[parent2_id]

            if parent2_id in p1_record.spouse_ids:
                # Partnership already exists
                continue

            # Check if they're in the same cluster (siblings) - don't create partnership for siblings
            same_cluster = False
            for cluster in clusters:
                if parent1_id in cluster and parent2_id in cluster:
                    same_cluster = True
                    break

            if same_cluster:
                # These might be siblings sharing children (unlikely but possible in complex families)
                continue

            # Create PartnershipMention in database (if not dry run)
            if not self.dry_run:
                parent1 = p1_record.person
                parent2 = p2_record.person

                # Check if partnership already exists in database
                existing = PartnershipMention.objects.filter(
                    partners=parent1
                ).filter(
                    partners=parent2
                ).first()

                if not existing:
                    partnership = PartnershipMention.objects.create(
                        partnership_type='MARRIAGE'
                    )
                    partnership.partners.add(parent1, parent2)
                    # Link to source documents from shared children
                    for child_id in shared_children:
                        child_person = PersonMention.objects.get(id=child_id)
                        for doc in child_person.source_documents.all():
                            partnership.source_documents.add(doc)

                    partnerships_created += 1

                    # Update the PersonRecords in the graph so refinement phase can use this
                    p1_record.spouse_ids.add(parent2_id)
                    p2_record.spouse_ids.add(parent1_id)

                    # Also update spouse_names for name-based matching
                    spouse1_name = p1_record._normalize_name(p1_record.given_names, p1_record.surname)
                    spouse2_name = p2_record._normalize_name(p2_record.given_names, p2_record.surname)
                    p1_record.spouse_names.add(spouse2_name)
                    p2_record.spouse_names.add(spouse1_name)
            else:
                partnerships_created += 1

        return partnerships_created

    def _refine_clusters_with_transitive_relationships(self, graph: DependencyGraph,
                                                       clusters: List[Set[int]]) -> List[Set[int]]:
        """
        Phase 2.5: Refine clusters using transitive relationship matching (Kirielle et al.).

        After initial clustering, re-compute similarities using cluster-aware relationship
        matching. This handles cases where:
        - Person A is married to "Catharina Aarsen" (person_id=1)
        - Person B is married to "Catharina Aarzen" (person_id=2)
        - person_id=1 and person_id=2 are in the same cluster (spelling variants)
        - Therefore A and B should be detected as duplicates (same spouse cluster)

        Iterates until convergence (no new merges).
        """
        # Build person_to_cluster mapping
        person_to_cluster = {}
        for cluster_id, cluster in enumerate(clusters):
            for person_id in cluster:
                person_to_cluster[person_id] = cluster_id

        # Iterative refinement loop
        for ref_iteration in range(self.max_iterations):
            self.stdout.write(f"  Refinement iteration {ref_iteration + 1}/{self.max_iterations}")

            # Try to merge clusters using cluster-aware relationship matching
            merged_any = False
            new_clusters = []
            merged_flags = [False] * len(clusters)

            for i, cluster1 in enumerate(clusters):
                if merged_flags[i]:
                    continue

                best_merge = None
                best_similarity = self.merge_threshold

                for j in range(i + 1, len(clusters)):
                    if merged_flags[j]:
                        continue

                    cluster2 = clusters[j]

                    # Re-calculate similarities using cluster-aware relationship matching
                    sims = []
                    for p1_id in cluster1:
                        for p2_id in cluster2:
                            # Use cluster-aware similarity calculation
                            rel_node = graph.calculate_overall_similarity(
                                p1_id, p2_id, person_to_cluster=person_to_cluster
                            )
                            graph.validate_constraints(p1_id, p2_id, rel_node)

                            if rel_node.constraints_valid:
                                sims.append(rel_node.similarity)

                    if not sims:
                        continue

                    avg_sim = sum(sims) / len(sims)

                    if avg_sim > best_similarity:
                        # Check for partial match group (siblings)
                        potential_merge = cluster1 | cluster2
                        if not graph.detect_partial_match_group(potential_merge):
                            best_similarity = avg_sim
                            best_merge = j

                if best_merge is not None:
                    # Merge clusters
                    merged_cluster = cluster1 | clusters[best_merge]
                    new_clusters.append(merged_cluster)
                    merged_flags[i] = True
                    merged_flags[best_merge] = True
                    merged_any = True
                else:
                    new_clusters.append(cluster1)
                    merged_flags[i] = True

            clusters = new_clusters

            if not merged_any:
                self.stdout.write("  No new merges found, refinement complete")
                break

            # Update person_to_cluster mapping for next iteration
            person_to_cluster = {}
            for cluster_id, cluster in enumerate(clusters):
                for person_id in cluster:
                    person_to_cluster[person_id] = cluster_id

        return clusters

    def _apply_clusters(self, graph: DependencyGraph, clusters: List[Set[int]]):
        """
        Phase 3: Apply clusters to database.

        For each cluster with >1 person, create PotentialDuplicate records
        for human review and potential automatic merging.

        IMPORTANT: Only create records for pairs that were actually computed
        (i.e., have a relational node). This avoids creating spurious 0% confidence
        records from transitive closure.
        """
        duplicates_created = 0

        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            cluster_list = sorted(list(cluster))

            # Only create records for pairs that were actually computed
            # This avoids the transitive closure problem where A-B and B-C
            # cause A-C to be in the same cluster but with no actual similarity computed
            for i in range(len(cluster_list)):
                for j in range(i + 1, len(cluster_list)):
                    p1_id = cluster_list[i]
                    p2_id = cluster_list[j]

                    # Only process if we actually computed this pair
                    key = tuple(sorted([p1_id, p2_id]))
                    if key not in graph.relational_nodes:
                        continue  # Skip - this pair wasn't directly compared

                    rel_node = graph.relational_nodes[key]
                    similarity = rel_node.similarity

                    # Only create PotentialDuplicate records for pairs above merge_threshold
                    # This prevents low-confidence transitive links from polluting the database
                    if similarity < self.merge_threshold:
                        continue

                    # Build detailed match reasons from atomic similarities
                    match_reasons = ['graph_cluster']
                    for attr_name, atomic_sim in rel_node.atomic_sims.items():
                        if atomic_sim >= 0.9:
                            match_reasons.append(f'exact_{attr_name}')
                        elif atomic_sim >= 0.7:
                            match_reasons.append(f'similar_{attr_name}')

                    if not self.dry_run:
                        # Create or update PotentialDuplicate record
                        # Convert similarity (0-1) to confidence_score (0-100)
                        PotentialDuplicate.objects.update_or_create(
                            mention1_id=min(p1_id, p2_id),
                            mention2_id=max(p1_id, p2_id),
                            defaults={
                                'confidence_score': similarity * 100.0,
                                'match_reasons': match_reasons,
                                'review_status': 'PENDING'
                            }
                        )

                    duplicates_created += 1

        self.stdout.write(f"  Created {duplicates_created} potential duplicate records")
