import logging
from collections import defaultdict

from django.contrib import admin, messages
from django.db import transaction
from django.shortcuts import redirect, render
from django.urls import path, reverse
from django.utils import timezone

from ..models import (
    EntityMerge,
    Event,
    ParentChildRelationship,
    Partnership,
    Person,
    PotentialDuplicate,
)

logger = logging.getLogger(__name__)


class MatchReasonFilter(admin.SimpleListFilter):
    title = 'match reason'
    parameter_name = 'match_reason'

    def lookups(self, request, model_admin):
        return [
            ('cross_gen', 'Cross-generation (same person as child and parent)'),
            ('exact_name', 'Exact name match'),
            ('shared_parents', 'Same parents'),
            ('dm_code', 'Phonetic match (DM code)'),
            ('conflicting_events', 'Conflicting events (recycled name)'),
            ('same_birth', 'Same birth event'),
            ('same_death', 'Same death event'),
        ]

    def queryset(self, request, queryset):
        reason_map = {
            'cross_gen': 'same_name_and_gen',
            'exact_name': 'exact_name_normalized',
            'shared_parents': ['same_parents_family_group', 'same_parents_relationship', 'shared_parents'],
            'dm_code': 'dm_code_match',
            'conflicting_events': 'CONFLICTING_EVENTS',
            'same_birth': 'same_birth',
            'same_death': 'same_death',
        }

        if self.value() in reason_map:
            search_terms = reason_map[self.value()]
            if isinstance(search_terms, list):
                # Match any of the terms
                from django.db.models import Q
                q = Q()
                for term in search_terms:
                    q |= Q(match_reasons__contains=[term])
                return queryset.filter(q)
            else:
                return queryset.filter(match_reasons__contains=[search_terms])

        return queryset


@admin.register(PotentialDuplicate)
class PotentialDuplicateAdmin(admin.ModelAdmin):
    # Keep minimal list display - we'll override changelist_view to show clusters
    list_display = ['__str__']
    list_filter = ['review_status', MatchReasonFilter]
    search_fields = ['person1__given_names', 'person1__surname', 'person2__given_names', 'person2__surname']

    # We'll hide the default list view and show custom cluster view instead
    def has_add_permission(self, request):
        """Disable manual creation - duplicates should be detected automatically"""
        return False

    def changelist_view(self, request, extra_context=None):
        """Override to show cluster-based view instead of pairwise list"""
        # Check if we should show the cluster view
        if 'show_pairs' not in request.GET:
            # Redirect to custom cluster list
            return redirect(reverse('admin:genealogy_potentialduplicate_cluster_list'))

        # Otherwise show default (for backwards compat/debugging)
        return super().changelist_view(request, extra_context=extra_context)

    def _compute_clusters(self, review_status='PENDING'):
        """
        Compute clusters from PotentialDuplicate records.

        Returns:
            list of dicts with cluster info:
            [
                {
                    'id': 0,
                    'person_ids': [uuid1, uuid2, uuid3],
                    'size': 3,
                    'avg_confidence': 85.5,
                    'match_reasons': ['graph_cluster', 'exact_given_names', ...],
                },
                ...
            ]
        """
        # Build adjacency list
        connections = defaultdict(set)
        pair_data = {}  # (p1_id, p2_id) -> {'confidence': ..., 'reasons': ...}

        for dup in PotentialDuplicate.objects.filter(review_status=review_status):
            p1_id = dup.person1_id
            p2_id = dup.person2_id
            connections[p1_id].add(p2_id)
            connections[p2_id].add(p1_id)

            key = tuple(sorted([p1_id, p2_id]))
            pair_data[key] = {
                'confidence': dup.confidence_score,
                'reasons': dup.match_reasons or []
            }

        # Find connected components using DFS
        visited = set()
        clusters = []

        def dfs(person_id, cluster_members):
            if person_id in visited:
                return
            visited.add(person_id)
            cluster_members.add(person_id)
            for connected_id in connections[person_id]:
                dfs(connected_id, cluster_members)

        for person_id in connections:
            if person_id not in visited:
                cluster_members = set()
                dfs(person_id, cluster_members)

                # Get all pairs within this cluster
                member_list = list(cluster_members)
                confidences = []
                all_reasons = set()

                for i in range(len(member_list)):
                    for j in range(i + 1, len(member_list)):
                        key = tuple(sorted([member_list[i], member_list[j]]))
                        if key in pair_data:
                            confidences.append(pair_data[key]['confidence'])
                            all_reasons.update(pair_data[key]['reasons'])

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                clusters.append({
                    'id': len(clusters),
                    'person_ids': sorted(cluster_members),
                    'size': len(cluster_members),
                    'avg_confidence': avg_confidence,
                    'match_reasons': sorted(list(all_reasons)),
                })

        # Sort by size (largest first), then by confidence
        clusters.sort(key=lambda c: (-c['size'], -c['avg_confidence']))

        # Re-assign IDs after sorting
        for i, cluster in enumerate(clusters):
            cluster['id'] = i

        return clusters

    def get_urls(self):
        """Add custom URLs for cluster-based workflow"""
        urls = super().get_urls()
        custom_urls = [
            path(
                'clusters/',
                self.admin_site.admin_view(self.cluster_list_view),
                name='genealogy_potentialduplicate_cluster_list',
            ),
            path(
                'clusters/<int:cluster_id>/',
                self.admin_site.admin_view(self.cluster_detail_view),
                name='genealogy_potentialduplicate_cluster_detail',
            ),
            path(
                'clusters/<int:cluster_id>/merge/',
                self.admin_site.admin_view(self.cluster_merge_view),
                name='genealogy_potentialduplicate_cluster_merge',
            ),
            path(
                'clusters/<int:cluster_id>/reject/',
                self.admin_site.admin_view(self.cluster_reject_view),
                name='genealogy_potentialduplicate_cluster_reject',
            ),
        ]
        return custom_urls + urls

    def cluster_list_view(self, request):
        """Show list of all clusters (groups of potential duplicates)"""
        # Get search query
        search_query = request.GET.get('q', '').strip()

        clusters = self._compute_clusters(review_status='PENDING')

        # Enrich clusters with Person objects for display
        for cluster in clusters:
            cluster['persons'] = list(Person.objects.filter(
                id__in=cluster['person_ids']
            ).order_by('given_names', 'surname'))

        # Filter clusters by search query if provided
        if search_query:
            filtered_clusters = []
            query_lower = search_query.lower()
            for cluster in clusters:
                # Search in person names
                for person in cluster['persons']:
                    if (query_lower in person.given_names.lower() or
                        query_lower in person.surname.lower()):
                        filtered_clusters.append(cluster)
                        break
            clusters = filtered_clusters

        context = {
            'title': 'Duplicate Person Clusters',
            'clusters': clusters,
            'search_query': search_query,
            'opts': self.model._meta,
            'has_permission': True,
            'has_view_permission': True,
            'site_title': self.admin_site.site_title,
            'site_header': self.admin_site.site_header,
            'site_url': self.admin_site.site_url,
            'cl': None,  # No changelist object for custom view
            'has_add_permission': False,
            'has_change_permission': True,
            'has_delete_permission': False,
        }

        return render(request, 'admin/genealogy/potentialduplicate/cluster_list.html', context)

    def cluster_detail_view(self, request, cluster_id):
        """Show detailed view of a single cluster with all persons side-by-side"""
        clusters = self._compute_clusters(review_status='PENDING')

        if cluster_id >= len(clusters):
            messages.error(request, f"Cluster {cluster_id} not found")
            return redirect('admin:genealogy_potentialduplicate_cluster_list')

        cluster = clusters[cluster_id]

        # Get Person objects for all members
        persons = Person.objects.filter(id__in=cluster['person_ids']).prefetch_related(
            'events',
            'events__place',
            'parent_relationships',
            'parent_relationships__parent',
            'child_relationships',
            'child_relationships__child',
            'source_chunks',
            'source_documents'
        ).order_by('given_names', 'surname')

        # Get pairwise similarity matrix and format it for easy template access
        # Create a lookup dict: {(person1_id, person2_id): similarity_data}
        similarity_lookup = {}
        for dup in PotentialDuplicate.objects.filter(
            person1_id__in=cluster['person_ids'],
            person2_id__in=cluster['person_ids']
        ):
            # Store in both directions for easy lookup
            similarity_lookup[(dup.person1_id, dup.person2_id)] = {
                'confidence': dup.confidence_score,
                'reasons': dup.match_reasons or []
            }
            similarity_lookup[(dup.person2_id, dup.person1_id)] = {
                'confidence': dup.confidence_score,
                'reasons': dup.match_reasons or []
            }

        # Build a 2D matrix for the template - zip with persons for easy iteration
        persons_list = list(persons)
        matrix_with_persons = []
        for p1 in persons_list:
            row_data = {
                'person': p1,
                'similarities': []
            }
            for p2 in persons_list:
                if p1.id == p2.id:
                    row_data['similarities'].append(None)  # Same person
                else:
                    sim_data = similarity_lookup.get((p1.id, p2.id))
                    row_data['similarities'].append(sim_data)
            matrix_with_persons.append(row_data)

        context = {
            'title': f'Cluster #{cluster_id}: {cluster["size"]} Potential Duplicates',
            'cluster': cluster,
            'persons': persons_list,
            'matrix_with_persons': matrix_with_persons,
            'opts': self.model._meta,
            'has_permission': True,
        }

        return render(request, 'admin/genealogy/potentialduplicate/cluster_detail.html', context)

    def cluster_merge_view(self, request, cluster_id):
        """Merge all persons in a cluster into one"""
        clusters = self._compute_clusters(review_status='PENDING')

        if cluster_id >= len(clusters):
            messages.error(request, f"Cluster {cluster_id} not found")
            return redirect('admin:genealogy_potentialduplicate_cluster_list')

        cluster = clusters[cluster_id]

        # Get Person objects
        persons = list(Person.objects.filter(id__in=cluster['person_ids']).prefetch_related(
            'events',
            'parent_relationships',
            'child_relationships',
            'partnerships',
            'source_chunks',
            'source_documents'
        ).order_by('given_names', 'surname'))

        if request.method == 'POST':
            return self._process_cluster_merge(request, cluster, persons)

        context = {
            'title': f'Merge Cluster #{cluster_id}',
            'cluster': cluster,
            'persons': persons,
            'opts': self.model._meta,
            'has_permission': True,
        }

        return render(request, 'admin/genealogy/potentialduplicate/cluster_merge.html', context)

    def cluster_reject_view(self, request, cluster_id):
        """Mark entire cluster as not duplicates"""
        clusters = self._compute_clusters(review_status='PENDING')

        if cluster_id >= len(clusters):
            messages.error(request, f"Cluster {cluster_id} not found")
            return redirect('admin:genealogy_potentialduplicate_cluster_list')

        cluster = clusters[cluster_id]

        # Mark all pairs in this cluster as REJECTED
        affected = 0
        for dup in PotentialDuplicate.objects.filter(
            person1_id__in=cluster['person_ids'],
            person2_id__in=cluster['person_ids'],
            review_status='PENDING'
        ):
            dup.review_status = 'REJECTED'
            dup.reviewed_by = request.user.username
            dup.reviewed_at = timezone.now()
            dup.save()
            affected += 1

        messages.success(
            request,
            f"Marked cluster #{cluster_id} ({cluster['size']} persons, {affected} pairs) as NOT duplicates"
        )
        return redirect('admin:genealogy_potentialduplicate_cluster_list')

    def _process_cluster_merge(self, request, cluster, persons):
        """Process the multi-person merge form"""
        try:
            with transaction.atomic():
                # Get selected persons to merge (may be subset of cluster)
                persons_to_merge_ids = request.POST.getlist('persons_to_merge')

                if len(persons_to_merge_ids) < 2:
                    messages.error(request, "Please select at least 2 persons to merge.")
                    return redirect('admin:genealogy_potentialduplicate_cluster_merge', cluster_id=cluster['id'])

                # Filter to only the selected persons
                selected_persons = [p for p in persons if str(p.id) in persons_to_merge_ids]

                # Get base person (must be one of the selected persons)
                base_person_id = request.POST.get('base_person')
                if base_person_id not in persons_to_merge_ids:
                    messages.error(request, "Base person must be one of the selected persons to merge.")
                    return redirect('admin:genealogy_potentialduplicate_cluster_merge', cluster_id=cluster['id'])

                # Get the base person to copy attributes from
                base_person = Person.objects.get(id=base_person_id)

                # Create a NEW canonical entity (don't modify the original)
                canonical_person = Person.objects.create(
                    entity_type='CANONICAL',
                    given_names=request.POST.get('given_names', base_person.given_names),
                    surname=request.POST.get('surname', base_person.surname),
                    maiden_name=request.POST.get('maiden_name', base_person.maiden_name or ''),
                    gender=request.POST.get('gender', base_person.gender),
                    genealogical_id=request.POST.get('genealogical_id', base_person.genealogical_id or ''),
                    generation=int(request.POST.get('generation', base_person.generation or 0)) if request.POST.get('generation') else None
                )

                # Build pairwise confidence lookup from PotentialDuplicate records
                pairwise_confidences = {}
                for dup in PotentialDuplicate.objects.filter(
                    person1_id__in=persons_to_merge_ids,
                    person2_id__in=persons_to_merge_ids
                ):
                    key1 = (str(dup.person1_id), str(dup.person2_id))
                    key2 = (str(dup.person2_id), str(dup.person1_id))
                    pairwise_confidences[key1] = {
                        'confidence': dup.confidence_score,
                        'reasons': dup.match_reasons or []
                    }
                    pairwise_confidences[key2] = {
                        'confidence': dup.confidence_score,
                        'reasons': dup.match_reasons or []
                    }

                # Process ALL selected persons (including the base) as sources
                for source_person in selected_persons:
                    # Copy source documents and chunks to canonical
                    canonical_person.source_documents.add(*source_person.source_documents.all())
                    canonical_person.source_chunks.add(*source_person.source_chunks.all())

                    # Copy events to canonical (create new Event records pointing to canonical)
                    for event in source_person.events.all():
                        # Create a copy of the event for the canonical person
                        Event.objects.create(
                            event_type=event.event_type,
                            person=canonical_person,
                            date=event.date,
                            date_estimated=event.date_estimated,
                            place=event.place,
                            description=event.description
                        )

                    # Copy parent relationships to canonical
                    for rel in source_person.parent_relationships.all():
                        # Check if canonical already has this parent
                        if not ParentChildRelationship.objects.filter(
                            child=canonical_person, parent=rel.parent
                        ).exists():
                            ParentChildRelationship.objects.create(
                                child=canonical_person,
                                parent=rel.parent,
                                relationship_type=rel.relationship_type,
                                partnership=rel.partnership
                            )

                    # Copy child relationships to canonical
                    for rel in source_person.child_relationships.all():
                        if not ParentChildRelationship.objects.filter(
                            parent=canonical_person, child=rel.child
                        ).exists():
                            ParentChildRelationship.objects.create(
                                parent=canonical_person,
                                child=rel.child,
                                relationship_type=rel.relationship_type,
                                partnership=rel.partnership
                            )

                    # Copy partnerships to canonical
                    for partnership in source_person.partnerships.all():
                        # Get the other partner(s) in this partnership
                        other_partners = [p for p in partnership.partners.all() if p.id != source_person.id]

                        # Check if canonical already has a partnership with these same partners
                        existing_partnership = None
                        for cp_partnership in canonical_person.partnerships.all():
                            cp_partners = set(p.id for p in cp_partnership.partners.all() if p.id != canonical_person.id)
                            other_partner_ids = set(p.id for p in other_partners)
                            if cp_partners == other_partner_ids:
                                existing_partnership = cp_partnership
                                break

                        if not existing_partnership and other_partners:
                            # Create new partnership for canonical
                            new_partnership = Partnership.objects.create(
                                partnership_type=partnership.partnership_type,
                                start_date=partnership.start_date,
                                start_date_estimated=partnership.start_date_estimated,
                                start_place=partnership.start_place,
                                end_date=partnership.end_date,
                                end_date_estimated=partnership.end_date_estimated,
                                end_reason=partnership.end_reason
                            )
                            new_partnership.partners.add(canonical_person, *other_partners)

                    # Build pairwise similarities for this source entity
                    # (similarities with all OTHER sources in the cluster)
                    pairwise_sims = {}
                    for other_source in selected_persons:
                        if other_source.id != source_person.id:
                            key = (str(source_person.id), str(other_source.id))
                            if key in pairwise_confidences:
                                pairwise_sims[str(other_source.id)] = pairwise_confidences[key]['confidence']

                    # Get confidence for this specific source -> canonical pairing
                    # Use the confidence with the base_person as a representative score
                    key = (str(source_person.id), base_person_id)
                    pair_data = pairwise_confidences.get(key, {'confidence': 100.0, 'reasons': ['manual_merge']})

                    # Create EntityMerge record tracking this source -> canonical merge
                    EntityMerge.objects.create(
                        canonical_entity=canonical_person,
                        source_entity=source_person,
                        confidence_score=pair_data['confidence'],
                        pairwise_similarities=pairwise_sims,
                        merge_algorithm='manual',
                        merge_reason={
                            'match_reasons': pair_data['reasons'],
                            'cluster_id': cluster['id'],
                            'cluster_size': len(selected_persons),
                            'base_person_id': base_person_id
                        },
                        merged_by=request.user.username,
                        merged_at=timezone.now()
                    )

                    # Mark source person as merged
                    source_person.canonical_entity = canonical_person
                    source_person.save()

                # Mark pairs involving merged persons as MERGED
                merged_person_ids = [p.id for p in selected_persons]
                for dup in PotentialDuplicate.objects.filter(
                    person1_id__in=merged_person_ids,
                    person2_id__in=merged_person_ids
                ):
                    dup.review_status = 'MERGED'
                    dup.reviewed_by = request.user.username
                    dup.reviewed_at = timezone.now()
                    dup.save()

                # If not all persons in cluster were merged, keep remaining pairs as PENDING
                # (they might need separate review)

                messages.success(
                    request,
                    f"Successfully merged {len(selected_persons)} persons into new canonical entity: {canonical_person.full_name}"
                )
                return redirect('admin:genealogy_potentialduplicate_cluster_list')

        except Exception as e:
            messages.error(request, f"Error during merge: {e}")
            logger.exception("Cluster merge failed")
            return redirect('admin:genealogy_potentialduplicate_cluster_list')
