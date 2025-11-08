import logging
from collections import defaultdict
from uuid import UUID

from django.contrib import admin, messages
from django.db import transaction
from django.shortcuts import redirect, render
from django.urls import path, reverse
from django.utils import timezone

from ..models import (Identity, MentionToIdentity, MergeEvent, PersonMention,
                      PotentialDuplicate)
from .merge_logic import merge_mentions, unmerge_mentions

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
    search_fields = ['mention1__given_names', 'mention1__surname', 'mention2__given_names', 'mention2__surname']

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
            p1_id = dup.mention1_id
            p2_id = dup.mention2_id
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
            cluster['persons'] = list(PersonMention.objects.filter(
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
        from django.db.models import Count
        persons = PersonMention.objects.filter(id__in=cluster['person_ids']).prefetch_related(
            'events',
            'events__place',
            'parent_relationships',
            'parent_relationships__parent_mention',
            'child_relationships',
            'child_relationships__child_mention',
            'source_chunks',
            'source_documents',
            'mentiontoidentity__identity__mention_mappings'
        ).annotate(
            mention_count=Count('mentiontoidentity__identity__mention_mappings')
        ).order_by('given_names', 'surname')

        # Get pairwise similarity matrix and format it for easy template access
        # Create a lookup dict: {(mention1_id, mention2_id): similarity_data}
        similarity_lookup = {}
        for dup in PotentialDuplicate.objects.filter(
            mention1_id__in=cluster['person_ids'],
            mention2_id__in=cluster['person_ids']
        ):
            # Store in both directions for easy lookup
            similarity_lookup[(dup.mention1_id, dup.mention2_id)] = {
                'confidence': dup.confidence_score,
                'reasons': dup.match_reasons or []
            }
            similarity_lookup[(dup.mention2_id, dup.mention1_id)] = {
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
        """Show merge confirmation page or process merge"""
        clusters = self._compute_clusters(review_status='PENDING')

        if cluster_id >= len(clusters):
            messages.error(request, f"Cluster {cluster_id} not found")
            return redirect('admin:genealogy_potentialduplicate_cluster_list')

        cluster = clusters[cluster_id]

        # Check if this is the final merge submission (has 'confirm_merge' in POST)
        if request.method == 'POST' and 'confirm_merge' in request.POST:
            # Get all persons in cluster
            persons = list(PersonMention.objects.filter(id__in=cluster['person_ids']).prefetch_related(
                'events',
                'parent_relationships',
                'child_relationships',
                'partnerships',
                'source_chunks',
                'source_documents'
            ).order_by('given_names', 'surname'))
            return self._process_cluster_merge(request, cluster, persons)

        # Otherwise, show the confirmation page
        # Get selected person IDs (from detail page checkboxes)
        if request.method == 'POST':
            selected_ids = [UUID(mid) for mid in request.POST.getlist('persons_to_merge')]
            if len(selected_ids) < 2:
                messages.error(request, "Please select at least 2 persons to merge.")
                return redirect('admin:genealogy_potentialduplicate_cluster_detail', cluster_id=cluster_id)
        else:
            # If GET request, merge all persons in cluster
            selected_ids = cluster['person_ids']

        # Get Person objects for selected mentions
        persons = list(PersonMention.objects.filter(id__in=selected_ids).prefetch_related(
            'events',
            'parent_relationships',
            'child_relationships',
            'partnerships',
            'source_chunks',
            'source_documents'
        ).order_by('given_names', 'surname'))

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
            mention1_id__in=cluster['person_ids'],
            mention2_id__in=cluster['person_ids'],
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
        """Process the multi-person merge form using new reversible architecture"""
        try:
            # Get selected mentions to merge (may be subset of cluster)
            mention_ids_to_merge = [UUID(mid) for mid in request.POST.getlist('persons_to_merge')]

            if len(mention_ids_to_merge) < 2:
                messages.error(request, "Please select at least 2 persons to merge.")
                return redirect('admin:genealogy_potentialduplicate_cluster_merge', cluster_id=cluster['id'])

            # Build merge reason from PotentialDuplicate data
            confidences = []
            all_reasons = set()
            for dup in PotentialDuplicate.objects.filter(
                mention1_id__in=mention_ids_to_merge,
                mention2_id__in=mention_ids_to_merge
            ):
                confidences.append(dup.confidence_score)
                all_reasons.update(dup.match_reasons or [])

            avg_confidence = sum(confidences) / len(confidences) if confidences else 100.0

            merge_reason = {
                'cluster_id': cluster['id'],
                'cluster_size': len(mention_ids_to_merge),
                'avg_confidence': avg_confidence,
                'match_reasons': list(all_reasons),
                'merged_via': 'admin_ui'
            }

            # Perform the merge using our simple new logic!
            target_identity = merge_mentions(
                mention_ids=mention_ids_to_merge,
                target_identity_id=None,  # Create new identity
                merged_by=request.user.username,
                merge_reason=merge_reason
            )

            # Mark PotentialDuplicate pairs WITHIN the merged set as MERGED
            merged_count = 0
            for dup in PotentialDuplicate.objects.filter(
                mention1_id__in=mention_ids_to_merge,
                mention2_id__in=mention_ids_to_merge
            ):
                dup.review_status = 'MERGED'
                dup.reviewed_by = request.user.username
                dup.reviewed_at = timezone.now()
                dup.save()
                merged_count += 1

            # If this was a SUBSET of the cluster, mark pairs between merged and non-merged as REJECTED
            all_cluster_ids = set(cluster['person_ids'])
            non_merged_ids = all_cluster_ids - set(mention_ids_to_merge)

            rejected_count = 0
            if non_merged_ids:
                # Mark pairs between merged mentions and non-merged mentions as REJECTED
                for dup in PotentialDuplicate.objects.filter(
                    mention1_id__in=mention_ids_to_merge,
                    mention2_id__in=non_merged_ids,
                    review_status='PENDING'
                ):
                    dup.review_status = 'REJECTED'
                    dup.reviewed_by = request.user.username
                    dup.reviewed_at = timezone.now()
                    dup.save()
                    rejected_count += 1

                for dup in PotentialDuplicate.objects.filter(
                    mention1_id__in=non_merged_ids,
                    mention2_id__in=mention_ids_to_merge,
                    review_status='PENDING'
                ):
                    dup.review_status = 'REJECTED'
                    dup.reviewed_by = request.user.username
                    dup.reviewed_at = timezone.now()
                    dup.save()
                    rejected_count += 1

            success_msg = f"Successfully merged {len(mention_ids_to_merge)} mentions into identity: {target_identity.display_name}"
            if rejected_count > 0:
                success_msg += f" (and marked {rejected_count} pairs with non-merged mentions as REJECTED)"

            messages.success(request, success_msg)
            return redirect('admin:genealogy_potentialduplicate_cluster_list')

        except Exception as e:
            messages.error(request, f"Error during merge: {e}")
            logger.exception("Cluster merge failed")
            return redirect('admin:genealogy_potentialduplicate_cluster_list')
