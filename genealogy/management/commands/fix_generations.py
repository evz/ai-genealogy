"""
Management command to fix generation numbers based on parent-child relationships.

Walks the relationship graph and ensures children are always one generation higher than parents.
"""
import logging
from collections import defaultdict
from typing import Dict, Set

from django.core.management.base import BaseCommand
from django.db import transaction

from genealogy.models import ParentChildRelationship, Person

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Fix generation numbers based on parent-child relationship graph"

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be changed without saving to database'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed information for every person checked'
        )

    def handle(self, *args, **options):
        self.dry_run = options['dry_run']
        self.verbose = options['verbose']
        self.stats = {
            'checked': 0,
            'fixed': 0,
            'conflicts': 0,
            'correct': 0,
        }

        if self.dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN MODE - No changes will be saved\n"))

        # Build relationship graph
        self.stdout.write("Building relationship graph...")
        parent_to_children = defaultdict(set)
        child_to_parents = defaultdict(set)

        for rel in ParentChildRelationship.objects.select_related('parent', 'child'):
            parent_to_children[rel.parent_id].add(rel.child_id)
            child_to_parents[rel.child_id].add(rel.parent_id)

        # Find all persons with generation numbers
        persons = Person.objects.exclude(generation__isnull=True).in_bulk()

        self.stdout.write(f"Checking {len(persons)} persons...\n")

        # Check each person
        fixes = {}  # person_id -> (old_generation, new_generation, reason)

        for person_id, person in persons.items():
            self.stats['checked'] += 1

            if self.stats['checked'] % 100 == 0:
                self.stdout.write(f"  Progress: {self.stats['checked']}/{len(persons)} persons checked")

            # Get parent generations
            parent_ids = child_to_parents.get(person_id, set())
            parent_data = []
            for pid in parent_ids:
                if pid in persons and persons[pid].generation is not None:
                    parent_data.append({
                        'id': pid,
                        'name': persons[pid].full_name,
                        'generation': persons[pid].generation
                    })

            # Get children generations
            child_ids = parent_to_children.get(person_id, set())
            child_data = []
            for cid in child_ids:
                if cid in persons and persons[cid].generation is not None:
                    child_data.append({
                        'id': cid,
                        'name': persons[cid].full_name,
                        'generation': persons[cid].generation
                    })

            # Determine correct generation based on relationships
            correct_generation = None
            reason = []

            if parent_data:
                parent_generations = [p['generation'] for p in parent_data]
                max_parent_gen = max(parent_generations)
                expected_from_parents = max_parent_gen + 1

                if self.verbose:
                    self.stdout.write(f"\n{person.full_name} (currently Gen {person.generation}):")
                    for p in parent_data:
                        self.stdout.write(f"  Parent: {p['name']} (Gen {p['generation']})")
                    self.stdout.write(f"  Expected from parents: Gen {expected_from_parents}")

                if correct_generation is None:
                    correct_generation = expected_from_parents
                    reason.append(f"max parent gen ({max_parent_gen}) + 1")
                elif correct_generation != expected_from_parents:
                    self.stats['conflicts'] += 1
                    reason.append(f"CONFLICT: parents suggest {expected_from_parents}")
                    # Prefer parent-based generation
                    correct_generation = expected_from_parents

            if child_data:
                child_generations = [c['generation'] for c in child_data]
                min_child_gen = min(child_generations)
                expected_from_children = min_child_gen - 1

                if self.verbose:
                    if not parent_data:
                        self.stdout.write(f"\n{person.full_name} (currently Gen {person.generation}):")
                    for c in child_data:
                        self.stdout.write(f"  Child: {c['name']} (Gen {c['generation']})")
                    self.stdout.write(f"  Expected from children: Gen {expected_from_children}")

                if correct_generation is None:
                    correct_generation = expected_from_children
                    reason.append(f"min child gen ({min_child_gen}) - 1")
                elif correct_generation != expected_from_children:
                    # Only flag as conflict if we don't already have parent data
                    if not parent_data:
                        self.stats['conflicts'] += 1
                        reason.append(f"CONFLICT: children suggest {expected_from_children}")
                        correct_generation = expected_from_children
                    else:
                        # This is actually a child that needs fixing - not a conflict
                        pass

            # If we determined a correct generation and it differs from current, fix it
            if correct_generation is not None and correct_generation != person.generation:
                fixes[person_id] = {
                    'person': person,
                    'old_gen': person.generation,
                    'new_gen': correct_generation,
                    'reason': ', '.join(reason),
                    'parents': parent_data,
                    'children': child_data,
                }
                self.stats['fixed'] += 1
            elif correct_generation is not None:
                self.stats['correct'] += 1
                if self.verbose:
                    self.stdout.write(f"  ✓ Generation {correct_generation} is correct")

        # Show all fixes
        if fixes:
            self.stdout.write(self.style.WARNING(f"\n{'='*80}"))
            self.stdout.write(self.style.WARNING(f"PROPOSED CHANGES ({len(fixes)} persons)"))
            self.stdout.write(self.style.WARNING(f"{'='*80}\n"))

            for person_id, fix_data in sorted(fixes.items(), key=lambda x: (x[1]['old_gen'], x[1]['person'].full_name)):
                person = fix_data['person']
                old_gen = fix_data['old_gen']
                new_gen = fix_data['new_gen']
                reason = fix_data['reason']

                self.stdout.write(f"\n{person.full_name}:")
                self.stdout.write(f"  Current: Gen {old_gen}")
                self.stdout.write(f"  Fix to:  Gen {new_gen}")
                self.stdout.write(f"  Reason:  {reason}")

                if fix_data['parents']:
                    self.stdout.write("  Parents:")
                    for p in fix_data['parents']:
                        self.stdout.write(f"    - {p['name']} (Gen {p['generation']})")

                if fix_data['children']:
                    self.stdout.write("  Children:")
                    for c in fix_data['children'][:5]:  # Show max 5 children
                        self.stdout.write(f"    - {c['name']} (Gen {c['generation']})")
                    if len(fix_data['children']) > 5:
                        self.stdout.write(f"    ... and {len(fix_data['children']) - 5} more")

        # Apply fixes
        if fixes and not self.dry_run:
            self.stdout.write(f"\n{self.style.WARNING('Applying fixes...')}")
            with transaction.atomic():
                for person_id, fix_data in fixes.items():
                    Person.objects.filter(id=person_id).update(generation=fix_data['new_gen'])
            self.stdout.write(self.style.SUCCESS("✓ Fixes applied"))

        # Print summary
        self.stdout.write(self.style.SUCCESS(f"\n{'='*80}"))
        self.stdout.write(self.style.SUCCESS("SUMMARY"))
        self.stdout.write(self.style.SUCCESS(f"{'='*80}"))
        self.stdout.write(f"  Persons checked:      {self.stats['checked']}")
        self.stdout.write(f"  Already correct:      {self.stats['correct']}")
        self.stdout.write(self.style.WARNING(f"  Need fixing:          {self.stats['fixed']}"))
        self.stdout.write(f"  Conflicts detected:   {self.stats['conflicts']}")

        if self.dry_run and fixes:
            self.stdout.write(self.style.WARNING("\n⚠ DRY RUN - No changes were saved"))
            self.stdout.write("Run without --dry-run to apply these changes")
