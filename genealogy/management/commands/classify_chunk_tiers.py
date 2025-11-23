"""
Management command to classify existing TextChunks into search tiers.
"""

from django.core.management.base import BaseCommand

from genealogy.models import TextChunk
from genealogy.utils.chunk_classification import classify_chunk_tier, get_tier_statistics


class Command(BaseCommand):
    help = 'Classify existing TextChunks into metadata or narrative tiers'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be changed without actually updating',
        )
        parser.add_argument(
            '--chunk-type',
            type=str,
            help='Only classify chunks of this type (e.g., individual_entry)',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        chunk_type_filter = options.get('chunk_type')

        self.stdout.write(self.style.WARNING('Starting chunk tier classification...\n'))

        # Get chunks to classify
        queryset = TextChunk.objects.all()
        if chunk_type_filter:
            queryset = queryset.filter(chunk_type=chunk_type_filter)
            self.stdout.write(f'Filtering to chunk_type={chunk_type_filter}\n')

        total = queryset.count()
        self.stdout.write(f'Total chunks to classify: {total}\n')

        # Track changes
        changes = {
            'metadata': 0,
            'narrative': 0,
            'unchanged': 0
        }

        # Classify each chunk
        batch_size = 500
        for i in range(0, total, batch_size):
            batch = queryset[i:i+batch_size]

            for chunk in batch:
                new_tier = classify_chunk_tier(
                    text_content=chunk.text_content,
                    chunk_type=chunk.chunk_type,
                    extracted_events=chunk.extracted_events,
                    extracted_relationships=chunk.extracted_relationships
                )

                if chunk.search_tier != new_tier:
                    changes[new_tier] += 1

                    if not dry_run:
                        chunk.search_tier = new_tier
                        chunk.save(update_fields=['search_tier'])
                else:
                    changes['unchanged'] += 1

            # Progress indicator
            processed = min(i + batch_size, total)
            self.stdout.write(f'Processed {processed}/{total} chunks...', ending='\r')
            self.stdout.flush()

        self.stdout.write('\n')

        # Report results
        if dry_run:
            self.stdout.write(self.style.WARNING('\nDRY RUN - No changes made\n'))

        self.stdout.write(self.style.SUCCESS('\nClassification Summary:'))
        self.stdout.write(f'  Set to metadata tier: {changes["metadata"]}')
        self.stdout.write(f'  Set to narrative tier: {changes["narrative"]}')
        self.stdout.write(f'  Unchanged: {changes["unchanged"]}\n')

        # Get final statistics
        stats = get_tier_statistics(queryset)
        self.stdout.write(self.style.SUCCESS('Final Tier Distribution:'))
        for tier, count in stats['by_tier'].items():
            pct = stats['percentages'][tier]
            self.stdout.write(f'  {tier}: {count} chunks ({pct:.1f}%)')

        self.stdout.write(self.style.SUCCESS('\n✓ Classification complete'))
