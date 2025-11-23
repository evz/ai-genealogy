"""
Remove embeddings from metadata-tier chunks to save space.
"""

from django.core.management.base import BaseCommand

from genealogy.models import TextChunk


class Command(BaseCommand):
    help = 'Remove embeddings from metadata-tier chunks'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        # Find metadata chunks with embeddings
        metadata_with_embeddings = TextChunk.objects.filter(
            search_tier='metadata',
            embedding__isnull=False
        )

        count = metadata_with_embeddings.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS('No metadata chunks have embeddings. Nothing to clean.'))
            return

        self.stdout.write(f'Found {count} metadata chunks with embeddings')

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - No changes will be made'))

            # Show some examples
            self.stdout.write('\nExample chunks that would be cleaned:')
            for chunk in metadata_with_embeddings[:5]:
                self.stdout.write(f'  [{chunk.genealogical_identifier}] {chunk.subject} ({len(chunk.text_content)} chars)')
        else:
            # Set embeddings to None
            metadata_with_embeddings.update(embedding=None)
            self.stdout.write(self.style.SUCCESS(f'✓ Removed embeddings from {count} metadata chunks'))

            # Show space savings estimate
            self.stdout.write(f'\nEstimated space saved: ~{count * 1024 * 4 / 1024 / 1024:.1f} MB')
