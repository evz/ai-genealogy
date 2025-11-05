"""
Management command to backfill PersonMention records for existing TextChunks.

This command finds TextChunks with subjects and genealogical_identifiers but no
primary_person_mention, and creates PersonMention records for them.
"""
import logging

from django.core.management.base import BaseCommand
from django.db import transaction

from genealogy.models import PersonMention, TextChunk
from genealogy.utils import parse_name

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Backfill PersonMention records for existing TextChunks with subjects"

    def add_arguments(self, parser):
        parser.add_argument(
            '--document-id',
            type=str,
            help='Process only chunks from specific document ID'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be created without saving to database'
        )

    def handle(self, *args, **options):
        self.dry_run = options['dry_run']
        self.stats = {
            'chunks_processed': 0,
            'person_mentions_created': 0,
            'skipped_no_subject': 0,
            'skipped_no_genealogical_id': 0,
            'skipped_already_has_mention': 0,
        }

        if self.dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN MODE - No changes will be saved"))

        # Find chunks that need backfilling:
        # - Have chunk_type = 'GENEALOGY_ENTRY' (INDIVIDUAL_ENTRY chunks)
        # - Have a subject
        # - Have a genealogical_identifier
        # - Don't already have a primary_person_mention
        chunks = TextChunk.objects.filter(
            chunk_type='GENEALOGY_ENTRY',
            subject__isnull=False,
            genealogical_identifier__isnull=False,
            primary_person_mention__isnull=True,
        ).exclude(
            subject='',
            genealogical_identifier='',
        )

        if options['document_id']:
            chunks = chunks.filter(document_id=options['document_id'])

        chunks = chunks.select_related('document').order_by('document', 'sequence_number')

        total_chunks = chunks.count()
        self.stdout.write(f"Found {total_chunks} chunks that need PersonMention backfill")

        if total_chunks == 0:
            self.stdout.write(self.style.SUCCESS("Nothing to backfill!"))
            return

        for i, chunk in enumerate(chunks, 1):
            if i % 50 == 0:
                self.stdout.write(f"  Progress: {i}/{total_chunks} chunks")

            try:
                with transaction.atomic():
                    self._backfill_chunk(chunk)
                    if self.dry_run:
                        raise Exception("Dry run - rollback")
            except Exception as e:
                if not self.dry_run:
                    logger.error(f"Error backfilling chunk {chunk.id}: {e}", exc_info=True)
                    self.stdout.write(self.style.ERROR(f"  Error in chunk {chunk.sequence_number}: {e}"))

        # Print summary
        self.stdout.write(self.style.SUCCESS("\nBackfill Complete!"))
        self.stdout.write(f"  Chunks processed: {self.stats['chunks_processed']}")
        self.stdout.write(f"  PersonMentions created: {self.stats['person_mentions_created']}")
        self.stdout.write(f"  Skipped (no subject): {self.stats['skipped_no_subject']}")
        self.stdout.write(f"  Skipped (no genealogical_id): {self.stats['skipped_no_genealogical_id']}")
        self.stdout.write(f"  Skipped (already has mention): {self.stats['skipped_already_has_mention']}")

    def _backfill_chunk(self, chunk: TextChunk):
        """Backfill PersonMention for a single chunk"""
        self.stats['chunks_processed'] += 1

        # Double-check conditions (should be guaranteed by query, but be safe)
        if not chunk.subject:
            self.stats['skipped_no_subject'] += 1
            return

        if not chunk.genealogical_identifier:
            self.stats['skipped_no_genealogical_id'] += 1
            return

        if chunk.primary_person_mention:
            self.stats['skipped_already_has_mention'] += 1
            return

        # Parse name into given_names and surname
        given_names, surname = parse_name(chunk.subject)

        if self.dry_run:
            self.stdout.write(
                f"  Would create PersonMention: {chunk.subject} "
                f"(given='{given_names}', surname='{surname}', "
                f"genealogical_id='{chunk.genealogical_identifier}', "
                f"generation={chunk.generation_number})"
            )
            self.stats['person_mentions_created'] += 1
        else:
            # Check if a PersonMention with this genealogical_id already exists
            existing = PersonMention.objects.filter(
                genealogical_id=chunk.genealogical_identifier
            ).first()

            if existing:
                # Reuse existing PersonMention
                self.stdout.write(
                    self.style.WARNING(
                        f"  Reusing existing PersonMention for {chunk.genealogical_identifier}: {existing.full_name}"
                    )
                )
                chunk.primary_person_mention = existing
                chunk.save(update_fields=['primary_person_mention'])

                # Ensure bidirectional links
                existing.source_documents.add(chunk.document)
                existing.source_chunks.add(chunk)
            else:
                # Create new PersonMention
                person_mention = PersonMention.objects.create(
                    given_names=given_names,
                    surname=surname,
                    generation=chunk.generation_number,
                    genealogical_id=chunk.genealogical_identifier,
                )

                # Link to document and chunk
                person_mention.source_documents.add(chunk.document)
                person_mention.source_chunks.add(chunk)

                # Link chunk to person mention
                chunk.primary_person_mention = person_mention
                chunk.save(update_fields=['primary_person_mention'])

                logger.debug(
                    f"Created PersonMention for {chunk.subject} "
                    f"with genealogical_id={chunk.genealogical_identifier}"
                )

                self.stats['person_mentions_created'] += 1
