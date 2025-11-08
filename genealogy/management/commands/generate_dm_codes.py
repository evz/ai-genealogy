"""
Management command to generate Daitch-Mokotoff phonetic codes for TextChunk instances.

This command uses the extracted_people field (extracted by LLM)
to generate DM codes for surname matching in the RAG+RRF system.
"""

import logging
import re
from typing import List, Set

from abydos.phonetic import DaitchMokotoff
from django.core.management.base import BaseCommand
from django.db import transaction

from genealogy.models import TextChunk

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Generate Daitch-Mokotoff phonetic codes from extracted_people field"

    def add_arguments(self, parser):
        parser.add_argument(
            "--chunk-ids",
            nargs="+",
            type=str,
            help="Specific chunk IDs to process (default: all chunks without DM codes)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Regenerate DM codes even if they already exist",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Number of chunks to process in each batch (default: 100)",
        )

    def handle(self, *args, **options):
        """Main command handler"""
        self.verbosity = options["verbosity"]
        batch_size = options["batch_size"]
        force = options["force"]
        chunk_ids = options.get("chunk_ids")

        self.dm_encoder = DaitchMokotoff()

        self.stdout.write("🚀 Starting DM code generation from LLM-extracted person names")

        # Get chunks to process
        if chunk_ids:
            chunks = TextChunk.objects.filter(id__in=chunk_ids)
        elif force:
            chunks = TextChunk.objects.exclude(extracted_people=[])  # Only chunks with person names
        else:
            chunks = TextChunk.objects.filter(dm_codes=[], extracted_people__len__gt=0)

        total_chunks = chunks.count()

        if total_chunks == 0:
            self.stdout.write("✅ No chunks with person names need DM code generation")
            return

        self.stdout.write(f"📝 Processing {total_chunks} text chunks with person names")

        # Convert to list to prevent queryset re-evaluation during iteration
        chunks_list = list(chunks)

        # Process in batches
        processed = 0
        failed = 0

        for i in range(0, total_chunks, batch_size):
            batch = chunks_list[i:i + batch_size]
            batch_results = self._process_batch(batch)

            processed += batch_results["success"]
            failed += batch_results["failed"]

            if self.verbosity >= 1:
                self.stdout.write(
                    f"📊 Batch {i//batch_size + 1}: "
                    f"{batch_results['success']} success, {batch_results['failed']} failed"
                )

        # Final summary
        self.stdout.write(
            self.style.SUCCESS(
                f"✅ DM code generation complete!\n"
                f"   Processed: {processed}\n"
                f"   Failed: {failed}\n"
                f"   Total: {total_chunks}"
            )
        )

    def _process_batch(self, chunks) -> dict:
        """Process a batch of chunks"""
        success_count = 0
        failed_count = 0

        for chunk in chunks:
            try:
                dm_codes = self._extract_dm_codes_from_names(chunk.extracted_people)

                with transaction.atomic():
                    chunk.dm_codes = dm_codes
                    chunk.save(update_fields=["dm_codes"])
                success_count += 1

                if self.verbosity >= 2:
                    self.stdout.write(
                        f"✅ Generated {len(dm_codes)} DM codes for chunk {chunk.id} "
                        f"from {len(chunk.extracted_people)} person names"
                    )

            except Exception as e:
                failed_count += 1
                if self.verbosity >= 1:
                    self.stdout.write(f"❌ Error processing chunk {chunk.id}: {e}")

        return {"success": success_count, "failed": failed_count}

    def _extract_dm_codes_from_names(self, extracted_people: List[str]) -> List[str]:
        """Extract DM codes from LLM-extracted person names"""
        if not extracted_people:
            return []

        dm_codes: Set[str] = set()

        for name in extracted_people:
            codes = self._get_dm_codes_for_name(name)
            dm_codes.update(codes)

        # Return as sorted list for consistency
        return sorted(list(dm_codes))

    def _get_dm_codes_for_name(self, name: str) -> List[str]:
        """Generate DM codes for a single name"""
        if not name or not isinstance(name, str):
            return []

        # Clean the name
        name = name.strip()
        if not name:
            return []

        codes = []

        # Split name into parts (first, middle, last names)
        parts = re.split(r'[,\s]+', name)

        for part in parts:
            part = part.strip()
            if len(part) >= 2:  # Only process names with at least 2 characters
                try:
                    # Clean the name part (remove prefixes, non-letters, etc.)
                    cleaned_part = self._clean_name_part(part)
                    if cleaned_part and len(cleaned_part) >= 2:
                        dm_code_set = self.dm_encoder.encode(cleaned_part)
                        # DaitchMokotoff.encode() returns a set of codes
                        if isinstance(dm_code_set, set):
                            for code in dm_code_set:
                                if code and code.strip():
                                    codes.append(code.strip())
                        elif dm_code_set and dm_code_set.strip():
                            codes.append(dm_code_set.strip())
                except Exception as e:
                    if self.verbosity >= 2:
                        self.stdout.write(f"⚠️  Failed to encode '{part}': {e}")

        return codes

    def _clean_name_part(self, name_part: str) -> str:
        """Clean a name part before DM encoding"""
        # Remove common Dutch/German prefixes
        prefixes = ['van', 'der', 'de', 'du', 'von', 'da', 'di', 'del', 'ter', 'ten', 'op']
        suffixes = ['jr', 'sr', 'ii', 'iii', 'iv', 'zoon', 'dochter']

        # Convert to lowercase for comparison
        lower_part = name_part.lower()

        # Remove prefixes
        for prefix in prefixes:
            if lower_part.startswith(prefix + ' '):
                name_part = name_part[len(prefix)+1:].strip()
                lower_part = name_part.lower()
                break

        # Remove suffixes
        for suffix in suffixes:
            if lower_part.endswith(' ' + suffix):
                name_part = name_part[:-len(suffix)-1].strip()
                break

        # Remove non-alphabetic characters except hyphens and apostrophes
        cleaned = re.sub(r'[^a-zA-Z\-\']', '', name_part)

        return cleaned
