"""
Management command to generate embeddings for TextChunk instances using Ollama.

This command connects to your Ollama server and generates vector embeddings
for text chunks to enable semantic search in the RAG+RRF system.
"""

import logging
import time
from typing import Optional

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from genealogy.models import TextChunk

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Generate embeddings for TextChunk instances using Ollama"

    def add_arguments(self, parser):
        parser.add_argument(
            "--chunk-ids",
            nargs="+",
            type=str,
            help="Specific chunk IDs to process (default: all chunks without embeddings)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Regenerate embeddings even if they already exist",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=10,
            help="Number of chunks to process in each batch (default: 10)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="zylonai/multilingual-e5-large:latest",
            help="Ollama embedding model to use",
        )

    def handle(self, *args, **options):
        """Main command handler"""
        self.verbosity = options["verbosity"]
        batch_size = options["batch_size"]
        force = options["force"]
        model = options["model"]
        chunk_ids = options.get("chunk_ids")

        # Get Ollama configuration
        ollama_host = getattr(settings, "OLLAMA_HOST", "localhost")
        ollama_port = getattr(settings, "OLLAMA_PORT", 11434)
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"

        self.stdout.write(f"🚀 Starting embedding generation using {model}")
        self.stdout.write(f"📡 Ollama server: {self.ollama_url}")

        # Test Ollama connection
        if not self._test_ollama_connection():
            raise CommandError("❌ Cannot connect to Ollama server")

        # Get chunks to process
        if chunk_ids:
            chunks = TextChunk.objects.filter(id__in=chunk_ids)
        elif force:
            chunks = TextChunk.objects.all()
        else:
            chunks = TextChunk.objects.filter(embedding__isnull=True)

        total_chunks = chunks.count()

        if total_chunks == 0:
            self.stdout.write("✅ No chunks need embedding generation")
            return

        self.stdout.write(f"📝 Processing {total_chunks} text chunks")

        # Convert to list to prevent queryset re-evaluation during iteration
        chunks_list = list(chunks)

        # Process in batches
        processed = 0
        failed = 0

        for i in range(0, total_chunks, batch_size):
            batch = chunks_list[i:i + batch_size]
            batch_results = self._process_batch(batch, model)

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
                f"✅ Embedding generation complete!\n"
                f"   Processed: {processed}\n"
                f"   Failed: {failed}\n"
                f"   Total: {total_chunks}"
            )
        )

    def _test_ollama_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            if self.verbosity >= 2:
                self.stdout.write("✅ Ollama server connection successful")
            return True
        except requests.RequestException as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Cannot connect to Ollama server: {e}")
            )
            return False

    def _process_batch(self, chunks, model: str) -> dict:
        """Process a batch of chunks"""
        success_count = 0
        failed_count = 0

        for chunk in chunks:
            try:
                embedding = self._generate_embedding(chunk.text_content, model)
                if embedding:
                    with transaction.atomic():
                        chunk.embedding = embedding
                        chunk.save(update_fields=["embedding"])
                    success_count += 1

                    if self.verbosity >= 2:
                        self.stdout.write(f"✅ Generated embedding for chunk {chunk.id}")
                else:
                    failed_count += 1
                    if self.verbosity >= 1:
                        self.stdout.write(f"❌ Failed to generate embedding for chunk {chunk.id}")

            except Exception as e:
                failed_count += 1
                if self.verbosity >= 1:
                    self.stdout.write(f"❌ Error processing chunk {chunk.id}: {e}")

        return {"success": success_count, "failed": failed_count}

    def _generate_embedding(self, text: str, model: str) -> Optional[list]:
        """Generate embedding for text using Ollama"""
        try:
            # Clean and prepare text
            clean_text = text.strip()
            if not clean_text:
                return None

            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": clean_text
                },
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            embedding = result.get("embedding")

            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                return embedding
            else:
                if self.verbosity >= 2:
                    self.stdout.write(f"❌ Invalid embedding response: {result}")
                return None

        except requests.RequestException as e:
            if self.verbosity >= 1:
                self.stdout.write(f"❌ Ollama API error: {e}")
            return None
        except Exception as e:
            if self.verbosity >= 1:
                self.stdout.write(f"❌ Unexpected error: {e}")
            return None
