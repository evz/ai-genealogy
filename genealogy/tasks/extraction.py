"""Entity extraction tasks and utilities for genealogy documents

The business logic has been extracted to ExtractionService for better testability.
"""
import logging

from celery import shared_task
from django.core.exceptions import ValidationError
from django.db.models import Q

from ..extraction_strategies import get_strategy
from ..models import Document
from ..ollama_utils import OllamaClient, get_default_models
from ..services import ChunkEnrichmentService, ExtractionService

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def extract_entities_from_chunks(self, document_id: str):  # noqa: ARG001
    """
    Phase 2: Extract genealogy entities using section-specific strategies.

    Uses the strategy pattern to apply different extraction approaches based on
    BookSection types. Each section type maps to a specific extraction strategy.

    Args:
        document_id: UUID string of the Document to extract from

    Returns:
        dict: Extraction result summary
    """
    try:
        # Get the document
        document = Document.objects.get(id=document_id)
        logger.info(f"Starting entity extraction for document {document}")

        # Get all book sections
        sections = document.book_sections.all().order_by('start_page')

        if not sections.exists():
            logger.warning(f"No BookSections defined for document {document}")
            return {
                "success": False,
                "error": "No BookSections defined - please configure book sections first",
                "document_id": str(document_id),
            }

        # Initialize Ollama client once for all sections
        ollama = OllamaClient(timeout=3600)
        if not ollama.is_available():
            raise RuntimeError("Ollama server not available")

        # Get model from document config or use default
        model = document.llm_model_used or get_default_models()["llm_model"]
        logger.info(f"Using model: {model}")

        # Initialize extraction and enrichment services
        extraction_service = ExtractionService(ollama)
        enrichment_service = ChunkEnrichmentService(ollama)

        # Get embedding model from defaults
        embedding_model = get_default_models()["embedding_model"]
        logger.info(f"Using embedding model: {embedding_model}")

        total_chunks_processed = 0
        total_chunks_failed = 0
        total_enrichments_completed = 0
        sections_processed = {}

        # Process each section with its appropriate strategy
        for section in sections:
            logger.info(f"Processing section: {section.title} ({section.section_type}, pages {section.start_page}-{section.end_page})")

            # Get the extraction strategy for this section type
            try:
                strategy = get_strategy(section.section_type)
            except KeyError as e:
                logger.error(f"Unknown section type: {section.section_type}")
                continue

            logger.info(f"Using strategy: {strategy.strategy_name}")

            # Get unprocessed chunks for this section
            # Filter by page range and strategy-specific chunk filter
            chunk_filter = Q(
                start_page__gte=section.start_page,
                start_page__lte=section.end_page,
                entities_extracted=False,
                **strategy.get_chunk_filter()
            )

            unprocessed_chunks = document.text_chunks.filter(chunk_filter).order_by("sequence_number")
            chunk_count = unprocessed_chunks.count()

            logger.info(f"Found {chunk_count} chunks to process in this section")

            if chunk_count == 0:
                sections_processed[section.title] = {
                    'strategy': strategy.strategy_name,
                    'processed': 0,
                    'failed': 0,
                }
                continue

            # Use service to extract from chunks
            result = extraction_service.extract_from_chunks_in_section(
                chunks=unprocessed_chunks,
                section_type=section.section_type,
                model=model
            )

            section_processed = result['processed']
            section_failed = result['failed']

            total_chunks_processed += section_processed
            total_chunks_failed += section_failed

            # Enrich successfully extracted chunks with embeddings and DM codes
            if section_processed > 0:
                logger.info(f"Enriching chunks in section '{section.title}'")

                # Get ALL chunks in this section (not just those with entity extraction)
                # This ensures family headers, generation headers, etc. also get embeddings for RAG
                all_section_chunks = document.text_chunks.filter(
                    start_page__gte=section.start_page,
                    start_page__lte=section.end_page,
                ).order_by("sequence_number")

                # Enrich with embeddings and DM codes
                enrichment_result = enrichment_service.enrich_chunks_batch(
                    chunks=all_section_chunks,
                    embedding_model=embedding_model,
                    generate_embedding=True,
                    generate_dm_codes=True,
                    force=False,  # Skip chunks that already have enrichments
                )

                total_enrichments_completed += enrichment_result['processed']

                logger.info(
                    f"Enrichment complete for section '{section.title}': "
                    f"{enrichment_result['embeddings_generated']} embeddings, "
                    f"{enrichment_result['dm_codes_generated']} DM code sets generated "
                    f"({enrichment_result['total_dm_codes']} total codes)"
                )

            sections_processed[section.title] = {
                'strategy': strategy.strategy_name,
                'processed': section_processed,
                'failed': section_failed,
            }

            logger.info(
                f"Section '{section.title}' complete: "
                f"{section_processed} processed, {section_failed} failed"
            )

        # Mark document as extraction completed
        document.extraction_completed = True
        document.llm_model_used = model
        document.save(update_fields=["extraction_completed", "llm_model_used"])

        logger.info(
            f"Extraction complete for document {document}: "
            f"{total_chunks_processed} processed, {total_chunks_failed} failed, "
            f"{total_enrichments_completed} enriched across {len(sections_processed)} sections"
        )

        return {
            "success": True,
            "message": f"Processed {total_chunks_processed} chunks ({total_chunks_failed} failed), enriched {total_enrichments_completed}",
            "document_id": str(document_id),
            "chunks_processed": total_chunks_processed,
            "chunks_failed": total_chunks_failed,
            "chunks_enriched": total_enrichments_completed,
            "sections_processed": sections_processed,
        }

    except ValidationError:
        error_msg = f"Invalid UUID format: {document_id}"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except Document.DoesNotExist:
        error_msg = f"Document with id {document_id} not found"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
        }

    except Exception as e:
        error_msg = f"Entity extraction failed for {document_id}: {e!s}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "document_id": str(document_id),
        }
