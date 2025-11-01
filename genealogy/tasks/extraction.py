"""Entity extraction tasks and utilities for genealogy documents"""
import logging

from django.core.exceptions import ValidationError

from celery import shared_task

from ..models import Document, TextChunk
from ..ollama_utils import OllamaClient, get_default_models
from ..prompts import build_extraction_prompt, parse_extraction_output

logger = logging.getLogger(__name__)


def extract_entities_from_chunk(chunk, ollama, model):
    """
    Extract entities from a single chunk using LLM

    Args:
        chunk: TextChunk instance
        ollama: OllamaClient instance
        model: Model name to use for extraction

    Returns:
        dict: {
            'success': bool,
            'people_count': int,
            'relationships_count': int,
            'events_count': int,
            'error': str (if failed)
        }
    """
    try:
        logger.info(f"Extracting from chunk {chunk.sequence_number}")

        # Build extraction prompt
        prompt = build_extraction_prompt(chunk)

        # Calculate required context window based on chunk size
        # Estimate: chunk tokens + prompt overhead (~2000 tokens) + output buffer (~2000 tokens)
        estimated_tokens = len(chunk.text_content) // 4 + 4000

        # Round up to nearest power of 2 for efficiency
        # Min 4096 (efficient for small chunks), max 131072 (128K, model limit)
        num_ctx = max(4096, min(131072, 2 ** (estimated_tokens - 1).bit_length()))

        if num_ctx > 8192:
            logger.info(f"Large chunk: using context window {num_ctx} tokens (chunk: ~{len(chunk.text_content)//4} tokens)")

        # Query LLM
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'num_ctx': num_ctx,
                'temperature': 0.0,
            }
        )

        if not response:
            logger.error(f"No response from LLM for chunk {chunk.sequence_number}")
            return {
                'success': False,
                'error': 'No response from LLM'
            }

        # Parse output
        extracted_data = parse_extraction_output(response)

        # Save to chunk fields
        chunk.extracted_people = extracted_data['people']
        chunk.extracted_relationships = extracted_data['parent_child'] + extracted_data['partnerships']
        chunk.extracted_events = extracted_data['events']
        chunk.entities_extracted = True
        chunk.save(update_fields=[
            'extracted_people',
            'extracted_relationships',
            'extracted_events',
            'entities_extracted'
        ])

        result = {
            'success': True,
            'people_count': len(extracted_data['people']),
            'relationships_count': len(extracted_data['parent_child']) + len(extracted_data['partnerships']),
            'events_count': len(extracted_data['events'])
        }

        logger.info(
            f"Chunk {chunk.sequence_number}: "
            f"{result['people_count']} people, "
            f"{result['relationships_count']} rels, "
            f"{result['events_count']} events"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to extract from chunk {chunk.sequence_number}: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


@shared_task(bind=True)
def extract_entities_from_chunks(self, document_id: str):  # noqa: ARG001
    """
    Phase 2: Extract genealogy entities from GENEALOGY_ENTRY chunks using LLM

    Args:
        document_id: UUID string of the Document to extract from

    Returns:
        dict: Extraction result summary
    """
    try:
        # Get the document
        document = Document.objects.get(id=document_id)
        logger.info(f"Starting entity extraction for document {document}")

        # Get unprocessed GENEALOGY_ENTRY chunks
        unprocessed_chunks = document.text_chunks.filter(
            entities_extracted=False,
            chunk_type="GENEALOGY_ENTRY"
        ).order_by("sequence_number")

        if not unprocessed_chunks.exists():
            # Mark document as extraction completed
            document.extraction_completed = True
            document.save(update_fields=["extraction_completed"])

            return {
                "success": True,
                "message": "No chunks to process - extraction completed",
                "document_id": str(document_id),
                "chunks_processed": 0,
            }

        # Initialize Ollama client
        ollama = OllamaClient(timeout=3600)
        if not ollama.is_available():
            raise RuntimeError("Ollama server not available")

        # Get model from document config or use default
        model = document.llm_model_used or get_default_models()["llm_model"]
        logger.info(f"Using model: {model}")

        chunks_processed = 0
        chunks_failed = 0

        for chunk in unprocessed_chunks:
            result = extract_entities_from_chunk(chunk, ollama, model)
            if result['success']:
                chunks_processed += 1
            else:
                chunks_failed += 1

        # Mark document as extraction completed
        document.extraction_completed = True
        document.llm_model_used = model
        document.save(update_fields=["extraction_completed", "llm_model_used"])

        logger.info(
            f"Extraction complete for document {document}: "
            f"{chunks_processed} processed, {chunks_failed} failed"
        )

        return {
            "success": True,
            "message": f"Processed {chunks_processed} chunks ({chunks_failed} failed)",
            "document_id": str(document_id),
            "chunks_processed": chunks_processed,
            "chunks_failed": chunks_failed,
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
