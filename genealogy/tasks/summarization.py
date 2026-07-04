"""Celery tasks for generating text chunk summaries"""

import logging

from celery import shared_task

from ..models import TextChunk
from ..ollama_utils import OllamaClient

logger = logging.getLogger(__name__)

# Minimum text length to trigger summarization
MIN_LENGTH_FOR_SUMMARY = 1000

# Model to use for summarization (fast model)
SUMMARY_MODEL = "llama3.1:8b"


def generate_summary(text: str, target_reduction: float = 0.5) -> str | None:
    """
    Generate a summary of genealogical text using a fast LLM.

    Args:
        text: Original text to summarize
        target_reduction: Target size as fraction of original (0.5 = half size)

    Returns:
        Summary string, or None if summarization failed
    """
    target_length = int(len(text) * target_reduction)

    prompt = f"""Summarize the following genealogical text.
- Keep the SAME LANGUAGE as the original (Dutch or English)
- Preserve ALL names, dates, places, and occupations
- Target length: approximately {target_length} characters
- Keep biographical facts, remove narrative filler and repetition

TEXT:
{text}

SUMMARY:"""

    try:
        client = OllamaClient()
        summary = client.generate(
            model=SUMMARY_MODEL,
            prompt=prompt,
            temperature=0.1,
            num_ctx=8192
        )
        return summary.strip() if summary else None
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return None


@shared_task(bind=True, max_retries=3)
def summarize_chunk(self, chunk_id: str) -> dict:
    """
    Generate summary for a single text chunk.

    Args:
        chunk_id: UUID of the TextChunk to summarize

    Returns:
        {"status": "success"|"skipped"|"error", "chunk_id": str, ...}
    """
    try:
        chunk = TextChunk.objects.get(id=chunk_id)

        # Skip if already has summary
        if chunk.text_summary:
            return {
                "status": "skipped",
                "chunk_id": str(chunk_id),
                "reason": "already_summarized"
            }

        # Skip short chunks
        if len(chunk.text_content) < MIN_LENGTH_FOR_SUMMARY:
            return {
                "status": "skipped",
                "chunk_id": str(chunk_id),
                "reason": "too_short",
                "length": len(chunk.text_content)
            }

        # Generate summary
        summary = generate_summary(chunk.text_content)

        if summary:
            chunk.text_summary = summary
            chunk.save(update_fields=['text_summary'])

            reduction = len(summary) / len(chunk.text_content)
            logger.info(
                f"Summarized chunk {chunk_id}: "
                f"{len(chunk.text_content)} -> {len(summary)} chars "
                f"({reduction*100:.1f}%)"
            )

            return {
                "status": "success",
                "chunk_id": str(chunk_id),
                "original_length": len(chunk.text_content),
                "summary_length": len(summary),
                "reduction_ratio": reduction
            }
        else:
            return {
                "status": "error",
                "chunk_id": str(chunk_id),
                "reason": "summarization_failed"
            }

    except TextChunk.DoesNotExist:
        return {
            "status": "error",
            "chunk_id": str(chunk_id),
            "reason": "chunk_not_found"
        }
    except Exception as e:
        logger.exception(f"Error summarizing chunk {chunk_id}: {e}")
        raise self.retry(exc=e, countdown=60)


@shared_task
def summarize_all_chunks(document_id: str = None, batch_size: int = 50) -> dict:
    """
    Queue summarization tasks for all chunks that need summaries.

    Args:
        document_id: Optional - limit to chunks from this document
        batch_size: Number of chunks to process in this batch

    Returns:
        {"queued": int, "skipped": int, "total_eligible": int}
    """
    # Find chunks that need summaries
    queryset = TextChunk.objects.filter(
        text_summary__isnull=True
    ).exclude(
        text_content__regex=r'^.{0,999}$'  # Exclude chunks under 1000 chars
    )

    if document_id:
        queryset = queryset.filter(document_id=document_id)

    # Only process narrative-type chunks that would be returned by search
    biographical_types = ['individual_entry', 'biographical_text', 'narrative_context']
    queryset = queryset.filter(chunk_type__in=biographical_types)

    total_eligible = queryset.count()

    # Queue batch of tasks
    chunk_ids = list(queryset.values_list('id', flat=True)[:batch_size])

    queued = 0
    for chunk_id in chunk_ids:
        summarize_chunk.delay(str(chunk_id))
        queued += 1

    logger.info(f"Queued {queued} summarization tasks ({total_eligible} total eligible)")

    return {
        "queued": queued,
        "total_eligible": total_eligible,
        "document_id": document_id
    }
