"""Transcription and translation tasks for archival SourceImage records"""

import logging

from celery import shared_task
from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils import timezone

from ..loghi_client import LoghiClient
from ..models import SourceImage
from ..ollama_utils import OllamaClient

logger = logging.getLogger(__name__)

# TODO: tune this for deepseek-ocr's expected prompt conventions
OCR_PROMPT = "Free OCR."

TRANSLATION_PROMPT_HANDWRITTEN = """You are translating a Dutch handwritten-text-recognition (HTR) transcription \
of a 19th-century legal/court document into English. The source text may contain OCR/HTR errors \
(garbled words, misread letters) since it comes from automated transcription of old handwriting. \
Translate as faithfully as possible, and where a word or phrase is clearly garbled or nonsensical, \
keep the original Dutch word in [brackets] rather than guessing, so the reader can spot likely \
transcription errors. Output only the translation, no commentary.

Dutch text:
---
{text}
---

English translation:"""

# TODO: user will supply a printed/typed-specific variant; reuse the handwritten
# prompt as an interim default for the ollama_deepseek_ocr path until then.
TRANSLATION_PROMPT_PRINTED = TRANSLATION_PROMPT_HANDWRITTEN

TRANSLATION_PROMPTS = {
    "loghi": TRANSLATION_PROMPT_HANDWRITTEN,
    "ollama_deepseek_ocr": TRANSLATION_PROMPT_PRINTED,
}


@shared_task(bind=True)
def transcribe_source_image(self, source_image_id: str):  # noqa: ARG001
    """
    Transcribe a SourceImage using the pipeline appropriate to its is_handwritten flag:
    Loghi HTR for handwritten documents, Ollama deepseek-ocr for printed/typed documents.

    On success, enqueues translate_source_image for the same image.
    """
    try:
        image = SourceImage.objects.get(id=source_image_id)
    except SourceImage.DoesNotExist:
        error_msg = f"SourceImage with id {source_image_id} not found"
        logger.exception(error_msg)
        return {"success": False, "error": error_msg}
    except ValidationError:
        error_msg = f"Invalid UUID format: {source_image_id}"
        logger.exception(error_msg)
        return {"success": False, "error": error_msg}

    image.transcription_status = "processing"
    image.save(update_fields=["transcription_status"])

    try:
        if image.is_handwritten:
            text = LoghiClient().transcribe(image.image_file.path)
            method = "loghi"
        else:
            ollama = OllamaClient()
            text = ollama.generate_with_image(
                model=settings.OLLAMA_OCR_MODEL,
                prompt=OCR_PROMPT,
                image=image.image_file.path,
            )
            method = "ollama_deepseek_ocr"

        if not text:
            raise RuntimeError("Transcription returned no text")

        image.raw_transcription = text
        image.transcription_method = method
        image.transcription_status = "completed"
        image.transcription_error = ""
        image.transcribed_at = timezone.now()
        image.save(
            update_fields=[
                "raw_transcription",
                "transcription_method",
                "transcription_status",
                "transcription_error",
                "transcribed_at",
            ]
        )

        translate_source_image.delay(str(image.id))

        return {"success": True, "source_image_id": str(image.id), "method": method}

    except Exception as e:
        error_msg = f"Transcription failed for source image {source_image_id}: {e!s}"
        logger.error(error_msg, exc_info=True)
        image.transcription_status = "failed"
        image.transcription_error = str(e)
        image.save(update_fields=["transcription_status", "transcription_error"])
        return {"success": False, "error": error_msg, "source_image_id": str(image.id)}


@shared_task(bind=True)
def translate_source_image(self, source_image_id: str):  # noqa: ARG001
    """
    Translate a SourceImage's raw_transcription into English using aya-expanse,
    with a prompt chosen by the transcription method used.

    Requires transcription_status == "completed".
    """
    try:
        image = SourceImage.objects.get(id=source_image_id)
    except SourceImage.DoesNotExist:
        error_msg = f"SourceImage with id {source_image_id} not found"
        logger.exception(error_msg)
        return {"success": False, "error": error_msg}
    except ValidationError:
        error_msg = f"Invalid UUID format: {source_image_id}"
        logger.exception(error_msg)
        return {"success": False, "error": error_msg}

    if image.transcription_status != "completed":
        error_msg = (
            f"Cannot translate source image {source_image_id}: "
            f"transcription_status is {image.transcription_status!r}, not 'completed'"
        )
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    image.translation_status = "processing"
    image.save(update_fields=["translation_status"])

    try:
        prompt_template = TRANSLATION_PROMPTS[image.transcription_method]
        prompt = prompt_template.format(text=image.raw_transcription)

        ollama = OllamaClient()
        translation = ollama.generate(model=settings.OLLAMA_TRANSLATION_MODEL, prompt=prompt)

        if not translation:
            raise RuntimeError("Translation returned no text")

        image.translation = translation
        image.translation_status = "completed"
        image.translation_model = settings.OLLAMA_TRANSLATION_MODEL
        image.translation_error = ""
        image.translated_at = timezone.now()
        image.save(
            update_fields=[
                "translation",
                "translation_status",
                "translation_model",
                "translation_error",
                "translated_at",
            ]
        )

        return {"success": True, "source_image_id": str(image.id)}

    except Exception as e:
        error_msg = f"Translation failed for source image {source_image_id}: {e!s}"
        logger.error(error_msg, exc_info=True)
        image.translation_status = "failed"
        image.translation_error = str(e)
        image.save(update_fields=["translation_status", "translation_error"])
        return {"success": False, "error": error_msg, "source_image_id": str(image.id)}
