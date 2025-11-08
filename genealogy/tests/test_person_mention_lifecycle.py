"""Tests for PersonMention lifecycle management

Tests that PersonMentions created during chunking (with genealogical_id) are:
- Created during the chunking process
- Preserved when running create_entities --clean
- Deleted when rechunking happens
"""

import pytest
from django.core.management import call_command

from genealogy.models import (BookSection, Document, DocumentPage,
                              PersonMention, TextChunk)
from genealogy.tasks.chunking import create_document_chunks


@pytest.mark.django_db
class TestPersonMentionLifecycle:
    """Test PersonMention lifecycle during chunking and entity creation"""

    @pytest.fixture
    def sample_document_with_ocr(self):
        """Create a document with OCR text ready for chunking"""
        document = Document.objects.create(
            title="Test Genealogy Book",
            ocr_completed=True
        )

        # Create a book section
        section = BookSection.objects.create(
            document=document,
            title="Descendant Genealogy",
            section_type="DESCENDANT_GENEALOGY",
            start_page=1,
            end_page=1
        )

        # Create a page with OCR text containing individual entries
        page = DocumentPage.objects.create(
            document=document,
            page_number=1,
            ocr_completed=True,
            ocr_text="""<|ref|>sub_title<|/ref|><|det|>[[100,100,400,120]]<|/det|>
Tweede generatie

<|ref|>sub_title<|/ref|><|det|>[[100,150,600,170]]<|/det|>
II.3. Kinderen van Jan van Zanten en Maria Pieterse:

<|ref|>text<|/ref|><|det|>[[100,200,700,250]]<|/det|>
a. Thomas van Zanten, * Amsterdam 1.1.1850, † Rotterdam 15.3.1920

<|ref|>text<|/ref|><|det|>[[100,260,700,310]]<|/det|>
b. Pieter van Zanten, * Amsterdam 5.6.1852
"""
        )

        return document

    def test_chunking_creates_person_mentions(self, sample_document_with_ocr):
        """
        Test that running the chunking task creates PersonMentions
        with genealogical_id for individual entries
        """
        document = sample_document_with_ocr

        # Initially no chunks or PersonMentions
        assert TextChunk.objects.count() == 0
        assert PersonMention.objects.count() == 0

        # Run the chunking task
        result = create_document_chunks(str(document.id))

        # Verify chunking succeeded
        assert result['success'] is True
        assert result['chunks_created'] > 0

        # Verify PersonMentions were created
        person_mentions = PersonMention.objects.filter(genealogical_id__isnull=False)
        assert person_mentions.count() == 2  # Thomas and Pieter

        # Verify they have the correct genealogical_id
        thomas = PersonMention.objects.get(genealogical_id="II.3.a")
        assert thomas.given_names == "Thomas"
        assert thomas.surname == "van Zanten"
        assert thomas.generation == 2

        pieter = PersonMention.objects.get(genealogical_id="II.3.b")
        assert pieter.given_names == "Pieter"
        assert pieter.surname == "van Zanten"

        # Verify they're linked to chunks
        thomas_chunk = TextChunk.objects.get(primary_person_mention=thomas)
        assert thomas_chunk.subject == "Thomas van Zanten"
        assert thomas_chunk.genealogical_identifier == "II.3.a"

    def test_rechunking_deletes_and_recreates_person_mentions(self, sample_document_with_ocr):
        """
        Test that rechunking deletes old PersonMentions and creates new ones
        """
        document = sample_document_with_ocr

        # First chunking
        result = create_document_chunks(str(document.id))
        assert result['success'] is True

        # Get the PersonMentions from first chunking
        first_thomas_id = PersonMention.objects.get(genealogical_id="II.3.a").id

        # Run chunking again (rechunking)
        result = create_document_chunks(str(document.id))
        assert result['success'] is True

        # Old PersonMention should be deleted (by ID)
        assert not PersonMention.objects.filter(id=first_thomas_id).exists()

        # New PersonMention should be created with same genealogical_id
        new_thomas = PersonMention.objects.get(genealogical_id="II.3.a")
        assert new_thomas.id != first_thomas_id  # Different instance
        assert new_thomas.given_names == "Thomas"
        assert new_thomas.surname == "van Zanten"

    def test_create_entities_clean_preserves_chunking_person_mentions(self, sample_document_with_ocr):
        """
        Test that create_entities --clean preserves PersonMentions created during chunking
        """
        document = sample_document_with_ocr

        # Run chunking to create PersonMentions with genealogical_id
        result = create_document_chunks(str(document.id))
        assert result['success'] is True

        # Verify PersonMentions exist
        chunking_person_count = PersonMention.objects.filter(genealogical_id__isnull=False).count()
        assert chunking_person_count == 2

        # Get the specific PersonMention IDs
        thomas = PersonMention.objects.get(genealogical_id="II.3.a")
        thomas_id = thomas.id

        # Mark chunks as entities_extracted so create_entities will process them
        TextChunk.objects.filter(document=document).update(entities_extracted=True)

        # Create a PersonMention WITHOUT genealogical_id (simulating create_entities output)
        extra_person = PersonMention.objects.create(
            given_names="Jan",
            surname="Pieterse",
            generation=2,
            genealogical_id=None,
        )
        extra_person.source_documents.add(document)

        # Now we have 3 PersonMentions: 2 from chunking, 1 from entities
        assert PersonMention.objects.count() == 3

        # Run create_entities --clean (in actual mode, not dry-run)
        # This should delete PersonMentions without genealogical_id
        # but preserve those with genealogical_id
        call_command('create_entities', document_id=str(document.id), clean=True)

        # PersonMentions from chunking should still exist
        assert PersonMention.objects.filter(genealogical_id__isnull=False).count() == 2
        assert PersonMention.objects.filter(id=thomas_id).exists()

        # The extra PersonMention without genealogical_id should have been deleted
        # (along with any new ones created by create_entities)
        assert not PersonMention.objects.filter(id=extra_person.id).exists()
