"""Tests for genealogy.chunking.persistence module

This module tests the persistence layer that saves chunks to the database.
"""

import pytest

from genealogy.chunking.models import (BoundingBox, ChunkType, GroundingToken,
                                       TextChunk)
from genealogy.chunking.persistence import save_chunks_to_db
from genealogy.models import Document, PersonMention
from genealogy.models import TextChunk as TextChunkModel


@pytest.mark.django_db
class TestSaveChunksToDb:
    """Test save_chunks_to_db function"""

    @pytest.fixture
    def mock_document(self):
        """Create a mock document for testing"""
        return Document.objects.create(
            title="Test Genealogy Document"
        )

    @pytest.fixture
    def page_map(self):
        """Create a simple page map"""
        return [
            {'page_number': 1, 'start_char': 0, 'end_char': 100},
            {'page_number': 2, 'start_char': 100, 'end_char': 200},
        ]

    def test_saves_subject_and_genealogical_identifier(self, mock_document, page_map):
        """Test that subject and genealogical_identifier are correctly saved for INDIVIDUAL_ENTRY chunks"""

        # Create an INDIVIDUAL_ENTRY chunk with all the necessary context
        chunks = [
            TextChunk(
                chunk_type=ChunkType.INDIVIDUAL_ENTRY,
                content="a. Pieter van Zanten, * Amsterdam 1.1.1850, † Rotterdam 15.3.1920",
                grounding_tokens=[
                    GroundingToken(
                        element_type='text',
                        bbox=BoundingBox(x1=100, y1=100, x2=400, y2=120),
                        content="a. Pieter van Zanten, * Amsterdam 1.1.1850",
                        raw_match="<|grounding|>text<|x1=100|><|y1=100|><|x2=400|><|y2=120|>",
                    )
                ],
                generation="Tweede generatie",
                family_group="II.3. Kinderen van Jan van Zanten en Maria Pieterse",
                family_group_id="II.3",
                parents=("Jan van Zanten", "Maria Pieterse"),
                individual_marker="a.",
                subject="Pieter van Zanten",  # This is set by the handler
                extracted_people=["Pieter van Zanten", "Jan van Zanten", "Maria Pieterse"],
            )
        ]

        book_text = "a. Pieter van Zanten, * Amsterdam 1.1.1850, † Rotterdam 15.3.1920"

        # Save chunks to database
        saved_chunks = save_chunks_to_db(
            chunks=chunks,
            document=mock_document,
            page_map=page_map,
            book_text=book_text,
            start_sequence=1
        )

        # Verify one chunk was saved
        assert len(saved_chunks) == 1
        db_chunk = saved_chunks[0]

        # Verify subject is saved correctly
        assert db_chunk.subject == "Pieter van Zanten"

        # Verify genealogical_identifier is built correctly (family_group_id.individual_marker)
        assert db_chunk.genealogical_identifier == "II.3.a"

        # Verify chunk type
        assert db_chunk.chunk_type == "individual_entry"

        # Verify other fields are still correct
        assert db_chunk.generation_number == 2
        assert "Pieter van Zanten" in db_chunk.extracted_people

    def test_genealogical_identifier_with_different_markers(self, mock_document, page_map):
        """Test genealogical_identifier with different individual markers"""

        chunks = [
            TextChunk(
                chunk_type=ChunkType.INDIVIDUAL_ENTRY,
                content="l. Lucas van Zanten, * 1855",
                grounding_tokens=[
                    GroundingToken(
                        element_type='text',
                        bbox=BoundingBox(x1=100, y1=100, x2=400, y2=120),
                        content="l. Lucas van Zanten, * 1855",
                        raw_match="<|grounding|>text<|x1=100|><|y1=100|><|x2=400|><|y2=120|>",
                    )
                ],
                generation="Vierde generatie",
                family_group_id="IV.12",
                individual_marker="l.",
                subject="Lucas van Zanten",
            )
        ]

        book_text = "l. Lucas van Zanten, * 1855"

        saved_chunks = save_chunks_to_db(
            chunks=chunks,
            document=mock_document,
            page_map=page_map,
            book_text=book_text,
            start_sequence=1
        )

        db_chunk = saved_chunks[0]
        assert db_chunk.genealogical_identifier == "IV.12.l"
        assert db_chunk.subject == "Lucas van Zanten"

    def test_no_genealogical_identifier_for_non_individual_entries(self, mock_document, page_map):
        """Test that genealogical_identifier is None for non-INDIVIDUAL_ENTRY chunks"""

        chunks = [
            TextChunk(
                chunk_type=ChunkType.GENERATION_HEADER,
                content="Tweede generatie",
                grounding_tokens=[
                    GroundingToken(
                        element_type='sub_title',
                        bbox=BoundingBox(x1=100, y1=100, x2=400, y2=120),
                        content="Tweede generatie",
                        raw_match="<|grounding|>sub_title<|x1=100|><|y1=100|><|x2=400|><|y2=120|>",
                    )
                ],
                generation="Tweede generatie",
            )
        ]

        book_text = "Tweede generatie"

        saved_chunks = save_chunks_to_db(
            chunks=chunks,
            document=mock_document,
            page_map=page_map,
            book_text=book_text,
            start_sequence=1
        )

        db_chunk = saved_chunks[0]
        assert db_chunk.genealogical_identifier is None
        assert db_chunk.subject is None

    def test_no_genealogical_identifier_without_family_group_id(self, mock_document, page_map):
        """Test that genealogical_identifier is None if family_group_id is missing"""

        chunks = [
            TextChunk(
                chunk_type=ChunkType.INDIVIDUAL_ENTRY,
                content="a. Pieter van Zanten, * 1850",
                grounding_tokens=[
                    GroundingToken(
                        element_type='text',
                        bbox=BoundingBox(x1=100, y1=100, x2=400, y2=120),
                        content="a. Pieter van Zanten, * 1850",
                        raw_match="<|grounding|>text<|x1=100|><|y1=100|><|x2=400|><|y2=120|>",
                    )
                ],
                generation="Tweede generatie",
                individual_marker="a.",
                subject="Pieter van Zanten",
                # Note: family_group_id is None
            )
        ]

        book_text = "a. Pieter van Zanten, * 1850"

        saved_chunks = save_chunks_to_db(
            chunks=chunks,
            document=mock_document,
            page_map=page_map,
            book_text=book_text,
            start_sequence=1
        )

        db_chunk = saved_chunks[0]
        # Should still save subject even without genealogical_identifier
        assert db_chunk.subject == "Pieter van Zanten"
        # But genealogical_identifier should be None
        assert db_chunk.genealogical_identifier is None

    def test_creates_person_mention_for_individual_entry(self, mock_document, page_map):
        """Test that PersonMention is created for individual entry chunks with subjects"""
        chunks = [
            TextChunk(
                chunk_type=ChunkType.INDIVIDUAL_ENTRY,
                content="a. Thomas van Zanten, * Amsterdam 1.1.1850",
                grounding_tokens=[
                    GroundingToken(
                        element_type='text',
                        bbox=BoundingBox(x1=100, y1=100, x2=400, y2=120),
                        content="a. Thomas van Zanten, * Amsterdam 1.1.1850",
                        raw_match="",
                    )
                ],
                generation="Tweede generatie",
                family_group_id="II.3",
                individual_marker="a.",
                subject="Thomas van Zanten",
                extracted_people=["Thomas van Zanten"],
            )
        ]

        book_text = "a. Thomas van Zanten, * Amsterdam 1.1.1850"

        saved_chunks = save_chunks_to_db(
            chunks=chunks,
            document=mock_document,
            page_map=page_map,
            book_text=book_text,
            start_sequence=1
        )

        db_chunk = saved_chunks[0]

        # Check that PersonMention was created
        assert db_chunk.primary_person_mention is not None
        person = db_chunk.primary_person_mention

        # Check PersonMention fields
        assert person.given_names == "Thomas"
        assert person.surname == "van Zanten"  # Dutch surname with prefix
        assert person.genealogical_id == "II.3.a"
        assert person.generation == 2  # "Tweede generatie" -> 2

        # Check relationships
        assert mock_document in person.source_documents.all()
        assert db_chunk in person.source_chunks.all()

    def test_no_person_mention_for_non_individual_entry(self, mock_document, page_map):
        """Test that PersonMention is NOT created for non-individual-entry chunks"""
        chunks = [
            TextChunk(
                chunk_type=ChunkType.GENERATION_HEADER,
                content="Tweede generatie",
                grounding_tokens=[
                    GroundingToken(
                        element_type='sub_title',
                        bbox=BoundingBox(x1=100, y1=100, x2=400, y2=120),
                        content="Tweede generatie",
                        raw_match="",
                    )
                ],
                generation="Tweede generatie",
            )
        ]

        book_text = "Tweede generatie"

        saved_chunks = save_chunks_to_db(
            chunks=chunks,
            document=mock_document,
            page_map=page_map,
            book_text=book_text,
            start_sequence=1
        )

        db_chunk = saved_chunks[0]
        # Should NOT have created a PersonMention
        assert db_chunk.primary_person_mention is None

    def test_dutch_surname_parsing_in_person_mention(self, mock_document, page_map):
        """Test that Dutch surnames with prefixes are correctly parsed in PersonMention"""
        test_cases = [
            ("Pieter van Zanten", "Pieter", "van Zanten"),
            ("Jan van der Meer", "Jan", "van der Meer"),
            ("Maria van den Berg", "Maria", "van den Berg"),
            ("Hendrik de Jong", "Hendrik", "de Jong"),
        ]

        for i, (full_name, expected_given, expected_surname) in enumerate(test_cases):
            chunks = [
                TextChunk(
                    chunk_type=ChunkType.INDIVIDUAL_ENTRY,
                    content=f"{chr(97+i)}. {full_name}, * 1850",
                    grounding_tokens=[
                        GroundingToken(
                            element_type='text',
                            bbox=BoundingBox(x1=100, y1=100, x2=400, y2=120),
                            content=f"{chr(97+i)}. {full_name}, * 1850",
                            raw_match="",
                        )
                    ],
                    generation="Tweede generatie",
                    family_group_id="II.3",
                    individual_marker=f"{chr(97+i)}.",
                    subject=full_name,
                    extracted_people=[full_name],
                )
            ]

            book_text = f"{chr(97+i)}. {full_name}, * 1850"

            saved_chunks = save_chunks_to_db(
                chunks=chunks,
                document=mock_document,
                page_map=page_map,
                book_text=book_text,
                start_sequence=i+1
            )

            db_chunk = saved_chunks[0]
            person = db_chunk.primary_person_mention

            assert person is not None, f"PersonMention not created for {full_name}"
            assert person.given_names == expected_given, f"Failed for {full_name}"
            assert person.surname == expected_surname, f"Failed for {full_name}"

    def test_no_person_mention_without_subject(self, mock_document, page_map):
        """Test that PersonMention is NOT created if subject is missing"""
        chunks = [
            TextChunk(
                chunk_type=ChunkType.INDIVIDUAL_ENTRY,
                content="a. Some text...",
                grounding_tokens=[
                    GroundingToken(
                        element_type='text',
                        bbox=BoundingBox(x1=100, y1=100, x2=400, y2=120),
                        content="a. Some text...",
                        raw_match="",
                    )
                ],
                generation="Tweede generatie",
                family_group_id="II.3",
                individual_marker="a.",
                subject=None,  # No subject!
            )
        ]

        book_text = "a. Some text..."

        saved_chunks = save_chunks_to_db(
            chunks=chunks,
            document=mock_document,
            page_map=page_map,
            book_text=book_text,
            start_sequence=1
        )

        db_chunk = saved_chunks[0]
        # Should NOT have created a PersonMention
        assert db_chunk.primary_person_mention is None
