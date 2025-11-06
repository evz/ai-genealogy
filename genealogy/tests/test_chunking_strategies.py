"""Tests for chunking strategies"""
from django.test import TestCase

from genealogy.chunking.models import ChunkType
from genealogy.chunking_strategies.descendant_genealogy import (
    DescendantGenealogyChunkingStrategy,
)
from genealogy.models import Document

from .helpers import create_ocr_text


class TestDescendantGenealogyChunkingStrategy(TestCase):
    """Test the descendant genealogy chunking strategy"""

    def setUp(self):
        """Create test document"""
        self.document = Document.objects.create(
            title="Test Genealogy",
            languages="nld",
        )
        self.strategy = DescendantGenealogyChunkingStrategy()

    def test_chunks_simple_text(self):
        """Should chunk basic text content"""
        section_text = create_ocr_text([
            {'content': 'TWEEDE GENERATIE', 'element_type': 'sub_title'},
            {'content': 'II.1. Kinderen van Jan Jansen en Maria Pietersen:', 'element_type': 'sub_title'},
            {'content': 'a. Pieter Jansen, geboren 1850.', 'element_type': 'text'},
        ])

        chunks = self.strategy.chunk_section(section_text, self.document, [])

        # Should create chunks for: generation header, family group header, individual entry
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_type, ChunkType.GENERATION_HEADER)
        self.assertEqual(chunks[1].chunk_type, ChunkType.FAMILY_GROUP_HEADER)
        self.assertEqual(chunks[2].chunk_type, ChunkType.INDIVIDUAL_ENTRY)

    def test_extracts_images_from_flow(self):
        """Should extract image tokens into separate chunks"""
        section_text = create_ocr_text([
            {'content': 'TWEEDE GENERATIE', 'element_type': 'sub_title'},
            {'content': '[Image of family portrait]', 'element_type': 'image', 'bbox': (100, 150, 400, 350)},
            {'content': 'Afbeelding 1: Familie portret', 'element_type': 'image_caption', 'bbox': (100, 360, 400, 380)},
            {'content': 'a. Pieter Jansen, geboren 1850.', 'element_type': 'text'},
        ])

        chunks = self.strategy.chunk_section(section_text, self.document, [])

        # Find image chunks
        image_chunks = [c for c in chunks if c.chunk_type in [ChunkType.IMAGE, ChunkType.IMAGE_CAPTION]]
        self.assertEqual(len(image_chunks), 2)

        # Verify image chunk
        self.assertEqual(image_chunks[0].chunk_type, ChunkType.IMAGE)
        self.assertIn("Image of family portrait", image_chunks[0].content)

        # Verify image caption chunk
        self.assertEqual(image_chunks[1].chunk_type, ChunkType.IMAGE_CAPTION)
        self.assertIn("Familie portret", image_chunks[1].content)

    def test_extracts_info_boxes_from_flow(self):
        """Should extract inverted (info box) tokens"""
        section_text = create_ocr_text([
            {'content': 'TWEEDE GENERATIE', 'element_type': 'sub_title'},
            {'content': 'Let op: Deze familie verhuisde naar Amerika.', 'element_type': 'text', 'is_inverted': True},
            {'content': 'Zie ook pagina 45 voor meer details.', 'element_type': 'text', 'is_inverted': True},
            {'content': 'a. Pieter Jansen, geboren 1850.', 'element_type': 'text'},
        ])

        chunks = self.strategy.chunk_section(section_text, self.document, [])

        # Find info box chunks
        info_box_chunks = [c for c in chunks if c.chunk_type == ChunkType.INFO_BOX]
        self.assertEqual(len(info_box_chunks), 1)

        # Should contain both inverted tokens
        self.assertIn("Deze familie verhuisde naar Amerika", info_box_chunks[0].content)
        self.assertIn("Zie ook pagina 45", info_box_chunks[0].content)
        self.assertTrue(info_box_chunks[0].is_info_box)

    def test_groups_consecutive_info_boxes(self):
        """Should group consecutive inverted tokens, separate non-consecutive ones"""
        section_text = create_ocr_text([
            {'content': 'First info box line 1', 'element_type': 'text', 'is_inverted': True},
            {'content': 'First info box line 2', 'element_type': 'text', 'is_inverted': True},
            {'content': 'Regular text separator', 'element_type': 'text'},
            {'content': 'Second info box', 'element_type': 'text', 'is_inverted': True},
        ])

        chunks = self.strategy.chunk_section(section_text, self.document, [])

        # Find info box chunks
        info_box_chunks = [c for c in chunks if c.chunk_type == ChunkType.INFO_BOX]
        self.assertEqual(len(info_box_chunks), 2)

        # First info box should have both lines
        self.assertIn("First info box line 1", info_box_chunks[0].content)
        self.assertIn("First info box line 2", info_box_chunks[0].content)

        # Second info box is separate
        self.assertIn("Second info box", info_box_chunks[1].content)
        self.assertNotIn("First", info_box_chunks[1].content)

    def test_maintains_genealogical_context(self):
        """Should track generation and family group context across chunks"""
        section_text = create_ocr_text([
            {'content': 'TWEEDE GENERATIE', 'element_type': 'sub_title'},
            {'content': 'II.1. Kinderen van Jan Jansen en Maria Pietersen:', 'element_type': 'sub_title'},
            {'content': 'a. Pieter Jansen, geboren 1850.', 'element_type': 'text'},
            {'content': 'b. Klaas Jansen, geboren 1852.', 'element_type': 'text'},
        ])

        chunks = self.strategy.chunk_section(section_text, self.document, [])

        # Strategy should track context
        self.assertEqual(self.strategy.current_generation, "TWEEDE GENERATIE")
        self.assertEqual(self.strategy.current_family_group_id, "II.1")

        # Both individual entries should have the same context
        individual_chunks = [c for c in chunks if c.chunk_type == ChunkType.INDIVIDUAL_ENTRY]
        self.assertEqual(len(individual_chunks), 2)

        self.assertEqual(individual_chunks[0].generation, "TWEEDE GENERATIE")
        self.assertEqual(individual_chunks[0].family_group_id, "II.1")

        self.assertEqual(individual_chunks[1].generation, "TWEEDE GENERATIE")
        self.assertEqual(individual_chunks[1].family_group_id, "II.1")

    def test_handles_inverted_subtitle(self):
        """Should handle inverted sub_title elements as info boxes"""
        section_text = create_ocr_text([
            {'content': 'TWEEDE GENERATIE', 'element_type': 'sub_title'},
            {'content': 'Belangrijk: Familie afstamming onduidelijk', 'element_type': 'sub_title', 'is_inverted': True},
            {'content': 'a. Pieter Jansen, geboren 1850.', 'element_type': 'text'},
        ])

        chunks = self.strategy.chunk_section(section_text, self.document, [])

        # Should create info box for inverted subtitle
        info_box_chunks = [c for c in chunks if c.chunk_type == ChunkType.INFO_BOX]
        self.assertEqual(len(info_box_chunks), 1)
        self.assertIn("Familie afstamming onduidelijk", info_box_chunks[0].content)

    def test_empty_section(self):
        """Should handle empty section gracefully"""
        section_text = ""

        chunks = self.strategy.chunk_section(section_text, self.document, [])

        self.assertEqual(len(chunks), 0)

    def test_only_images(self):
        """Should handle section with only images"""
        section_text = create_ocr_text([
            {'content': '[Image 1]', 'element_type': 'image'},
            {'content': '[Image 2]', 'element_type': 'image'},
        ])

        chunks = self.strategy.chunk_section(section_text, self.document, [])

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_type, ChunkType.IMAGE)
        self.assertEqual(chunks[1].chunk_type, ChunkType.IMAGE)

    def test_only_info_boxes(self):
        """Should handle section with only info boxes"""
        section_text = create_ocr_text([
            {'content': 'Info box 1', 'element_type': 'text', 'is_inverted': True},
            {'content': 'Info box 2', 'element_type': 'text', 'is_inverted': True},
        ])

        chunks = self.strategy.chunk_section(section_text, self.document, [])

        # Should group consecutive info boxes
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, ChunkType.INFO_BOX)
        self.assertIn("Info box 1", chunks[0].content)
        self.assertIn("Info box 2", chunks[0].content)
