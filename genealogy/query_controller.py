"""
Intelligent query controller: RAG-first with long-context fallback.
Chooses the optimal approach based on query type and retrieval results.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from .ollama_utils import OllamaClient

logger = logging.getLogger(__name__)


class QueryController:
    """Smart controller for genealogy queries using RAG-first, long-context fallback strategy"""

    def __init__(self):
        self.ollama = OllamaClient()

        # Model configuration based on your analysis
        self.models = {
            'aya_35b': {
                'name': 'aya:35b-23',
                'context': 32768,  # Push aya beyond default 8K
                'strength': 'Dutch accuracy, entity extraction',
                'use_for': 'rag_answers'
            },
            'llama_8b': {
                'name': 'llama3.1:8b',
                'context': 65536,
                'strength': 'Long context workhorse',
                'use_for': 'long_context'
            },
            'qwen_7b': {
                'name': 'qwen2.5:7b',
                'context': 65536,
                'strength': 'Instruction following, JSON',
                'use_for': 'structured_output'
            },
            'mistral_12b': {
                'name': 'mistral-nemo:12b',
                'context': 65536,
                'strength': 'Complex reasoning',
                'use_for': 'relationship_analysis'
            }
        }

        # Thresholds for decision making
        self.config = {
            'min_rag_passages': 3,  # Minimum passages needed for RAG approach
            'max_rag_context': 8000,  # Max chars for RAG context (fits in aya:35b-23)
            'long_context_threshold': 32000,  # When to use long context models
            'fallback_timeout': 120,  # Seconds before timing out
        }

    def query(self, question: str, language: str = 'auto') -> Dict:
        """
        Main query interface - tries RAG first, falls back to long context.

        Args:
            question: User's genealogy question
            language: 'en', 'nl', or 'auto' for detection

        Returns:
            Dict with answer, method used, context size, model, etc.
        """
        start_time = time.time()

        # Detect language if auto
        if language == 'auto':
            language = self._detect_language(question)

        # Step 1: Try RAG retrieval
        rag_passages = self._retrieve_rag_passages(question)

        if len(rag_passages) >= self.config['min_rag_passages']:
            # RAG has enough context - use aya:35b-23 for accuracy
            return self._answer_with_rag(question, rag_passages, language, start_time)
        else:
            # RAG insufficient - fall back to long context
            return self._answer_with_long_context(question, language, start_time)

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        dutch_indicators = ['van', 'de', 'der', 'den', 'kinderen', 'geboren', 'overleden']
        english_indicators = ['the', 'and', 'of', 'children', 'born', 'died']

        text_lower = text.lower()
        dutch_count = sum(1 for word in dutch_indicators if word in text_lower)
        english_count = sum(1 for word in english_indicators if word in text_lower)

        return 'nl' if dutch_count > english_count else 'en'

    def _retrieve_rag_passages(self, question: str) -> List[Dict]:
        """
        Retrieve relevant passages using hybrid RAG approach.
        This would integrate with your existing RAG/RRF system.
        """
        # TODO: Integrate with your hybrid retrieval system
        # For now, return mock data to test the controller logic

        # This would call your hybrid retrieval:
        # - Vector search with embeddings
        # - Trigram fuzzy matching
        # - Phonetic Daitch-Mokotoff matching
        # - RRF fusion of results

        mock_passages = [
            {
                'text': 'II.1.a Aart van Santen, * ca. 1699, ~ 15.2.1733 Haaften...',
                'score': 0.95,
                'doc_id': 'chap03',
                'chunk_no': 7,
                'genealogy_ids': ['II.1.a'],
                'anchor': 'BK:derde-generatie:p003:7'
            }
        ]

        logger.info(f"RAG retrieval found {len(mock_passages)} passages for: {question[:50]}...")
        return mock_passages

    def _answer_with_rag(self, question: str, passages: List[Dict], language: str, start_time: float) -> Dict:
        """Answer using RAG with aya:35b-23 for Dutch accuracy"""

        # Build context from passages with anchors
        context_parts = []
        for passage in passages:
            anchor = passage.get('anchor', 'Unknown')
            text = passage['text']
            context_parts.append(f"[{anchor}]\n{text}")

        context = '\n\n'.join(context_parts)

        # Language-specific instructions
        lang_instruction = {
            'nl': "Antwoord in het Nederlands. Citeer Nederlandse namen exact.",
            'en': "Answer in English. Quote Dutch names verbatim."
        }.get(language, "Answer in English. Quote Dutch names verbatim.")

        prompt = f"""{lang_instruction}

Based on the genealogical text below, answer the question precisely.

GENEALOGICAL SOURCES:
{context}

QUESTION: {question}

ANSWER:"""

        # Use aya:35b-23 for Dutch accuracy
        model_config = self.models['aya_35b']
        response = self.ollama.generate(
            model=model_config['name'],
            prompt=prompt,
            options={
                'num_ctx': model_config['context'],
                'temperature': 0.1
            }
        )

        return {
            'answer': response,
            'method': 'rag',
            'model': model_config['name'],
            'context_size': len(context),
            'passages_used': len(passages),
            'language': language,
            'processing_time': time.time() - start_time,
            'anchors': [p.get('anchor') for p in passages]
        }

    def _answer_with_long_context(self, question: str, language: str, start_time: float) -> Dict:
        """Fallback to long context model when RAG is insufficient"""

        # Get large document chunk
        document_text = self._get_document_context()

        # Choose model based on context size
        if len(document_text) > self.config['long_context_threshold']:
            model_config = self.models['llama_8b']  # 65K context
        else:
            model_config = self.models['qwen_7b']   # Also 65K but smaller model

        # Language-specific instructions
        lang_instruction = {
            'nl': "Antwoord in het Nederlands op basis van het document.",
            'en': "Answer in English based on the document."
        }.get(language, "Answer in English based on the document.")

        prompt = f"""{lang_instruction}

DOCUMENT:
{document_text}

QUESTION: {question}

ANSWER:"""

        response = self.ollama.generate(
            model=model_config['name'],
            prompt=prompt,
            options={
                'num_ctx': model_config['context'],
                'temperature': 0.2
            }
        )

        return {
            'answer': response,
            'method': 'long_context',
            'model': model_config['name'],
            'context_size': len(document_text),
            'language': language,
            'processing_time': time.time() - start_time,
            'reason': 'insufficient_rag_passages'
        }

    def _get_document_context(self) -> str:
        """Get document text for long context processing"""
        # TODO: Integrate with your Document model
        # This should get the full document or relevant sections

        # For now return a placeholder
        return "Document context would be loaded here..."

    def analyze_query_requirements(self, question: str) -> Dict:
        """Analyze what kind of query this is and recommend approach"""

        question_lower = question.lower()

        # Query type classification
        query_types = []
        if any(word in question_lower for word in ['relationship', 'related', 'family tree']):
            query_types.append('relationship')
        if any(word in question_lower for word in ['born', 'birth', 'died', 'death', 'date']):
            query_types.append('biographical')
        if any(word in question_lower for word in ['summarize', 'overview', 'generation']):
            query_types.append('summary')
        if any(word in question_lower for word in ['where', 'place', 'location']):
            query_types.append('geographical')

        # Recommend approach
        if 'summary' in query_types or 'relationship' in query_types:
            recommended = 'long_context'
            reason = 'Requires broad context or multi-hop reasoning'
        else:
            recommended = 'rag'
            reason = 'Specific factual query suitable for RAG'

        return {
            'query_types': query_types,
            'recommended_approach': recommended,
            'reason': reason,
            'detected_language': self._detect_language(question)
        }


def create_query_controller() -> QueryController:
    """Factory function for creating query controller"""
    return QueryController()
