"""
Management command to test hybrid RAG+RRF retrieval and answer genealogical questions.
"""

import json
import logging

from django.core.management.base import BaseCommand

from genealogy.ollama_utils import OllamaClient
from genealogy.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Query the genealogy database using hybrid RAG+RRF retrieval"

    def add_arguments(self, parser):
        parser.add_argument(
            "question",
            type=str,
            help="The genealogical question to answer",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=5,
            help="Number of final results to retrieve (default: 5)",
        )
        parser.add_argument(
            "--expand-window",
            type=int,
            default=1,
            help="Number of chunks to expand before/after each match (default: 1)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="aya:35b-23",
            help="LLM model to use for answering (default: aya:35b-23)",
        )
        parser.add_argument(
            "--context-only",
            action="store_true",
            help="Only show retrieved context, don't generate answer",
        )
        parser.add_argument(
            "--show-scores",
            action="store_true",
            help="Show RRF scores for retrieved chunks",
        )
        parser.add_argument(
            "--language",
            type=str,
            choices=["en", "nl", "auto"],
            default="auto",
            help="Response language (default: auto-detect)",
        )

    def handle(self, *args, **options):
        question = options["question"]
        top_k = options["top_k"]
        expand_window = options["expand_window"]
        model = options["model"]
        context_only = options["context_only"]
        show_scores = options["show_scores"]
        language = options["language"]

        self.stdout.write(f"\n{'='*80}")
        self.stdout.write(f"QUESTION: {question}")
        self.stdout.write(f"{'='*80}\n")

        # Initialize retriever
        retriever = HybridRetriever()

        # Retrieve relevant chunks
        self.stdout.write("🔍 Searching using hybrid RAG+RRF retrieval...\n")
        chunks = retriever.retrieve(
            query=question,
            top_k=top_k,
            expand_window=expand_window,
        )

        if not chunks:
            self.stdout.write(self.style.WARNING("⚠️  No relevant chunks found"))
            return

        self.stdout.write(f"📚 Retrieved {len(chunks)} chunks\n")

        # Display retrieved context
        if show_scores:
            self.stdout.write("\nRETRIEVED CHUNKS (with RRF scores):")
            self.stdout.write("-" * 80)
            for chunk in chunks:
                score = chunk.get("rrf_score", 0.0)
                is_center = chunk.get("is_center", True)
                marker = "★" if is_center else " "
                self.stdout.write(
                    f"\n{marker} Score: {score:.4f} | "
                    f"Seq: {chunk['sequence_number']} | "
                    f"Page: {chunk['start_page']}-{chunk['end_page']} | "
                    f"Type: {chunk['chunk_type']}"
                )
                if chunk.get("genealogy_ids"):
                    self.stdout.write(f"  IDs: {chunk['genealogy_ids']}")
                if chunk.get("person_names"):
                    self.stdout.write(f"  Names: {chunk['person_names'][:5]}")
                self.stdout.write(f"\n  Text: {chunk['text_content'][:200]}...")
                self.stdout.write("-" * 80)

        # Build context for LLM
        context = retriever.build_context(chunks, include_anchors=True)

        if context_only:
            self.stdout.write("\nFULL CONTEXT:")
            self.stdout.write("=" * 80)
            self.stdout.write(context)
            self.stdout.write("=" * 80)
            return

        # Detect language if auto
        if language == "auto":
            language = self._detect_language(question)

        # Generate answer using LLM
        self.stdout.write(f"\n💬 Generating answer using {model}...\n")
        answer = self._generate_answer(question, context, model, language)

        # Display answer
        self.stdout.write("\nANSWER:")
        self.stdout.write("=" * 80)
        self.stdout.write(answer)
        self.stdout.write("=" * 80)
        self.stdout.write("")

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        dutch_indicators = ['van', 'de', 'der', 'den', 'kinderen', 'geboren', 'overleden', 'wie', 'wanneer', 'waar']
        english_indicators = ['the', 'and', 'of', 'children', 'born', 'died', 'who', 'when', 'where']

        text_lower = text.lower()
        dutch_count = sum(1 for word in dutch_indicators if word in text_lower)
        english_count = sum(1 for word in english_indicators if word in text_lower)

        detected = 'nl' if dutch_count > english_count else 'en'
        self.stdout.write(f"🌍 Detected language: {'Dutch' if detected == 'nl' else 'English'}")
        return detected

    def _generate_answer(self, question: str, context: str, model: str, language: str) -> str:
        """Generate answer using LLM"""
        # Language-specific instructions
        if language == 'nl':
            lang_instruction = """Je MOET in het Nederlands antwoorden. Citeer Nederlandse namen exact zoals ze in de bron staan.

Als er meerdere personen met dezelfde naam zijn in de passages:
1. Begin met: "Er zijn meerdere personen met deze naam in de bron:"
2. Lijst elke persoon AFZONDERLIJK met hun onderscheidende details
3. Gebruik de labels tussen haakjes om ze te onderscheiden
4. Combineer NOOIT informatie van verschillende personen"""
        else:
            lang_instruction = """You MUST answer in English. The source text is in Dutch - translate your answer to English, but quote Dutch names verbatim as they appear in the source.

If there are multiple people with the same name in the passages:
1. Start with: "There are multiple people with this name in the sources:"
2. List each person SEPARATELY with their distinguishing details
3. Use the bracketed labels to distinguish them
4. NEVER combine information from different people"""

        prompt = f"""{lang_instruction}

CRITICAL INSTRUCTIONS:
- Answer based ONLY on the information explicitly stated in the passages below
- DO NOT use any external knowledge or make assumptions
- DO NOT infer information that is not directly stated
- If the passages don't contain enough information to answer, say so
- The bracketed labels indicate different genealogical entries
- People with the same name AND same birth date/parents are the SAME person - combine their information
- People with the same name but DIFFERENT birth dates/parents are DIFFERENT people - list separately

DUTCH GENEALOGICAL ABBREVIATIONS:
- * or geb. = geboren (born)
- ~ or ged. = gedoopt (baptized)
- † or overl. = overleden (died)
- begr. = begraven (buried)
- x or tr. or ondertr. = getrouwd/ondertrouwd (married)
- wednr. or wedn. = weduwnaar (widower)
- wed. = weduwe (widow)
- dv. = dochter van (daughter of)
- zv. = zoon van (son of)
- geh. = gehuwd (married to)

GENEALOGICAL SOURCES:
{context}

QUESTION: {question}

Return your answer as JSON in this format:
{{
  "people": [
    {{
      "anchor": "the bracketed label from the source",
      "summary": "one-sentence summary with birth/death dates, location, parents, occupation, etc."
    }}
  ]
}}

If there is only one person, the "people" array will have one entry. If there are multiple people with the same name, list each separately.

JSON (in {'Dutch' if language == 'nl' else 'English'}):"""

        ollama = OllamaClient(timeout=120)
        response = ollama.generate(
            model=model,
            prompt=prompt,
            format='json',  # Force JSON output for better structure
            options={
                'num_ctx': 32768,  # Large context for multi-chunk retrieval
                'temperature': 0.1,
            }
        )

        if not response:
            return "❌ Failed to generate answer"

        # Parse and format the JSON response
        try:
            data = json.loads(response)

            # Format the answer from JSON
            if isinstance(data, dict):
                if 'people' in data and isinstance(data['people'], list):
                    # Multiple people format
                    formatted = []
                    if len(data['people']) > 1:
                        formatted.append(f"There are {len(data['people'])} people with this name in the sources:\n")
                    for i, person in enumerate(data['people'], 1):
                        formatted.append(f"{i}. {person.get('summary', 'No information')}")
                        if person.get('anchor'):
                            formatted.append(f"   Source: {person['anchor']}")
                    return '\n'.join(formatted)
                elif 'answer' in data:
                    return data['answer']

            # Fallback to raw response if structure is unexpected
            return response
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw response
            return response
