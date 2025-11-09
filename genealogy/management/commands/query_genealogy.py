"""
Management command to test hybrid RAG+RRF retrieval and answer genealogical questions.
"""

import logging

from django.core.management.base import BaseCommand

from genealogy.ollama_utils import OllamaClient
from genealogy.retrieval import HybridRetriever
from genealogy.services.agent_executor import AgentExecutor

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
            default="llama3.1:70b",
            help="LLM model to use for answering (default: llama3.1:70b)",
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
        parser.add_argument(
            "--show-enrichment",
            action="store_true",
            help="Show extracted people, events, and relationships in context",
        )
        parser.add_argument(
            "--agent",
            action="store_true",
            help="Use agentic workflow with iterative tool calling for complex queries",
        )

    def handle(self, *args, **options):
        question = options["question"]
        top_k = options["top_k"]
        expand_window = options["expand_window"]
        model = options["model"]
        context_only = options["context_only"]
        show_scores = options["show_scores"]
        language = options["language"]
        show_enrichment = options["show_enrichment"]
        use_agent = options["agent"]

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
                if chunk.get("genealogical_identifier"):
                    self.stdout.write(f"  ID: {chunk['genealogical_identifier']}")
                if chunk.get("subject"):
                    self.stdout.write(f"  Subject: {chunk['subject']}")
                if chunk.get("extracted_people"):
                    names = chunk['extracted_people'][:5]
                    self.stdout.write(f"  People: {', '.join(names)}")
                self.stdout.write(f"\n  Text: {chunk['text_content'][:200]}...")
                self.stdout.write("-" * 80)

        # Build context for LLM
        context = retriever.build_context(
            chunks,
            include_anchors=True,
            include_enrichment=show_enrichment
        )

        if context_only:
            self.stdout.write("\nFULL CONTEXT:")
            self.stdout.write("=" * 80)
            self.stdout.write(context)
            self.stdout.write("=" * 80)
            return

        # Detect language if auto
        if language == "auto":
            language = self._detect_language(question, silent=False)

        # Use agentic workflow if requested
        if use_agent:
            self.stdout.write(f"\n🤖 Using agentic workflow with {model}...\n")

            # Initialize agent
            agent = AgentExecutor(model=model, max_iterations=10, timeout=300)

            # Execute with initial context from RAG
            result = agent.execute(user_query=question, initial_context=context)

            # Display tool calls
            if result["tool_calls"]:
                self.stdout.write("\n🔧 TOOL CALLS:")
                self.stdout.write("-" * 80)
                for i, call in enumerate(result["tool_calls"], 1):
                    self.stdout.write(f"\n{i}. {call['tool']}({call['arguments']})")
                    if call.get('result'):
                        result_preview = str(call['result'])[:200]
                        if len(str(call['result'])) > 200:
                            result_preview += "..."
                        self.stdout.write(f"   Result: {result_preview}")
                self.stdout.write("\n" + "-" * 80)

            # Get answer
            if result["success"]:
                answer = result["answer"]
            else:
                answer = f"⚠️ Agent failed: {result['error']}\n\nPartial information gathered:\n{result.get('answer', 'No information gathered')}"
        else:
            # Generate answer using LLM
            self.stdout.write(f"\n💬 Generating answer using {model}...\n")
            answer = self._generate_answer(question, context, model, language)

        # Display answer
        self.stdout.write("\nANSWER:")
        self.stdout.write("=" * 80)
        self.stdout.write(answer)
        self.stdout.write("=" * 80)
        self.stdout.write("")

    def _detect_language(self, text: str, silent: bool = False) -> str:
        """Simple language detection"""
        dutch_indicators = ['van', 'de', 'der', 'den', 'kinderen', 'geboren', 'overleden', 'wie', 'wanneer', 'waar']
        english_indicators = ['the', 'and', 'of', 'children', 'born', 'died', 'who', 'when', 'where']

        text_lower = text.lower()
        dutch_count = sum(1 for word in dutch_indicators if word in text_lower)
        english_count = sum(1 for word in english_indicators if word in text_lower)

        detected = 'nl' if dutch_count > english_count else 'en'
        if not silent:
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
- If the passages don't contain the EXACT person asked about, still describe what you DID find
- For example, if asked about "Aart the mason" but only find "Aart the farmer", say: "I found Aart van Zanten but he was a farmer, not a mason"
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

ANSWER FORMAT:
- If there are multiple people with the same name, list each one separately
- Use the genealogical identifier (e.g., [V.1.a]) to distinguish between different people
- For each person, include: generation, dates, location, occupation, parents, spouse
- Be clear and specific about which person you're describing

ANSWER:"""

        ollama = OllamaClient(timeout=120)
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'num_ctx': 32768,  # Large context for multi-chunk retrieval
                'temperature': 0.1,
            }
        )

        if not response:
            return "❌ Failed to generate answer"

        return response
