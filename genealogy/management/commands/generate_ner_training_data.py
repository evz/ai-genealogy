"""
Generate NER training data from existing OCR text for genealogical entity extraction.

This command uses the existing regex patterns plus additional patterns to create
BIO-tagged training data suitable for fine-tuning a transformer model.
"""

import json
import random
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from genealogy.models import Document


class Command(BaseCommand):
    help = "Generate NER training data from existing OCR text for genealogical entities"

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-dir",
            type=str,
            default="training_data",
            help="Base directory to save training data (default: training_data)",
        )
        parser.add_argument(
            "--format",
            choices=["conll", "json", "both"],
            default="both",
            help="Output format: CoNLL-2003, JSON, or both (default: both)",
        )
        parser.add_argument(
            "--document-ids",
            nargs="+",
            help="Specific document IDs to process (default: all documents)",
        )
        parser.add_argument(
            "--min-chunk-length",
            type=int,
            default=50,
            help="Minimum chunk length to include (default: 50)",
        )
        parser.add_argument(
            "--include-headers",
            action="store_true",
            help="Include generation header chunks in training data",
        )
        parser.add_argument(
            "--train-ratio",
            type=float,
            default=0.7,
            help="Ratio of data for training (default: 0.7)",
        )
        parser.add_argument(
            "--dev-ratio",
            type=float,
            default=0.15,
            help="Ratio of data for development (default: 0.15)",
        )
        parser.add_argument(
            "--test-ratio",
            type=float,
            default=0.15,
            help="Ratio of data for testing (default: 0.15)",
        )
        parser.add_argument(
            "--max-examples-per-file",
            type=int,
            default=1000,
            help="Maximum examples per file for chunking (default: 1000)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducible splits (default: 42)",
        )

    def handle(self, *args, **options):  # noqa: ARG002
        # Validate split ratios
        total_ratio = options["train_ratio"] + options["dev_ratio"] + options["test_ratio"]
        if abs(total_ratio - 1.0) > 0.001:
            raise CommandError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Set random seed for reproducibility
        random.seed(options["seed"])

        # Create timestamped output directory
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(options["output_dir"]) / f"v1_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for split in ["train", "dev", "test", "metadata", "samples"]:
            (output_dir / split).mkdir(exist_ok=True)

        self.stdout.write(f"Output directory: {output_dir}")

        # Get documents to process
        documents = self._get_documents(options)
        if not documents:
            raise CommandError("No OCR-completed documents found")

        self.stdout.write(f"Processing {len(documents)} documents...")

        # Generate training examples per document (for document-level splits)
        document_examples = self._generate_document_examples(documents, options)

        # Create document-level splits
        doc_splits = self._create_document_splits(
            document_examples,
            options["train_ratio"],
            options["dev_ratio"],
            options["test_ratio"],
        )

        # Save data in requested formats
        data_stats = self._save_training_data(doc_splits, output_dir, options)

        # Generate and save comprehensive statistics
        self._save_statistics(data_stats, documents, output_dir, options)

        # Save human-readable samples
        self._save_samples(doc_splits, output_dir)

        self.stdout.write(
            self.style.SUCCESS(
                f"Training data generated successfully!\n"
                f"Location: {output_dir}\n"
                f"Train: {data_stats['train']['examples']} examples "
                f"({data_stats['train']['files']} files)\n"
                f"Dev: {data_stats['dev']['examples']} examples "
                f"({data_stats['dev']['files']} files)\n"
                f"Test: {data_stats['test']['examples']} examples "
                f"({data_stats['test']['files']} files)\n"
                f"Total entities: {sum(data_stats['entity_counts'].values())}"
            )
        )

    def _get_documents(self, options):
        """Get documents to process based on options"""
        if options["document_ids"]:
            documents = list(Document.objects.filter(id__in=options["document_ids"]))
            if not documents:
                raise CommandError("No documents found with provided IDs")
        else:
            documents = list(Document.objects.filter(ocr_completed=True))
        return documents

    def _generate_document_examples(self, documents, options):
        """Generate training examples grouped by document"""
        extractor = GenealogyEntityExtractor()
        document_examples = {}

        for document in documents:
            self.stdout.write(f"Processing document: {document.title}")

            # Get chunks for this document
            chunks = document.text_chunks.all()
            if not options["include_headers"]:
                chunks = chunks.exclude(chunk_type="HEADER")

            chunks = chunks.filter(text_content__isnull=False).exclude(text_content="")

            examples = []
            for chunk in chunks:
                if len(chunk.text_content.strip()) < options["min_chunk_length"]:
                    continue

                # Extract entities and create training example
                training_example = extractor.create_training_example(
                    chunk.text_content,
                    chunk_id=str(chunk.id),
                    document_title=document.title,
                    generation=chunk.generation_number,
                    existing_genealogy_ids=chunk.genealogy_ids,
                )

                if training_example:
                    examples.append(training_example)

            if examples:
                document_examples[str(document.id)] = {
                    "title": document.title,
                    "examples": examples,
                    "entity_counts": extractor.get_document_entity_counts(),
                }

        return document_examples

    def _create_document_splits(
        self,
        document_examples,
        train_ratio,
        dev_ratio,
        test_ratio,  # noqa: ARG002
    ):
        """Create train/dev/test splits at document level to prevent data leakage"""
        document_ids = list(document_examples.keys())
        random.shuffle(document_ids)

        n_docs = len(document_ids)

        # Handle single document case - split examples instead of documents
        if n_docs == 1:
            self.stdout.write(
                self.style.WARNING(
                    "Only one document found. Splitting examples within document " "instead of document-level split."
                )
            )
            doc_id = document_ids[0]
            all_examples = document_examples[doc_id]["examples"]
            random.shuffle(all_examples)

            n_examples = len(all_examples)
            train_end = int(n_examples * train_ratio)
            dev_end = train_end + int(n_examples * dev_ratio)

            return {
                "train": all_examples[:train_end],
                "dev": all_examples[train_end:dev_end],
                "test": all_examples[dev_end:],
            }

        # Multiple documents - use document-level splits
        train_end = int(n_docs * train_ratio)
        dev_end = train_end + int(n_docs * dev_ratio)

        splits = {
            "train": document_ids[:train_end],
            "dev": document_ids[train_end:dev_end],
            "test": document_ids[dev_end:],
        }

        # Collect examples for each split
        split_data = {}
        for split_name, doc_ids in splits.items():
            examples = []
            for doc_id in doc_ids:
                examples.extend(document_examples[doc_id]["examples"])
            split_data[split_name] = examples

        return split_data

    def _save_training_data(self, splits, output_dir, options):
        """Save training data in requested formats with chunking"""
        data_stats = {
            "train": {"examples": 0, "files": 0},
            "dev": {"examples": 0, "files": 0},
            "test": {"examples": 0, "files": 0},
            "entity_counts": Counter(),
        }

        max_per_file = options["max_examples_per_file"]

        for split_name, examples in splits.items():
            if not examples:
                continue

            # Count entities across all examples
            for example in examples:
                for label in example["labels"]:
                    if label.startswith("B-"):
                        entity_type = label[2:]
                        data_stats["entity_counts"][entity_type] += 1

            # Split into chunks if needed
            file_chunks = []
            for i in range(0, len(examples), max_per_file):
                file_chunks.append(examples[i : i + max_per_file])

            data_stats[split_name]["examples"] = len(examples)
            data_stats[split_name]["files"] = len(file_chunks)

            # Save each chunk
            for chunk_idx, chunk_examples in enumerate(file_chunks):
                file_suffix = f"_{chunk_idx:03d}" if len(file_chunks) > 1 else ""

                if options["format"] in ["conll", "both"]:
                    conll_file = output_dir / split_name / f"{split_name}{file_suffix}.conll"
                    self._save_conll_format(chunk_examples, conll_file)

                if options["format"] in ["json", "both"]:
                    json_file = output_dir / split_name / f"{split_name}{file_suffix}.json"
                    self._save_json_format(chunk_examples, json_file)

        return data_stats

    def _save_conll_format(self, examples, file_path):
        """Save examples in CoNLL-2003 BIO format"""
        with open(file_path, "w", encoding="utf-8") as f:
            for example in examples:
                # Write metadata as comments
                f.write(f"# Document: {example['metadata']['document_title']}\n")
                f.write(f"# Chunk: {example['metadata']['chunk_id']}\n")
                if example["metadata"]["generation"]:
                    f.write(f"# Generation: {example['metadata']['generation']}\n")
                f.write("\n")

                # Write tokens and labels
                for token, label in zip(example["tokens"], example["labels"], strict=False):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")  # Example separator

    def _save_json_format(self, examples, file_path):
        """Save examples in JSON format"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    def _save_statistics(self, data_stats, documents, output_dir, options):
        """Save comprehensive statistics about the training data"""
        stats = {
            "generation_timestamp": datetime.now(UTC).isoformat(),
            "parameters": {
                "min_chunk_length": options["min_chunk_length"],
                "include_headers": options["include_headers"],
                "train_ratio": options["train_ratio"],
                "dev_ratio": options["dev_ratio"],
                "test_ratio": options["test_ratio"],
                "max_examples_per_file": options["max_examples_per_file"],
                "seed": options["seed"],
                "format": options["format"],
            },
            "data_overview": {
                "total_documents": len(documents),
                "documents_processed": len([d for d in documents if d.text_chunks.exists()]),
                "total_examples": sum(
                    split["examples"]
                    for split in data_stats.values()
                    if isinstance(split, dict) and "examples" in split
                ),
                "train_examples": data_stats["train"]["examples"],
                "dev_examples": data_stats["dev"]["examples"],
                "test_examples": data_stats["test"]["examples"],
            },
            "entity_statistics": dict(data_stats["entity_counts"]),
            "file_statistics": {
                "train_files": data_stats["train"]["files"],
                "dev_files": data_stats["dev"]["files"],
                "test_files": data_stats["test"]["files"],
            },
            "document_details": [
                {
                    "id": str(doc.id),
                    "title": doc.title,
                    "languages": doc.languages,
                    "page_count": doc.page_count,
                    "chunk_count": doc.text_chunks.count(),
                }
                for doc in documents
            ],
        }

        stats_file = output_dir / "training_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def _save_samples(self, splits, output_dir):
        """Save human-readable samples for inspection"""
        for split_name, examples in splits.items():
            if not examples:
                continue

            # Take first 10 examples as samples
            samples = examples[:10]

            sample_file = output_dir / "samples" / f"{split_name}_samples.txt"
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write(f"=== {split_name.upper()} SAMPLES ===\n\n")

                for i, example in enumerate(samples, 1):
                    f.write(f"EXAMPLE {i}\n")
                    f.write(f"Document: {example['metadata']['document_title']}\n")
                    f.write(f"Chunk: {example['metadata']['chunk_id']}\n")
                    f.write(f"Original text: {example['text'][:200]}...\n\n")

                    f.write("Tokens and Labels:\n")
                    for token, label in zip(example["tokens"], example["labels"], strict=False):
                        if label != "O":
                            f.write(f"  {token} -> {label}\n")

                    f.write("\n" + "=" * 50 + "\n\n")


class GenealogyEntityExtractor:
    """Extract genealogical entities from text using enhanced regex patterns"""

    # Entity types we want to extract
    ENTITY_TYPES = [
        "GENEALOGY_ID",  # II.1.a, XII.5.b
        "PERSON_NAME",  # John Smith, Mary van der Berg
        "DATE",  # 1845, 15 maart 1920, circa 1850
        "PLACE",  # Amsterdam, Utrecht, Nederland
        "FAMILY_GROUP",  # X.9. Children of...
        "GENERATION_HEADER",  # EERSTE GENERATIE, etc.
    ]

    def __init__(self):
        self.entity_counts = {entity_type: 0 for entity_type in self.ENTITY_TYPES}
        self.document_entity_counts = {entity_type: 0 for entity_type in self.ENTITY_TYPES}
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns for entity extraction"""

        # Genealogical IDs (enhanced with OCR corrections)
        self.genealogy_id_pattern = re.compile(
            r"\b(?:"
            # Standard format: II.1.a, XII.5.b
            r"([IVXLCDMilvxlcdm]+)\.(\d+)\.([a-zA-Z])"
            r"|"
            # OCR corrupted formats: IL.1.a, VIL.2.b, etc.
            r"([A-Z]{1,4}[LI]+)\.(\d+)\.([a-zA-Z])"
            r"|"
            # Spaced variants: II. 1. a, XII. 5. b
            r"([IVXLCDMilvxlcdm]+)\.\s*(\d+)\.\s*([a-zA-Z])"
            r")\b"
        )

        # Family group headers (enhanced English and Dutch)
        self.family_group_pattern = re.compile(
            r"(?:"
            # Full family headers: "X.9. Children of John & Mary"
            r"\b([IVXLCDMilvxlcdm]+|[A-Z]{1,4}[LI]*)\.(\d+)\.\s+(?:Children\s+of|Kinderen\s+van|children\s+of|kinderen\s+van)(?:\s+[A-Z][a-zA-Z\s&,\-\.]*)?"
            r"|"
            # Short family headers: "X.9. Children"
            r"\b([IVXLCDMilvxlcdm]+|[A-Z]{1,4}[LI]*)\.(\d+)\.\s+(?:Children|Kinderen|children|kinderen)(?!\s+of)"
            r"|"
            # Variant with line breaks: "X.9.\nChildren of"
            r"\b([IVXLCDMilvxlcdm]+|[A-Z]{1,4}[LI]*)\.(\d+)\.\s*\n?\s*(?:Children\s+of|Kinderen\s+van|children\s+of|kinderen\s+van)"
            r")",
            re.IGNORECASE | re.MULTILINE,
        )

        # Generation headers (Dutch)
        self.generation_pattern = re.compile(
            r"\b(eerste|tweede|derde|vierde|vijfde|zesde|zevende|achtste|negende|tiende|elfde|twaalfde)\s+generatie\b",
            re.IGNORECASE,
        )

        # Dates (various formats)
        self.date_pattern = re.compile(
            r"\b(?:"
            r"(?:circa|ca\.?|around|about|omstreeks)?\s*"  # Optional circa
            r"(?:"
            r"(?:\d{1,2}[-/\s]*)?"  # Optional day
            r"(?:januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december|"
            r"january|february|march|april|may|june|july|august|september|october|november|december|"
            r"jan|feb|mrt|apr|mei|jun|jul|aug|sep|okt|nov|dec)[-/\s]*"  # Month names
            r")?\s*"
            r"(?:1[5-9]\d{2}|20[0-2]\d)"  # Years 1500-2029
            r")\b",
            re.IGNORECASE,
        )

        # Simple year dates
        self.year_pattern = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")

        # Dutch place names (enhanced patterns)
        self.place_pattern = re.compile(
            r"\b(?:"
            # Major Dutch cities
            r"(?:Amsterdam|Utrecht|Rotterdam|Den Haag|s-Gravenhage|Eindhoven|Tilburg|"
            r"Groningen|"
            r"Almere|Breda|Nijmegen|Enschede|Haarlem|Arnhem|Amersfoort|Zaanstad|Apeldoorn|"
            r"s-Hertogenbosch|Hoofddorp|Maastricht|Leiden|Dordrecht|Zoetermeer|Zwolle|"
            r"Deventer|Delft|Alkmaar|Leeuwarden|Westland|Hilversum|Venlo|Roosendaal|"
            r"Ede|Helmond|Purmerend|Leidschendam|Alphen|Gouda|Spijkenisse|Vlaardingen)"
            r"|"
            # Dutch geographical indicators
            r"(?:Nederland|Holland|Gelderland|Noord-Holland|Zuid-Holland|Zeeland|"
            r"Noord-Brabant|Limburg|Friesland|Overijssel|Flevoland|Drenthe|Utrecht|Groningen)"
            r"|"
            # General place pattern (capitalized words, possibly with Dutch particles)
            r"(?:[A-Z][a-z]+(?:\s+(?:aan|bij|in|op|onder|over|ter|te|ten|tot|van|voor)\s+[A-Z][a-z]+)*)"
            r")\b"
        )

        # Person names (enhanced Dutch patterns)
        self.person_name_pattern = re.compile(
            r"\b(?:"
            # Full Dutch names with particles
            r"(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\s+(?:van|de|der|den|te|ten|tot|op|aan|in|bij|d\'|du|le|la|des)\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)"
            r"|"
            # Names with initials + particles: "J.A. van Bolhuis", "H.W. de Jong"
            r"(?:[A-Z]\.(?:\s*[A-Z]\.)*\s+(?:van|de|der|den|te|ten|tot|op|aan|in|bij|d\'|du|le|la|des\s+)[A-Z][a-z]{2,})"
            r"|"
            # Simple first + last name: "Johannes Bulhuis", "Maria Jansen"
            r"(?:[A-Z][a-z]{3,}\s+[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})?)"
            r"|"
            # Formal names with titles: "Mr. Johannes van Zanten", "Dr. Maria de Jong"
            r"(?:(?:Mr|Mrs|Dr|Prof|Drs?)\.?\s+[A-Z][a-z]{2,}(?:\s+(?:van|de|der|den|te|ten|tot|op|aan|in|bij|d\'|du|le|la|des))?\s+[A-Z][a-z]{2,})"
            r")\b"
        )

    def create_training_example(
        self,
        text,
        chunk_id=None,
        document_title=None,
        generation=None,
        existing_genealogy_ids=None,
    ):
        """Create a BIO-tagged training example from text"""

        # Reset document-level counts for this example
        self.document_entity_counts = {entity_type: 0 for entity_type in self.ENTITY_TYPES}

        # Tokenize text (simple whitespace + punctuation splitting)
        tokens = self._tokenize(text)
        if len(tokens) < 3:  # Skip very short texts
            return None

        # Initialize all labels as 'O' (Outside)
        labels = ["O"] * len(tokens)

        # Extract entities and apply BIO tagging (order matters - more specific first)
        # Do most specific entities first to avoid conflicts
        self._tag_genealogy_ids(text, tokens, labels)
        self._tag_family_groups(text, tokens, labels)
        self._tag_generation_headers(text, tokens, labels)
        self._tag_person_names(text, tokens, labels)  # Names before dates/places
        self._tag_dates(text, tokens, labels)
        self._tag_places(text, tokens, labels)  # Places last to avoid over-tagging

        return {
            "tokens": tokens,
            "labels": labels,
            "text": text,
            "metadata": {
                "chunk_id": chunk_id,
                "document_title": document_title,
                "generation": generation,
                "existing_genealogy_ids": existing_genealogy_ids or [],
                "entity_counts": dict(self.document_entity_counts),
            },
        }

    def _tokenize(self, text):
        """Simple tokenization splitting on whitespace and punctuation"""
        # Split on whitespace and common punctuation, but keep the delimiters
        return re.findall(r"\S+", text)

    def _find_token_spans(self, text, tokens):
        """Find character spans for each token in the original text"""
        spans = []
        current_pos = 0

        for token in tokens:
            # Find the token in the text starting from current position
            start = text.find(token, current_pos)
            if start == -1:
                # Token not found, approximate position
                spans.append((current_pos, current_pos + len(token)))
                current_pos += len(token) + 1
            else:
                end = start + len(token)
                spans.append((start, end))
                current_pos = end

        return spans

    def _tag_entities_with_pattern(self, text, tokens, labels, pattern, entity_type):
        """Generic method to tag entities using a regex pattern"""
        token_spans = self._find_token_spans(text, tokens)

        for match in pattern.finditer(text):
            start, end = match.span()

            # Find overlapping tokens
            overlapping_tokens = []
            for i, (token_start, token_end) in enumerate(token_spans):
                if token_start < end and token_end > start:
                    overlapping_tokens.append(i)

            if overlapping_tokens:
                # Check if tokens are already tagged with different entity
                can_tag = True
                for token_idx in overlapping_tokens:
                    if labels[token_idx] != "O":
                        # Skip this entity if it conflicts with existing labels
                        can_tag = False
                        break

                if can_tag:
                    # Apply proper BIO tagging
                    for i, token_idx in enumerate(overlapping_tokens):
                        if i == 0:
                            labels[token_idx] = f"B-{entity_type}"
                        else:
                            labels[token_idx] = f"I-{entity_type}"

                    self.entity_counts[entity_type] += 1
                    self.document_entity_counts[entity_type] += 1

    def _tag_genealogy_ids(self, text, tokens, labels):
        """Tag genealogical IDs (II.1.a, etc.)"""
        self._tag_entities_with_pattern(text, tokens, labels, self.genealogy_id_pattern, "GENEALOGY_ID")

    def _tag_family_groups(self, text, tokens, labels):
        """Tag family group headers"""
        self._tag_entities_with_pattern(text, tokens, labels, self.family_group_pattern, "FAMILY_GROUP")

    def _tag_generation_headers(self, text, tokens, labels):
        """Tag generation headers"""
        self._tag_entities_with_pattern(text, tokens, labels, self.generation_pattern, "GENERATION_HEADER")

    def _tag_dates(self, text, tokens, labels):
        """Tag dates"""
        # Tag complex dates first
        self._tag_entities_with_pattern(text, tokens, labels, self.date_pattern, "DATE")
        # Then simple years (but don't overwrite existing DATE tags)
        self._tag_entities_with_pattern(text, tokens, labels, self.year_pattern, "DATE")

    def _tag_places(self, text, tokens, labels):
        """Tag place names"""
        self._tag_entities_with_pattern(text, tokens, labels, self.place_pattern, "PLACE")

    def _tag_person_names(self, text, tokens, labels):
        """Tag person names (most challenging, do this last)"""
        self._tag_entities_with_pattern(text, tokens, labels, self.person_name_pattern, "PERSON_NAME")

    def get_entity_statistics(self):
        """Get overall statistics about extracted entities"""
        return dict(self.entity_counts)

    def get_document_entity_counts(self):
        """Get entity counts for the current document/example"""
        return dict(self.document_entity_counts)
