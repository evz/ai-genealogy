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
from genealogy.patterns import GenealogyPatterns


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
        parser.add_argument(
            "--merge-with-conll",
            action="store_true",
            help="Merge with CoNLL-2002 Dutch dataset for general entity training",
        )
        parser.add_argument(
            "--conll-dir",
            type=str,
            default="datasets",
            help="Directory containing CoNLL-2002 dataset (default: datasets)",
        )
        parser.add_argument(
            "--generate-negatives",
            action="store_true",
            help="Generate negative examples for genealogy abbreviations (recommended)",
        )
        parser.add_argument(
            "--negative-examples-count",
            type=int,
            default=400,
            help="Number of negative examples to generate (default: 400)",
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

        # Generate negative examples if requested
        if options["generate_negatives"]:
            self.stdout.write("\n" + "=" * 50)
            self.stdout.write("Generating negative examples for genealogy abbreviations...")
            negative_examples = self._generate_negative_examples(options["negative_examples_count"])

            # Add negative examples to existing splits
            for split_name in ["train", "dev", "test"]:
                split_examples = negative_examples[split_name]
                if split_examples:
                    # Load existing data
                    existing_file = output_dir / split_name / f"{split_name}.json"
                    with open(existing_file, encoding="utf-8") as f:
                        existing_data = json.load(f)

                    # Merge with negative examples
                    combined_data = existing_data + split_examples
                    random.shuffle(combined_data)

                    # Save updated data
                    with open(existing_file, "w", encoding="utf-8") as f:
                        json.dump(combined_data, f, indent=2, ensure_ascii=False)

                    # Update CoNLL format too
                    conll_file = output_dir / split_name / f"{split_name}.conll"
                    with open(conll_file, "w", encoding="utf-8") as f:
                        for example in combined_data:
                            source = example.get("metadata", {}).get("source", "genealogy")
                            f.write(f"# Source: {source}\n")
                            if example.get("metadata", {}).get("document_title"):
                                f.write(f"# Document: {example['metadata']['document_title']}\n")
                            f.write("\n")

                            for token, label in zip(example["tokens"], example["labels"], strict=False):
                                f.write(f"{token}\t{label}\n")
                            f.write("\n")

            total_negatives = sum(len(examples) for examples in negative_examples.values())
            self.stdout.write(
                self.style.SUCCESS(
                    f"Added {total_negatives} negative examples to training data\n"
                    "These examples help the model avoid false positives on genealogy abbreviations."
                )
            )

        # Merge with CoNLL-2002 Dutch dataset if requested
        if options["merge_with_conll"]:
            self.stdout.write("\n" + "=" * 50)
            self.stdout.write("Merging with CoNLL-2002 Dutch dataset...")
            merged_output_dir = self._merge_with_conll(output_dir, options["conll_dir"])
            self.stdout.write(
                self.style.SUCCESS(
                    f"Merged dataset created at: {merged_output_dir}\n"
                    "This dataset combines genealogy-specific entities with general Dutch NER data."
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

    def _merge_with_conll(self, genealogy_dir: Path, conll_dir: str) -> Path:
        """Merge genealogy training data with CoNLL-2002 Dutch dataset"""
        conll_path = Path(conll_dir)
        merged_output_dir = Path(str(genealogy_dir) + "_merged")

        # Create output directory
        merged_output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["train", "dev", "test", "metadata"]:
            (merged_output_dir / subdir).mkdir(exist_ok=True)

        self.stdout.write("Converting CoNLL-2002 Dutch data...")

        # Convert CoNLL files to our format
        conll_train_data = self._convert_conll_file(conll_path / "ner" / "data" / "ned.train")
        conll_dev_data = self._convert_conll_file(conll_path / "ner" / "data" / "ned.testa")
        conll_test_data = self._convert_conll_file(conll_path / "ner" / "data" / "ned.testb")

        self.stdout.write(
            f"Converted CoNLL: {len(conll_train_data)} train, {len(conll_dev_data)} dev, {len(conll_test_data)} test"
        )

        # Load genealogy data
        with open(genealogy_dir / "train" / "train.json", encoding="utf-8") as f:
            gen_train = json.load(f)
        with open(genealogy_dir / "dev" / "dev.json", encoding="utf-8") as f:
            gen_dev = json.load(f)
        with open(genealogy_dir / "test" / "test.json", encoding="utf-8") as f:
            gen_test = json.load(f)

        # Merge datasets
        merged_train = gen_train + conll_train_data
        merged_dev = gen_dev + conll_dev_data
        merged_test = gen_test + conll_test_data

        # Shuffle combined datasets
        random.seed(42)
        random.shuffle(merged_train)
        random.shuffle(merged_dev)
        random.shuffle(merged_test)

        self.stdout.write(f"Merged totals: {len(merged_train)} train, {len(merged_dev)} dev, {len(merged_test)} test")

        # Save merged datasets
        splits = {"train": merged_train, "dev": merged_dev, "test": merged_test}

        for split_name, examples in splits.items():
            # Save JSON format
            json_file = merged_output_dir / split_name / f"{split_name}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)

            # Save CoNLL format
            conll_file = merged_output_dir / split_name / f"{split_name}.conll"
            with open(conll_file, "w", encoding="utf-8") as f:
                for example in examples:
                    # Write metadata as comments
                    source = example.get("metadata", {}).get("source", "genealogy")
                    f.write(f"# Source: {source}\n")
                    if example.get("metadata", {}).get("document_title"):
                        f.write(f"# Document: {example['metadata']['document_title']}\n")
                    f.write("\n")

                    # Write tokens and labels
                    for token, label in zip(example["tokens"], example["labels"], strict=False):
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")

        # Generate statistics
        stats = {
            "generation_timestamp": datetime.now(UTC).isoformat(),
            "datasets_merged": ["genealogy-specific", "CoNLL-2002-Dutch"],
            "data_overview": {
                "train_examples": len(merged_train),
                "dev_examples": len(merged_dev),
                "test_examples": len(merged_test),
                "total_examples": len(merged_train) + len(merged_dev) + len(merged_test),
            },
            "genealogy_contribution": {"train": len(gen_train), "dev": len(gen_dev), "test": len(gen_test)},
            "conll_contribution": {
                "train": len(conll_train_data),
                "dev": len(conll_dev_data),
                "test": len(conll_test_data),
            },
        }

        stats_file = merged_output_dir / "training_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return merged_output_dir

    def _convert_conll_file(self, conll_file: Path) -> list[dict]:
        """Convert CoNLL format file to our genealogy training format"""
        entity_mapping = {"PER": "PERSON_NAME", "LOC": "PLACE", "ORG": "ORG", "MISC": "MISC"}

        examples = []
        current_tokens = []
        current_labels = []

        with open(conll_file, encoding="latin1") as f:
            for line in f:
                line = line.strip()

                # Skip document start markers
                if line.startswith("-DOCSTART-"):
                    continue

                # Empty line = end of sentence
                if not line:
                    if current_tokens:
                        # Convert labels
                        converted_labels = []
                        for label in current_labels:
                            if label == "O":
                                converted_labels.append("O")
                            else:
                                prefix = label[:2]  # B- or I-
                                entity_type = label[2:]
                                if entity_type in entity_mapping:
                                    converted_labels.append(f"{prefix}{entity_mapping[entity_type]}")
                                else:
                                    converted_labels.append("O")  # Unknown entity type

                        # Create example
                        example = {
                            "tokens": current_tokens.copy(),
                            "labels": converted_labels,
                            "text": " ".join(current_tokens),
                            "metadata": {
                                "source": "CoNLL-2002-Dutch",
                                "chunk_id": f"conll_{len(examples)}",
                                "document_title": "CoNLL-2002 Dutch NER",
                                "generation": None,
                                "entity_counts": {},
                            },
                        }
                        examples.append(example)

                        current_tokens = []
                        current_labels = []
                    continue

                # Parse token line
                parts = line.split()
                if len(parts) >= 3:  # word, pos, entity
                    token = parts[0]
                    entity_label = parts[2]

                    current_tokens.append(token)
                    current_labels.append(entity_label)

        # Handle final example if file doesn't end with empty line
        if current_tokens:
            converted_labels = []
            for label in current_labels:
                if label == "O":
                    converted_labels.append("O")
                else:
                    prefix = label[:2]
                    entity_type = label[2:]
                    if entity_type in entity_mapping:
                        converted_labels.append(f"{prefix}{entity_mapping[entity_type]}")
                    else:
                        converted_labels.append("O")

            example = {
                "tokens": current_tokens.copy(),
                "labels": converted_labels,
                "text": " ".join(current_tokens),
                "metadata": {
                    "source": "CoNLL-2002-Dutch",
                    "chunk_id": f"conll_{len(examples)}",
                    "document_title": "CoNLL-2002 Dutch NER",
                    "generation": None,
                    "entity_counts": {},
                },
            }
            examples.append(example)

        return examples

    def _generate_negative_examples(self, total_count: int) -> dict:
        """Generate negative examples for genealogy abbreviations"""

        # Import the negative examples generator logic
        GENEALOGICAL_ABBREVIATIONS = [
            "a.",
            "b.",
            "c.",
            "d.",
            "e.",
            "f.",
            "g.",
            "h.",
            "~",
            "*",
            "+",
            "x",
            "hertr",
            "hertrouwd",
            "zv",
            "zoon van",
            "dv",
            "dochter van",
            "do",
            "so",
            "EL",
            "Echtelieden",
            "echtpaar",
            "JD",
            "Jonge Dochter",
            "JM",
            "Jonge Man",
            "kdv",
            "kleindochter van",
            "kzv",
            "kleinzoon van",
            "ls",
            "leeftsamen",
            "wed",
            "weduwe",
            "wednr",
            "weduwenaar",
            "Doopgetuige",
            "doopgetuige",
            "[buried]",
        ]

        DUTCH_PLACES = [
            "Haaften",
            "Opijnen",
            "Culemborg",
            "Amsterdam",
            "Utrecht",
            "Rotterdam",
            "Den Haag",
            "Eindhoven",
            "Tilburg",
            "Groningen",
            "Breda",
            "Nijmegen",
            "Haarlem",
            "Arnhem",
            "Amersfoort",
            "Maastricht",
            "Leiden",
            "Dordrecht",
            "Tiel",
            "Geldermalsen",
            "Beusichem",
            "Nederland",
            "Holland",
            "Gelderland",
        ]

        DUTCH_FIRST_NAMES = [
            "Jenneke",
            "Hendrik",
            "Anna",
            "Anneke",
            "Geurt",
            "Jannetje",
            "Johannes",
            "Maria",
            "Pieter",
            "Elisabeth",
            "Jacobus",
            "Catharina",
            "Willem",
            "Margaretha",
            "Cornelis",
            "Johanna",
            "Gerrit",
            "Neeltje",
            "Jan",
            "Antje",
            "Nicolaas",
            "Grietje",
            "Adriaan",
            "Aaltje",
        ]

        DUTCH_SURNAMES = [
            "van Zanten",
            "van Eck",
            "van Haaften",
            "Tukker",
            "Donker",
            "van der Berg",
            "de Jong",
            "van den Broek",
            "van der Meer",
            "de Vries",
            "van Dijk",
            "Bakker",
            "Jansen",
            "Visser",
            "Smit",
            "van der Steen",
            "de Wit",
            "van Leeuwen",
            "Mulder",
            "de Boer",
        ]

        def generate_date():
            day = random.randint(1, 28)
            month = random.randint(1, 12)
            year = random.randint(1650, 1900)
            return f"{day}.{month}.{year}"

        def is_date(token):
            import re

            return bool(re.match(r"^\d{1,2}\.\d{1,2}\.\d{4}$|^\d{1,2}\.\d{4}$|^\d{4}$", token))

        def create_labeled_example(text, category):
            tokens = text.split()
            labels = []

            for token in tokens:
                clean_token = token.rstrip(",.:;")

                if clean_token in GENEALOGICAL_ABBREVIATIONS:
                    labels.append("O")  # This is the key - abbreviations are NOT entities
                elif clean_token in DUTCH_PLACES:
                    labels.append("B-PLACE")
                elif is_date(clean_token):
                    labels.append("B-DATE")
                elif clean_token in DUTCH_FIRST_NAMES or any(clean_token in surname for surname in DUTCH_SURNAMES):
                    if labels and labels[-1].endswith("-PERSON_NAME"):
                        labels.append("I-PERSON_NAME")
                    else:
                        labels.append("B-PERSON_NAME")
                else:
                    labels.append("O")

            return {
                "tokens": tokens,
                "labels": labels,
                "text": text,
                "metadata": {
                    "source": "dutch_genealogy_negative_examples",
                    "category": category,
                    "chunk_id": f"neg_{category}_{random.randint(1000, 9999)}",
                    "document_title": "Dutch Genealogy Negative Examples",
                    "generation": None,
                    "entity_counts": {label[2:]: 1 for label in labels if label.startswith("B-")},
                },
            }

        # Generate different types of negative examples
        examples = []

        # Birth order examples (30%)
        templates = [
            "{letter} {first_name} {surname}, ~ {place} {date}",
            "{letter} {first_name} {surname}, * {place} {date}",
            "{letter} {first_name} {surname}, ~ {place} {date}, + {place2} {date2}",
        ]

        for _ in range(int(total_count * 0.3)):
            template = random.choice(templates)
            text = template.format(
                letter=random.choice(["a.", "b.", "c.", "d."]),
                first_name=random.choice(DUTCH_FIRST_NAMES),
                surname=random.choice(DUTCH_SURNAMES),
                place=random.choice(DUTCH_PLACES),
                place2=random.choice(DUTCH_PLACES),
                date=generate_date(),
                date2=generate_date(),
            )
            examples.append(create_labeled_example(text, "birth_order"))

        # Lifecycle examples (40%)
        lifecycle_templates = [
            "* {place} {date}",
            "~ {place} {date}",
            "+ {place} {date}",
            "x {place} {date} {first_name} {surname}",
            "{first_name} {surname}, ~ {place} {date}, + {place2} {date2}",
        ]

        for _ in range(int(total_count * 0.4)):
            template = random.choice(lifecycle_templates)
            text = template.format(
                first_name=random.choice(DUTCH_FIRST_NAMES),
                surname=random.choice(DUTCH_SURNAMES),
                place=random.choice(DUTCH_PLACES),
                place2=random.choice(DUTCH_PLACES),
                date=generate_date(),
                date2=generate_date(),
            )
            examples.append(create_labeled_example(text, "lifecycle"))

        # Relationship examples (30%)
        relationship_templates = [
            "zv {first_name} {surname} en {first_name2} {surname2}",
            "dv {first_name} {surname}",
            "{first_name} {surname}, zv {first_name2} {surname2}",
            "Doopgetuige {first_name} {surname}",
        ]

        for _ in range(int(total_count * 0.3)):
            template = random.choice(relationship_templates)
            text = template.format(
                first_name=random.choice(DUTCH_FIRST_NAMES),
                first_name2=random.choice(DUTCH_FIRST_NAMES),
                surname=random.choice(DUTCH_SURNAMES),
                surname2=random.choice(DUTCH_SURNAMES),
            )
            examples.append(create_labeled_example(text, "relationship"))

        # Split examples across train/dev/test
        random.shuffle(examples)
        train_end = int(len(examples) * 0.7)
        dev_end = train_end + int(len(examples) * 0.15)

        return {"train": examples[:train_end], "dev": examples[train_end:dev_end], "test": examples[dev_end:]}


class GenealogyEntityExtractor:
    """Extract genealogical entities from text using enhanced regex patterns"""

    # Entity types we want to extract (genealogy-specific only)
    # General entities (PERSON_NAME, PLACE, ORG) will come from CoNLL dataset
    ENTITY_TYPES = [
        "GENEALOGY_ID",  # II.1.a, XII.5.b
        "DATE",  # 1845, 15 maart 1920, circa 1850
        "FAMILY_GROUP",  # X.9. Children of...
        "GENERATION_HEADER",  # EERSTE GENERATIE, etc.
    ]

    def __init__(self):
        self.entity_counts = {entity_type: 0 for entity_type in self.ENTITY_TYPES}
        self.document_entity_counts = {entity_type: 0 for entity_type in self.ENTITY_TYPES}
        self._compile_patterns()

    def _compile_patterns(self):
        """Use centralized patterns from genealogy.patterns module (genealogy-specific only)"""
        # Use centralized patterns to ensure consistency
        # Only load patterns for genealogy-specific entities
        self.genealogy_id_pattern = GenealogyPatterns.GENEALOGY_ID
        self.family_group_pattern = GenealogyPatterns.FAMILY_GROUP
        self.generation_pattern = GenealogyPatterns.GENERATION_HEADER
        self.date_pattern = GenealogyPatterns.DATE_COMPREHENSIVE
        self.year_pattern = GenealogyPatterns.YEAR
        # Removed: place_pattern, person_name_pattern (will use CoNLL for these)

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

        # Extract entities and apply BIO tagging (genealogy-specific only)
        # General entities (PERSON_NAME, PLACE, ORG) will come from CoNLL dataset
        self._tag_genealogy_ids(text, tokens, labels)
        self._tag_family_groups(text, tokens, labels)
        self._tag_generation_headers(text, tokens, labels)
        self._tag_dates(text, tokens, labels)

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

    def get_entity_statistics(self):
        """Get overall statistics about extracted entities"""
        return dict(self.entity_counts)

    def get_document_entity_counts(self):
        """Get entity counts for the current document/example"""
        return dict(self.document_entity_counts)
