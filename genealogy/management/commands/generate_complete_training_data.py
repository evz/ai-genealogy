"""
Generate complete NER training data combining multiple sources.

This command orchestrates the generation of training data from:
1. CoNLL-2002 Dutch dataset (general Dutch NER)
2. OCR-extracted genealogical text chunks
3. Synthetic negative examples for Dutch genealogical abbreviations

Usage:
    python manage.py generate_complete_training_data --output-dir training_data_complete
"""

import json
import random
from datetime import UTC, datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from genealogy.management.commands.generate_ner_training_data import (
    Command as GenealogyCommand,
)
from genealogy.models import Document
from genealogy.training_tools.conll_converter import load_conll_dataset
from genealogy.training_tools.negative_examples import DutchGenealogyNegativeGenerator


class Command(BaseCommand):
    help = "Generate complete NER training data from all sources"

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-dir",
            type=str,
            default="training_data_complete",
            help="Base directory to save training data (default: training_data_complete)",
        )
        parser.add_argument(
            "--conll-dir",
            type=str,
            default="datasets",
            help="Directory containing CoNLL-2002 dataset (default: datasets)",
        )
        parser.add_argument(
            "--include-genealogy-chunks",
            action="store_true",
            help="Include genealogy data from OCR text chunks",
        )
        parser.add_argument(
            "--include-negative-examples",
            action="store_true",
            default=True,
            help="Include synthetic negative examples for Dutch abbreviations (default: True)",
        )
        parser.add_argument(
            "--negative-examples-count",
            type=int,
            default=150,
            help="Number of negative examples to generate (default: 150)",
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
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducible splits (default: 42)",
        )

    def handle(self, *args, **options):
        # Validate split ratios
        total_ratio = options["train_ratio"] + options["dev_ratio"] + options["test_ratio"]
        if abs(total_ratio - 1.0) > 0.001:
            raise CommandError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Set random seed
        random.seed(options["seed"])

        # Create timestamped output directory
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(options["output_dir"]) / f"complete_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["train", "dev", "test", "metadata", "sources"]:
            (output_dir / subdir).mkdir(exist_ok=True)

        self.stdout.write(f"🏗️  Creating complete training dataset in: {output_dir}")

        all_examples = []
        source_stats = {}

        # Source 1: CoNLL-2002 Dutch dataset
        self.stdout.write("\n📚 Processing CoNLL-2002 Dutch dataset...")
        conll_examples = load_conll_dataset(options["conll_dir"])
        all_examples.extend(conll_examples)
        source_stats["conll"] = len(conll_examples)
        self.stdout.write(f"  ✅ Loaded {len(conll_examples)} CoNLL examples")

        # Source 2: Genealogy chunks (optional)
        if options["include_genealogy_chunks"]:
            self.stdout.write("\n📜 Processing genealogy text chunks...")
            genealogy_examples = self._load_genealogy_data(output_dir / "sources")
            all_examples.extend(genealogy_examples)
            source_stats["genealogy"] = len(genealogy_examples)
            self.stdout.write(f"  ✅ Loaded {len(genealogy_examples)} genealogy examples")
        else:
            source_stats["genealogy"] = 0

        # Source 3: Negative examples (default: included)
        if options["include_negative_examples"]:
            self.stdout.write("\n🔧 Generating negative examples for Dutch abbreviations...")
            negative_examples = self._generate_negative_examples(options["negative_examples_count"])
            all_examples.extend(negative_examples)
            source_stats["negative_examples"] = len(negative_examples)
            self.stdout.write(f"  ✅ Generated {len(negative_examples)} negative examples")
        else:
            source_stats["negative_examples"] = 0

        # Shuffle and split data
        self.stdout.write(f"\n🔀 Shuffling and splitting {len(all_examples)} total examples...")
        random.shuffle(all_examples)

        splits = self._create_splits(all_examples, options["train_ratio"], options["dev_ratio"], options["test_ratio"])

        # Save datasets
        self._save_training_data(splits, output_dir, options)

        # Save metadata
        self._save_metadata(source_stats, splits, output_dir, options)

        self.stdout.write(
            self.style.SUCCESS(
                f"\n✅ COMPLETE: Generated complete training dataset\n"
                f"📁 Output: {output_dir}\n"
                f"📊 Total examples: {len(all_examples)}\n"
                f"   CoNLL: {source_stats['conll']}\n"
                f"   Genealogy: {source_stats['genealogy']}\n"
                f"   Negative: {source_stats['negative_examples']}\n"
                f"🎯 Splits: {len(splits['train'])} train, {len(splits['dev'])} dev, {len(splits['test'])} test"
            )
        )

    def _load_genealogy_data(self, sources_dir: Path) -> list[dict]:
        """Generate genealogy data from OCR chunks using existing command"""

        # Check if we have any documents to process
        document_count = Document.objects.filter(text_chunks__isnull=False).count()
        if document_count == 0:
            self.stdout.write("  ⚠️  No documents with text chunks found")
            return []

        # Use existing genealogy command to generate data
        genealogy_cmd = GenealogyCommand()

        # Create temporary directory for genealogy data
        temp_genealogy_dir = sources_dir / "genealogy_temp"
        temp_genealogy_dir.mkdir(exist_ok=True)

        # Generate genealogy training data
        genealogy_cmd.handle(
            output_dir=str(temp_genealogy_dir),
            format="json",
            min_chunk_length=30,
            include_headers=True,
        )

        # Load generated examples
        examples = []
        for json_file in temp_genealogy_dir.rglob("*.json"):
            if json_file.name.endswith("_samples.txt") or json_file.name == "training_stats.json":
                continue

            with open(json_file, encoding="utf-8") as f:
                file_examples = json.load(f)
                if isinstance(file_examples, list):
                    examples.extend(file_examples)

        return examples

    def _generate_negative_examples(self, count: int) -> list[dict]:
        """Generate negative examples for Dutch genealogical abbreviations"""

        generator = DutchGenealogyNegativeGenerator()

        # Generate proportional amounts of each type
        generator.generate_birth_order_examples(int(count * 0.08))
        generator.generate_lifecycle_examples(int(count * 0.08))
        generator.generate_relationship_examples(int(count * 0.08))
        generator.generate_witness_examples(int(count * 0.08))
        generator.generate_occupation_examples(int(count * 0.08))
        generator.generate_birth_baptism_combo_examples(int(count * 0.08))
        generator.generate_address_examples(int(count * 0.05))
        generator.generate_complex_entries(int(count * 0.05))
        generator.generate_unnamed_child_burial_examples(int(count * 0.08))
        generator.generate_source_citation_examples(int(count * 0.15))
        generator.generate_complex_multi_person_examples(int(count * 0.17))

        return generator.examples

    def _create_splits(self, examples: list, train_ratio: float, dev_ratio: float, test_ratio: float) -> dict:
        """Create train/dev/test splits"""

        n_examples = len(examples)
        train_end = int(n_examples * train_ratio)
        dev_end = train_end + int(n_examples * dev_ratio)

        return {
            "train": examples[:train_end],
            "dev": examples[train_end:dev_end],
            "test": examples[dev_end:],
        }

    def _save_training_data(self, splits: dict, output_dir: Path, options: dict):
        """Save training data in both JSON and CoNLL formats"""

        for split_name, examples in splits.items():
            if not examples:
                continue

            # Save JSON format
            json_file = output_dir / split_name / f"{split_name}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)

            # Save CoNLL format
            conll_file = output_dir / split_name / f"{split_name}.conll"
            with open(conll_file, "w", encoding="utf-8") as f:
                for example in examples:
                    # Write metadata
                    f.write(f"# Source: {example['metadata'].get('source', 'unknown')}\n")
                    f.write(f"# Text: {example['text'][:100]}...\n")
                    f.write("\n")

                    # Write tokens and labels
                    for token, label in zip(example["tokens"], example["labels"], strict=False):
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")

    def _save_metadata(self, source_stats: dict, splits: dict, output_dir: Path, options: dict):
        """Save comprehensive metadata about the training data"""

        metadata = {
            "generation_timestamp": datetime.now(UTC).isoformat(),
            "command": "generate_complete_training_data",
            "parameters": {
                "train_ratio": options["train_ratio"],
                "dev_ratio": options["dev_ratio"],
                "test_ratio": options["test_ratio"],
                "negative_examples_count": options["negative_examples_count"],
                "seed": options["seed"],
            },
            "sources": {
                "conll_2002_dutch": {
                    "examples": source_stats["conll"],
                    "description": "General Dutch NER from CoNLL-2002 dataset",
                },
                "genealogy_chunks": {
                    "examples": source_stats["genealogy"],
                    "description": "Genealogy-specific entities from OCR text chunks",
                },
                "negative_examples": {
                    "examples": source_stats["negative_examples"],
                    "description": "Synthetic examples for Dutch genealogical abbreviations",
                },
            },
            "data_splits": {
                "train_examples": len(splits["train"]),
                "dev_examples": len(splits["dev"]),
                "test_examples": len(splits["test"]),
                "total_examples": sum(len(split) for split in splits.values()),
            },
        }

        metadata_file = output_dir / "training_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
