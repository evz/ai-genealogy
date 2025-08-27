"""
Train a genealogical NER model using the generated training data.

This command fine-tunes a pre-trained multilingual transformer model
on the genealogical entity extraction task.

Requirements:
    pip install torch transformers datasets scikit-learn numpy
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

# Check if ML packages are available
try:
    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    HAS_ML_PACKAGES = True
except ImportError as e:
    HAS_ML_PACKAGES = False
    MISSING_PACKAGE = str(e)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Train a genealogical NER model using generated training data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--check-dependencies",
            action="store_true",
            help="Check if required ML packages are installed and exit",
        )
        parser.add_argument(
            "--training-data-dir",
            type=str,
            help="Directory containing the training data (e.g., training_data/v1_20250823_120000)",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default="GroNLP/bert-base-dutch-cased",
            help="Base model to fine-tune (default: Dutch BERT)",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="models",
            help="Directory to save the trained model (default: models)",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=8,
            help="Number of training epochs (default: 8)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            help="Training batch size (default: 16)",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=3e-5,
            help="Learning rate (default: 3e-5)",
        )
        parser.add_argument(
            "--max-length",
            type=int,
            default=1024,
            help="Maximum sequence length (default: 1024)",
        )
        parser.add_argument(
            "--eval-steps",
            type=int,
            default=500,
            help="Evaluation frequency in steps (default: 500)",
        )
        parser.add_argument(
            "--save-steps",
            type=int,
            default=1000,
            help="Model saving frequency in steps (default: 1000)",
        )
        parser.add_argument(
            "--warmup-ratio",
            type=float,
            default=0.1,
            help="Warmup ratio for learning rate scheduler (default: 0.1)",
        )
        parser.add_argument(
            "--weight-decay",
            type=float,
            default=0.01,
            help="Weight decay for optimizer (default: 0.01)",
        )
        parser.add_argument(
            "--gradient-accumulation-steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass (default: 1)",
        )
        parser.add_argument(
            "--lr-scheduler-type",
            type=str,
            default="cosine",
            choices=["linear", "cosine", "polynomial"],
            help="Learning rate scheduler type (default: cosine)",
        )
        parser.add_argument(
            "--early-stopping-patience",
            type=int,
            default=3,
            help="Early stopping patience in evaluation steps (default: 3)",
        )

    def handle(self, *args, **options):  # noqa: ARG002
        # Check dependencies first
        if options.get("check_dependencies"):
            self._check_dependencies()
            return

        if not HAS_ML_PACKAGES:
            self._show_dependency_error()
            return

        if not options.get("training_data_dir"):
            raise CommandError("--training-data-dir is required for training")

        training_data_dir = Path(options["training_data_dir"])
        if not training_data_dir.exists():
            raise CommandError(f"Training data directory not found: {training_data_dir}")

        # Create output directory with timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model_output_dir = Path(options["output_dir"]) / f"genealogy_ner_{timestamp}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write(f"Training directory: {training_data_dir}")
        self.stdout.write(f"Model output directory: {model_output_dir}")

        # Load and validate training data
        train_data, dev_data, test_data, label_list = self._load_training_data(training_data_dir)

        if not train_data:
            raise CommandError("No training data found in the specified directory")

        self.stdout.write(
            f"Data loaded - Train: {len(train_data)} examples, "
            f"Dev: {len(dev_data)} examples, Test: {len(test_data)} examples"
        )
        self.stdout.write(f"Entity types: {', '.join(label_list)}")

        # Initialize tokenizer and model
        tokenizer, model = self._initialize_model(options["model_name"], label_list)

        # Prepare datasets
        train_dataset = self._prepare_dataset(train_data, tokenizer, label_list, options["max_length"])
        eval_dataset = self._prepare_dataset(dev_data, tokenizer, label_list, options["max_length"])
        test_dataset = (
            self._prepare_dataset(test_data, tokenizer, label_list, options["max_length"]) if test_data else None
        )

        # Setup training arguments with improved parameters
        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            num_train_epochs=options["epochs"],
            per_device_train_batch_size=options["batch_size"],
            per_device_eval_batch_size=options["batch_size"],
            gradient_accumulation_steps=options["gradient_accumulation_steps"],
            learning_rate=options["learning_rate"],
            weight_decay=options["weight_decay"],
            warmup_ratio=options["warmup_ratio"],
            lr_scheduler_type=options["lr_scheduler_type"],
            logging_dir=str(model_output_dir / "logs"),
            logging_steps=50,  # More frequent logging
            eval_steps=options["eval_steps"],
            eval_strategy="steps",
            save_steps=options["save_steps"],
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            # Improved evaluation and saving
            save_total_limit=3,  # Keep only 3 best models
            # Better precision settings
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            dataloader_num_workers=2,  # Parallel data loading
            remove_unused_columns=False,  # Keep all columns for debugging
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,
        )

        # Initialize trainer
        trainer = self._initialize_trainer(
            model,
            tokenizer,
            training_args,
            train_dataset,
            eval_dataset,
            label_list,
            options,
        )

        # Train the model
        self.stdout.write("Starting training...")
        train_result = trainer.train()

        # Save the model and tokenizer
        trainer.save_model()
        tokenizer.save_pretrained(model_output_dir)

        # Evaluate on test set if available
        test_results = None
        if test_dataset:
            self.stdout.write("Evaluating on test set...")
            test_results = trainer.evaluate(eval_dataset=test_dataset)

        # Save training configuration and results
        self._save_training_info(
            model_output_dir,
            options,
            train_result,
            test_results,
            label_list,
            len(train_data),
            len(dev_data),
            len(test_data) if test_data else 0,
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Training completed!\n"
                f"Model saved to: {model_output_dir}\n"
                f"Training loss: {train_result.training_loss:.4f}\n"
                f"Training time: {train_result.metrics.get('train_runtime', 0):.1f} seconds"
            )
        )

        if test_results:
            self.stdout.write(f"Test F1 score: {test_results.get('eval_f1', 0):.4f}")

    def _check_dependencies(self):
        """Check if all required packages are installed"""
        if HAS_ML_PACKAGES:
            self.stdout.write(
                self.style.SUCCESS(
                    "✓ All required ML packages are installed:\n"
                    f"  - torch: {torch.__version__}\n"
                    f"  - transformers: {torch.__version__}\n"  # Assuming transformers version
                    "  - datasets: available\n"
                    "  - numpy: available\n"
                    "  - scikit-learn: available"
                )
            )
        else:
            self._show_dependency_error()

    def _show_dependency_error(self):
        """Show helpful error message for missing dependencies"""
        self.stdout.write(
            self.style.ERROR(
                "❌ Required ML packages are not installed.\n\n"
                "To use the NER training functionality, install the required packages:\n\n"
                "pip install torch transformers datasets scikit-learn numpy\n\n"
                "Note: This will install ~2GB of packages including PyTorch.\n"
                "You can still use the training data generation without these packages.\n\n"
                "Use --check-dependencies to verify installation."
            )
        )

    def _load_training_data(self, data_dir: Path) -> tuple[list, list, list, list]:
        """Load training data from CoNLL format files"""
        train_data = self._load_conll_files(data_dir / "train")
        dev_data = self._load_conll_files(data_dir / "dev")
        test_data = self._load_conll_files(data_dir / "test")

        # Extract all unique labels
        all_labels = set()
        for dataset in [train_data, dev_data, test_data]:
            for example in dataset:
                all_labels.update(example["labels"])

        # Create label list with proper ordering (O first, then B-, then I-)
        label_list = ["O"]
        entity_types = set()
        for label in all_labels:
            if label.startswith(("B-", "I-")):
                entity_types.add(label[2:])

        for entity_type in sorted(entity_types):
            label_list.extend([f"B-{entity_type}", f"I-{entity_type}"])

        return train_data, dev_data, test_data, label_list

    def _load_conll_files(self, split_dir: Path) -> list[dict]:
        """Load all CoNLL files from a split directory"""
        if not split_dir.exists():
            return []

        all_examples = []
        conll_files = sorted(split_dir.glob("*.conll"))

        for conll_file in conll_files:
            examples = self._parse_conll_file(conll_file)
            all_examples.extend(examples)

        return all_examples

    def _parse_conll_file(self, file_path: Path) -> list[dict]:
        """Parse a single CoNLL format file"""
        examples = []
        current_tokens = []
        current_labels = []

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()

                # Skip comments and empty lines between examples
                if line.startswith("#") or (not line and not current_tokens):
                    continue

                # End of example
                if not line and current_tokens:
                    examples.append(
                        {
                            "tokens": current_tokens.copy(),
                            "labels": current_labels.copy(),
                        }
                    )
                    current_tokens.clear()
                    current_labels.clear()
                    continue

                # Parse token and label
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[1])

        # Handle final example if file doesn't end with empty line
        if current_tokens:
            examples.append({"tokens": current_tokens, "labels": current_labels})

        return examples

    def _initialize_model(self, model_name: str, label_list: list[str]):
        """Initialize tokenizer and model"""
        self.stdout.write(f"Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create label mapping
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for label, i in label2id.items()}

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label_list),
            label2id=label2id,
            id2label=id2label,
        )

        return tokenizer, model

    def _prepare_dataset(self, examples: list[dict], tokenizer, label_list: list[str], max_length: int):
        """Convert examples to HuggingFace dataset format"""
        if not examples:
            return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})

        label2id = {label: i for i, label in enumerate(label_list)}

        # Tokenize and align labels
        tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for example in examples:
            # Tokenize with word-level tokenization tracking
            tokenized = tokenizer(
                example["tokens"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None,
            )

            # Align labels with subword tokens
            labels = []
            word_ids = tokenized.word_ids()
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens get -100 (ignored in loss)
                    labels.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word gets the label
                    if word_idx < len(example["labels"]):
                        labels.append(label2id[example["labels"][word_idx]])
                    else:
                        labels.append(label2id["O"])
                else:
                    # Subsequent subwords get -100 (ignored in loss)
                    labels.append(-100)
                previous_word_idx = word_idx

            tokenized_inputs["input_ids"].append(tokenized["input_ids"])
            tokenized_inputs["attention_mask"].append(tokenized["attention_mask"])
            tokenized_inputs["labels"].append(labels)

        return Dataset.from_dict(tokenized_inputs)

    def _initialize_trainer(
        self,
        model,
        tokenizer,
        training_args,
        train_dataset,
        eval_dataset,
        label_list,
        options,
    ):
        """Initialize HuggingFace trainer"""
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for p, label_val in zip(prediction, label, strict=False) if label_val != -100]
                for prediction, label in zip(predictions, labels, strict=False)
            ]
            true_labels = [
                [label_list[label_val] for p, label_val in zip(prediction, label, strict=False) if label_val != -100]
                for prediction, label in zip(predictions, labels, strict=False)
            ]

            # Flatten for sklearn metrics
            flat_true_labels = [label for sublist in true_labels for label in sublist]
            flat_predictions = [pred for sublist in true_predictions for pred in sublist]

            # Calculate overall metrics
            f1_weighted = f1_score(flat_true_labels, flat_predictions, average="weighted")
            f1_macro = f1_score(flat_true_labels, flat_predictions, average="macro")

            # Calculate per-entity-type F1 scores for important entities
            metrics = {"f1": f1_weighted, "f1_macro": f1_macro}

            # Get entity-specific F1 scores
            entity_types = [
                "GENEALOGY_ID",
                "PERSON_NAME",
                "DATE",
                "PLACE",
                "FAMILY_GROUP",
            ]
            for entity_type in entity_types:
                entity_labels = [f"B-{entity_type}", f"I-{entity_type}"]
                entity_true = [1 if label in entity_labels else 0 for label in flat_true_labels]
                entity_pred = [1 if label in entity_labels else 0 for label in flat_predictions]

                if sum(entity_true) > 0:  # Only calculate if entity exists in data
                    entity_f1 = f1_score(entity_true, entity_pred, average="binary", zero_division=0)
                    metrics[f"f1_{entity_type.lower()}"] = entity_f1

            return metrics

        # Add early stopping callback
        early_stopping = EarlyStoppingCallback(early_stopping_patience=options["early_stopping_patience"])

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping],
        )

    def _save_training_info(
        self,
        output_dir: Path,
        options: dict,
        train_result,
        test_results,
        label_list: list[str],
        train_size: int,
        dev_size: int,
        test_size: int,
    ):
        """Save training configuration and results"""
        training_info = {
            "model_info": {
                "base_model": options["model_name"],
                "training_timestamp": datetime.now(UTC).isoformat(),
                "entity_types": [label[2:] for label in label_list if label.startswith("B-")],
                "label_list": label_list,
            },
            "training_config": {
                "epochs": options["epochs"],
                "batch_size": options["batch_size"],
                "learning_rate": options["learning_rate"],
                "max_length": options["max_length"],
                "warmup_ratio": options["warmup_ratio"],
                "weight_decay": options["weight_decay"],
            },
            "data_info": {
                "train_examples": train_size,
                "dev_examples": dev_size,
                "test_examples": test_size,
                "training_data_dir": str(options["training_data_dir"]),
            },
            "training_results": {
                "training_loss": train_result.training_loss,
                "training_runtime": train_result.metrics.get("train_runtime", 0),
                "training_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            },
        }

        if test_results:
            training_info["test_results"] = test_results

        info_file = output_dir / "training_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)

        # Also save a simple model card
        model_card = f"""# Genealogy NER Model

This model was trained for genealogical named entity recognition on Dutch family history texts.

## Entity Types
{chr(10).join(f"- {entity}" for entity in training_info["model_info"]["entity_types"])}

## Training Info
- Base model: {options["model_name"]}
- Training examples: {train_size:,}
- Training time: {train_result.metrics.get("train_runtime", 0):.1f} seconds
- Final training loss: {train_result.training_loss:.4f}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
model = AutoModelForTokenClassification.from_pretrained("{output_dir}")
```
"""

        with open(output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(model_card)
