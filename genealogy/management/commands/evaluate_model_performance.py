#!/usr/bin/env python3
"""
NER Model Performance Evaluation Tool

Evaluates trained genealogy NER models against test data with detailed metrics:
- Per-entity-type precision, recall, F1 scores
- Confusion matrices and error analysis
- Genealogy-specific vs general entity performance
- Confidence score analysis
- Comparison with regex baseline

Usage:
    python manage.py evaluate_model_performance [--model-path models/genealogy_ner_latest] [--test-data training_data_merged/test] [--confidence-threshold 0.9]
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from django.core.management.base import BaseCommand

from genealogy.ner_extractor import GenealogyNERExtractor


class Command(BaseCommand):
    help = "Evaluate NER model performance against test data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--model-path", type=str, help="Path to trained model directory (auto-detects if not specified)"
        )
        parser.add_argument(
            "--test-data",
            type=str,
            default="training_data_merged/test",
            help="Path to test data directory (default: training_data_merged/test)",
        )
        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.9,
            help="Confidence threshold for predictions (default: 0.9)",
        )
        parser.add_argument("--compare-regex", action="store_true", help="Compare performance with regex baseline")
        parser.add_argument("--export-errors", action="store_true", help="Export misclassified examples for analysis")
        parser.add_argument(
            "--genealogy-only", action="store_true", help="Evaluate only on genealogy-specific test examples"
        )

    def handle(self, *args, **options):
        model_path = options.get("model_path")
        test_data_dir = Path(options["test_data"])
        confidence_threshold = options["confidence_threshold"]
        compare_regex = options["compare_regex"]
        export_errors = options["export_errors"]
        genealogy_only = options["genealogy_only"]

        if not test_data_dir.exists():
            self.stderr.write(f"❌ Test data directory not found: {test_data_dir}")
            return

        self.stdout.write("🧪 NER MODEL PERFORMANCE EVALUATION")
        self.stdout.write("=" * 50)

        # Auto-detect model if not specified
        if not model_path:
            model_path = self.find_latest_model()
            if not model_path:
                self.stderr.write("❌ No trained model found. Specify --model-path or train a model first.")
                return

        self.stdout.write(f"🤖 Model: {model_path}")
        self.stdout.write(f"📊 Test Data: {test_data_dir}")
        self.stdout.write(f"🎯 Confidence Threshold: {confidence_threshold}")

        # Load model
        try:
            model = self.load_model(model_path)
            if not model:
                return
        except Exception as e:
            self.stderr.write(f"❌ Failed to load model: {e}")
            return

        # Load test data
        test_sentences = self.load_test_data(test_data_dir / "test.conll", genealogy_only)
        if not test_sentences:
            self.stderr.write("❌ No test data found")
            return

        self.stdout.write(f"📝 Test Examples: {len(test_sentences)}")

        # Evaluate model
        self.stdout.write("\n🔍 EVALUATING MODEL PERFORMANCE")
        self.stdout.write("-" * 40)

        results = self.evaluate_model(model, test_sentences, confidence_threshold)
        self.print_performance_metrics(results)

        # Compare with regex if requested
        if compare_regex:
            self.stdout.write("\n🔄 COMPARING WITH REGEX BASELINE")
            self.stdout.write("-" * 35)
            regex_results = self.evaluate_regex_baseline(test_sentences)
            self.print_comparison(results, regex_results)

        # Export errors if requested
        if export_errors:
            self.export_error_analysis(results, Path(model_path).parent / "evaluation")

        self.stdout.write("\n✅ Model evaluation complete!")

    def find_latest_model(self) -> str | None:
        """Find the most recently trained model"""
        models_dir = Path("models")
        if not models_dir.exists():
            return None

        # Look for genealogy NER models
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("genealogy_ner_")]

        if not model_dirs:
            return None

        # Return the most recent one (by name timestamp)
        latest_model = sorted(model_dirs, key=lambda x: x.name)[-1]
        return str(latest_model)

    def load_model(self, model_path: str) -> GenealogyNERExtractor | None:
        """Load the trained NER model"""
        try:
            model = GenealogyNERExtractor(model_path=model_path)
            self.stdout.write("✅ Model loaded successfully")
            return model
        except Exception as e:
            self.stderr.write(f"❌ Failed to load model from {model_path}: {e}")
            return None

    def load_test_data(self, test_file: Path, genealogy_only: bool = False) -> list[dict]:
        """Load test data from CoNLL format"""
        sentences = []
        current_sentence = {"tokens": [], "labels": [], "source": None}

        with open(test_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line.startswith("# Source:"):
                    current_sentence["source"] = line.replace("# Source:", "").strip()
                elif line == "":
                    if current_sentence["tokens"]:
                        # Filter for genealogy-only if requested
                        if not genealogy_only or current_sentence["source"] == "genealogy-specific":
                            sentences.append(current_sentence)
                        current_sentence = {"tokens": [], "labels": [], "source": None}
                elif not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        token, label = parts[0], parts[1]
                        current_sentence["tokens"].append(token)
                        current_sentence["labels"].append(label)

        # Handle last sentence
        if current_sentence["tokens"]:
            if not genealogy_only or current_sentence["source"] == "genealogy-specific":
                sentences.append(current_sentence)

        return sentences

    def evaluate_model(self, model: GenealogyNERExtractor, test_sentences: list[dict], confidence_threshold: float) -> dict:
        """Evaluate model performance on test sentences"""

        # Track predictions and ground truth
        predictions = []
        ground_truths = []
        confidence_scores = []
        error_examples = []

        for sentence in test_sentences:
            tokens = sentence["tokens"]
            true_labels = sentence["labels"]
            text = " ".join(tokens)

            try:
                # Get model predictions
                entities = model.extract_entities(text)
                pred_labels = self.entities_to_bio_labels(entities, tokens, confidence_threshold)

                predictions.extend(pred_labels)
                ground_truths.extend(true_labels)

                # Store confidence scores
                for _entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        confidence_scores.append(entity["confidence"])

                # Track errors for detailed analysis
                if pred_labels != true_labels:
                    error_examples.append(
                        {
                            "tokens": tokens,
                            "predicted": pred_labels,
                            "ground_truth": true_labels,
                            "text": text,
                            "source": sentence.get("source", "unknown"),
                        }
                    )

            except Exception as e:
                # Handle prediction failures
                self.stderr.write(f"⚠️  Prediction failed for sentence: {text[:50]}... Error: {e}")
                pred_labels = ["O"] * len(tokens)
                predictions.extend(pred_labels)
                ground_truths.extend(true_labels)

        # Calculate metrics
        results = self.calculate_metrics(predictions, ground_truths)
        results["confidence_scores"] = confidence_scores
        results["error_examples"] = error_examples

        return results

    def entities_to_bio_labels(self, entities: dict, tokens: list[str], confidence_threshold: float) -> list[str]:
        """Convert entity predictions to BIO label sequence"""

        labels = ["O"] * len(tokens)
        text = " ".join(tokens)

        # Create token position mapping
        token_positions = []
        current_pos = 0
        for token in tokens:
            start = text.find(token, current_pos)
            end = start + len(token)
            token_positions.append((start, end))
            current_pos = end

        # Apply entity predictions
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                # Skip low-confidence predictions
                if entity["confidence"] < confidence_threshold:
                    continue

                entity_text = entity["text"]
                entity_start = entity.get("start", 0)
                entity_end = entity.get("end", entity_start + len(entity_text))

                # Find overlapping tokens
                first_token = None
                for i, (token_start, token_end) in enumerate(token_positions):
                    if token_start < entity_end and token_end > entity_start:
                        if first_token is None:
                            first_token = i
                            labels[i] = f"B-{entity_type}"
                        else:
                            labels[i] = f"I-{entity_type}"

        return labels

    def calculate_metrics(self, predictions: list[str], ground_truths: list[str]) -> dict:
        """Calculate precision, recall, F1 for each entity type"""

        # Extract entity spans from BIO labels
        pred_entities = self.extract_entity_spans(predictions)
        true_entities = self.extract_entity_spans(ground_truths)

        # Calculate metrics per entity type
        entity_types = set()
        for entities in [pred_entities, true_entities]:
            for entity_type, _ in entities:
                entity_types.add(entity_type)

        metrics = {}
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        for entity_type in entity_types:
            pred_spans = {span for et, span in pred_entities if et == entity_type}
            true_spans = {span for et, span in true_entities if et == entity_type}

            tp = len(pred_spans & true_spans)
            fp = len(pred_spans - true_spans)
            fn = len(true_spans - pred_spans)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": len(true_spans),
                "predicted": len(pred_spans),
            }

            # Build confusion matrix
            for pred_span in pred_spans:
                if pred_span in true_spans:
                    confusion_matrix[entity_type][entity_type] += 1
                else:
                    confusion_matrix["O"][entity_type] += 1  # False positive

            for true_span in true_spans:
                if true_span not in pred_spans:
                    confusion_matrix[entity_type]["O"] += 1  # False negative

        # Calculate macro averages
        if metrics:
            macro_precision = sum(m["precision"] for m in metrics.values()) / len(metrics)
            macro_recall = sum(m["recall"] for m in metrics.values()) / len(metrics)
            macro_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics)
        else:
            macro_precision = macro_recall = macro_f1 = 0

        return {
            "entity_metrics": metrics,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "confusion_matrix": dict(confusion_matrix),
        }

    def extract_entity_spans(self, labels: list[str]) -> list[tuple[str, tuple[int, int]]]:
        """Extract entity spans from BIO label sequence"""
        entities = []
        current_entity = None
        current_start = None

        for i, label in enumerate(labels):
            if label.startswith("B-"):
                # End previous entity if any
                if current_entity is not None:
                    entities.append((current_entity, (current_start, i - 1)))

                # Start new entity
                current_entity = label[2:]
                current_start = i

            elif label.startswith("I-"):
                # Continue current entity
                if current_entity != label[2:]:
                    # BIO inconsistency - treat as new entity
                    if current_entity is not None:
                        entities.append((current_entity, (current_start, i - 1)))
                    current_entity = label[2:]
                    current_start = i

            # End current entity if any
            elif current_entity is not None:
                entities.append((current_entity, (current_start, i - 1)))
                current_entity = None
                current_start = None

        # Handle entity at end of sequence
        if current_entity is not None:
            entities.append((current_entity, (current_start, len(labels) - 1)))

        return entities

    def print_performance_metrics(self, results: dict):
        """Print detailed performance metrics"""

        entity_metrics = results["entity_metrics"]

        # Overall metrics
        self.stdout.write("🎯 OVERALL PERFORMANCE")
        self.stdout.write(f"   Macro Precision: {results['macro_precision']:.3f}")
        self.stdout.write(f"   Macro Recall:    {results['macro_recall']:.3f}")
        self.stdout.write(f"   Macro F1:        {results['macro_f1']:.3f}")

        # Per-entity metrics
        if entity_metrics:
            self.stdout.write("\n📊 PER-ENTITY PERFORMANCE")
            self.stdout.write(f"{'Entity Type':<15} {'Prec':<6} {'Rec':<6} {'F1':<6} {'Supp':<6} {'Pred':<6}")
            self.stdout.write("-" * 55)

            for entity_type in sorted(entity_metrics.keys()):
                metrics = entity_metrics[entity_type]
                self.stdout.write(
                    f"{entity_type:<15} "
                    f"{metrics['precision']:<6.3f} "
                    f"{metrics['recall']:<6.3f} "
                    f"{metrics['f1']:<6.3f} "
                    f"{metrics['support']:<6d} "
                    f"{metrics['predicted']:<6d}"
                )

        # Confidence analysis
        if results.get("confidence_scores"):
            scores = results["confidence_scores"]
            self.stdout.write("\n🎲 CONFIDENCE ANALYSIS")
            self.stdout.write(f"   Mean Confidence: {sum(scores)/len(scores):.3f}")
            self.stdout.write(f"   Min Confidence:  {min(scores):.3f}")
            self.stdout.write(f"   Max Confidence:  {max(scores):.3f}")

            # Confidence distribution
            high_conf = sum(1 for s in scores if s >= 0.9)
            med_conf = sum(1 for s in scores if 0.7 <= s < 0.9)
            low_conf = sum(1 for s in scores if s < 0.7)
            total = len(scores)

            self.stdout.write(f"   High (≥0.9):     {high_conf:4d} ({high_conf/total*100:.1f}%)")
            self.stdout.write(f"   Medium (0.7-0.9): {med_conf:4d} ({med_conf/total*100:.1f}%)")
            self.stdout.write(f"   Low (<0.7):      {low_conf:4d} ({low_conf/total*100:.1f}%)")

        # Error summary
        if "error_examples" in results:
            errors = results["error_examples"]
            if errors:
                self.stdout.write("\n❌ ERROR ANALYSIS")
                self.stdout.write(f"   Sentences with errors: {len(errors)}")

                # Analyze error types
                genealogy_errors = [e for e in errors if e["source"] == "genealogy-specific"]
                conll_errors = [e for e in errors if e["source"] != "genealogy-specific"]

                if genealogy_errors:
                    self.stdout.write(f"   Genealogy-specific errors: {len(genealogy_errors)}")
                if conll_errors:
                    self.stdout.write(f"   CoNLL dataset errors: {len(conll_errors)}")

    def evaluate_regex_baseline(self, test_sentences: list[dict]) -> dict:
        """Evaluate regex baseline performance"""
        # Import regex patterns
        try:
            from genealogy.patterns import GenealogyPatterns
        except ImportError:
            self.stderr.write("⚠️  Regex patterns not available for comparison")
            return {}

        predictions = []
        ground_truths = []

        for sentence in test_sentences:
            tokens = sentence["tokens"]
            true_labels = sentence["labels"]
            text = " ".join(tokens)

            # Apply regex patterns
            pred_labels = self.apply_regex_patterns(text, tokens)

            predictions.extend(pred_labels)
            ground_truths.extend(true_labels)

        return self.calculate_metrics(predictions, ground_truths)

    def apply_regex_patterns(self, text: str, tokens: list[str]) -> list[str]:
        """Apply regex patterns to extract entities (simplified baseline)"""
        labels = ["O"] * len(tokens)

        # Simple regex patterns for baseline comparison
        patterns = {
            "PERSON_NAME": r"\b[A-Z][a-z]+ (?:van |de |der )?[A-Z][a-z]+\b",
            "DATE": r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|\b\d{4}\b",
            "PLACE": r"\b[A-Z][a-z]+(?:stad|dorf|berg|burg|dam)\b",
            "GENEALOGY_ID": r"\b[IVX]+\.\d+\.[a-z]\b",
        }

        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                start, end = match.span()
                # Map to tokens (simplified)
                for i, token in enumerate(tokens):
                    token_start = text.find(token)
                    if token_start >= start and token_start < end:
                        if labels[i] == "O":  # Don't overwrite existing labels
                            labels[i] = f"B-{entity_type}" if token_start == start else f"I-{entity_type}"

        return labels

    def print_comparison(self, model_results: dict, regex_results: dict):
        """Compare model and regex baseline performance"""

        self.stdout.write("📈 MODEL vs REGEX COMPARISON")
        self.stdout.write(f"{'Metric':<15} {'Model':<8} {'Regex':<8} {'Diff':<8}")
        self.stdout.write("-" * 45)

        # Compare macro metrics
        model_f1 = model_results["macro_f1"]
        regex_f1 = regex_results.get("macro_f1", 0)
        self.stdout.write(f"{'Macro F1':<15} {model_f1:<8.3f} {regex_f1:<8.3f} {model_f1-regex_f1:<+8.3f}")

        model_prec = model_results["macro_precision"]
        regex_prec = regex_results.get("macro_precision", 0)
        self.stdout.write(f"{'Macro Prec':<15} {model_prec:<8.3f} {regex_prec:<8.3f} {model_prec-regex_prec:<+8.3f}")

        model_rec = model_results["macro_recall"]
        regex_rec = regex_results.get("macro_recall", 0)
        self.stdout.write(f"{'Macro Recall':<15} {model_rec:<8.3f} {regex_rec:<8.3f} {model_rec-regex_rec:<+8.3f}")

        # Entity-level comparison
        self.stdout.write("\n📊 PER-ENTITY COMPARISON (F1 Scores)")
        self.stdout.write(f"{'Entity':<15} {'Model':<8} {'Regex':<8} {'Improvement':<12}")
        self.stdout.write("-" * 50)

        model_entities = model_results.get("entity_metrics", {})
        regex_entities = regex_results.get("entity_metrics", {})

        all_entities = set(model_entities.keys()) | set(regex_entities.keys())
        for entity_type in sorted(all_entities):
            model_f1 = model_entities.get(entity_type, {}).get("f1", 0)
            regex_f1 = regex_entities.get(entity_type, {}).get("f1", 0)
            improvement = model_f1 - regex_f1

            self.stdout.write(f"{entity_type:<15} {model_f1:<8.3f} {regex_f1:<8.3f} {improvement:<+12.3f}")

    def export_error_analysis(self, results: dict, output_dir: Path):
        """Export detailed error analysis for manual review"""

        output_dir.mkdir(parents=True, exist_ok=True)

        # Export error examples
        if "error_examples" in results:
            error_file = output_dir / "error_examples.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(results["error_examples"], f, indent=2, ensure_ascii=False)

            self.stdout.write(f"📤 Error examples exported to {error_file}")

        # Export confusion matrix
        if "confusion_matrix" in results:
            confusion_file = output_dir / "confusion_matrix.json"
            with open(confusion_file, "w", encoding="utf-8") as f:
                json.dump(results["confusion_matrix"], f, indent=2, ensure_ascii=False)

            self.stdout.write(f"📤 Confusion matrix exported to {confusion_file}")

        # Export detailed metrics
        metrics_file = output_dir / "detailed_metrics.json"
        exportable_results = {k: v for k, v in results.items() if k not in ["error_examples"]}  # Exclude large data
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(exportable_results, f, indent=2, ensure_ascii=False)

        self.stdout.write(f"📤 Detailed metrics exported to {metrics_file}")
