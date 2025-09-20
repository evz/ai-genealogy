#!/usr/bin/env python3
"""
Training Data Quality Evaluation Tool

Analyzes training data for the genealogy NER pipeline to identify:
- Entity distribution and balance
- Data quality issues (inconsistent labeling, problematic patterns)
- Dataset composition (genealogy vs CoNLL)
- Potential training issues

Usage:
    python manage.py evaluate_training_data [--data-dir training_data_merged] [--verbose]
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Evaluate training data quality for NER model"

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-dir",
            type=str,
            default="training_data_merged",
            help="Directory containing training data (default: training_data_merged)",
        )
        parser.add_argument("--verbose", action="store_true", help="Show detailed analysis and examples")
        parser.add_argument(
            "--export-issues", action="store_true", help="Export problematic examples to files for manual review"
        )

    def handle(self, *args, **options):
        data_dir = Path(options["data_dir"])
        verbose = options["verbose"]
        export_issues = options["export_issues"]

        if not data_dir.exists():
            self.stderr.write(f"❌ Data directory not found: {data_dir}")
            return

        self.stdout.write("🔍 TRAINING DATA QUALITY EVALUATION")
        self.stdout.write("=" * 60)

        # Load and analyze each split
        analyses = {}
        for split in ["train", "dev", "test"]:
            split_path = data_dir / split / f"{split}.conll"
            if split_path.exists():
                self.stdout.write(f"\n📊 Analyzing {split.upper()} split...")
                analyses[split] = self.analyze_split(split_path, verbose)
            else:
                self.stdout.write(f"⚠️  {split.upper()} split not found: {split_path}")

        # Overall analysis
        if analyses:
            self.stdout.write("\n📈 OVERALL ANALYSIS")
            self.stdout.write("-" * 30)
            self.analyze_overall(analyses, verbose)

        # Export issues if requested
        if export_issues and analyses:
            self.export_quality_issues(data_dir, analyses)

        self.stdout.write("\n✅ Training data evaluation complete!")

    def analyze_split(self, file_path: Path, verbose: bool) -> dict:
        """Analyze a single training split (train/dev/test)"""

        # Parse CoNLL format
        sentences, entity_stats, issues = self.parse_conll_file(file_path)

        analysis = {
            "total_sentences": len(sentences),
            "total_tokens": sum(len(s["tokens"]) for s in sentences),
            "entity_stats": entity_stats,
            "issues": issues,
            "sentences": sentences if verbose else [],  # Only store if verbose
        }

        # Print analysis
        self.print_split_analysis(analysis, verbose)

        return analysis

    def parse_conll_file(self, file_path: Path) -> tuple[list[dict], dict, dict]:
        """Parse CoNLL file and extract statistics"""

        sentences = []
        entity_stats = defaultdict(int)
        issues = {
            "inconsistent_bio": [],
            "single_char_entities": [],
            "suspicious_entities": [],
            "genealogy_vs_conll": {"genealogy": 0, "conll": 0},
        }

        current_sentence = {"tokens": [], "labels": [], "source": None, "document": None}
        current_entities = []

        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Handle comments (source/document info)
                if line.startswith("# Source:"):
                    current_sentence["source"] = line.replace("# Source:", "").strip()
                elif line.startswith("# Document:"):
                    current_sentence["document"] = line.replace("# Document:", "").strip()
                elif line == "":
                    # End of sentence
                    if current_sentence["tokens"]:
                        # Analyze sentence
                        self.analyze_sentence(current_sentence, entity_stats, issues, current_entities)
                        sentences.append(current_sentence)

                        # Reset for next sentence
                        current_sentence = {"tokens": [], "labels": [], "source": None, "document": None}
                        current_entities = []
                elif not line.startswith("#"):
                    # Token line
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        token, label = parts[0], parts[1]
                        current_sentence["tokens"].append(token)
                        current_sentence["labels"].append(label)

                        # Track entity spans
                        if label.startswith("B-"):
                            current_entities.append(
                                {
                                    "type": label[2:],
                                    "tokens": [token],
                                    "start": len(current_sentence["tokens"]) - 1,
                                    "line_num": line_num,
                                }
                            )
                        elif label.startswith("I-") and current_entities:
                            current_entities[-1]["tokens"].append(token)

        # Handle last sentence
        if current_sentence["tokens"]:
            self.analyze_sentence(current_sentence, entity_stats, issues, current_entities)
            sentences.append(current_sentence)

        return sentences, dict(entity_stats), issues

    def analyze_sentence(self, sentence: dict, entity_stats: dict, issues: dict, entities: list[dict]):
        """Analyze a single sentence for quality issues"""

        # Count data sources
        if sentence.get("source") == "CoNLL-2002-Dutch":
            issues["genealogy_vs_conll"]["conll"] += 1
        elif sentence.get("source") == "genealogy":
            issues["genealogy_vs_conll"]["genealogy"] += 1

        # Analyze entities
        for entity in entities:
            entity_type = entity["type"]
            entity_text = " ".join(entity["tokens"])

            entity_stats[entity_type] += 1

            # Check for single character entities
            if len(entity_text.strip()) <= 1:
                issues["single_char_entities"].append(
                    {"text": entity_text, "type": entity_type, "line_num": entity["line_num"]}
                )

            # Check for suspicious genealogy-specific entities
            if entity_type in ["GENEALOGY_ID", "FAMILY_GROUP"]:
                if not self.is_valid_genealogy_entity(entity_text, entity_type):
                    issues["suspicious_entities"].append(
                        {
                            "text": entity_text,
                            "type": entity_type,
                            "line_num": entity["line_num"],
                            "reason": "Invalid genealogy pattern",
                        }
                    )

        # Check BIO consistency
        bio_issues = self.check_bio_consistency(sentence["labels"])
        if bio_issues:
            issues["inconsistent_bio"].extend(
                [
                    {"sentence_tokens": sentence["tokens"], "labels": sentence["labels"], "issue": issue}
                    for issue in bio_issues
                ]
            )

    def is_valid_genealogy_entity(self, text: str, entity_type: str) -> bool:
        """Check if genealogy entity follows expected patterns (including OCR corruption)"""

        if entity_type == "GENEALOGY_ID":
            # Accept both clean and OCR-corrupted genealogy ID patterns
            # Clean: II.1.a, VII.4.b
            # OCR-corrupted: VIIL.4.d, VIl.1.e, l.1.a, (IX.2.a):, etc.

            # Remove surrounding punctuation that might be picked up by regex
            cleaned_text = text.strip("():;.,")

            # Basic pattern: Roman-numeralish + dot + number + dot + letter
            # Allow OCR corruption: l instead of I, mixing case, extra letters
            pattern = r"^[IVXLCDMilvxlcdm]+\.\d+\.[a-zA-Z]$"

            # Also allow single lowercase 'l' (common OCR error for 'I')
            if re.match(r"^l\.\d+\.[a-z]$", cleaned_text):
                return True

            return bool(re.match(pattern, cleaned_text))

        if entity_type == "FAMILY_GROUP":
            # Should contain "Children of" or "Kinderen van"
            return any(
                phrase in text.lower()
                for phrase in ["children of", "kinderen van", "zoon van", "daughter of", "dochter van"]
            )

        return True

    def check_bio_consistency(self, labels: list[str]) -> list[str]:
        """Check for BIO tagging inconsistencies"""
        issues = []

        for i, label in enumerate(labels):
            if label.startswith("I-"):
                entity_type = label[2:]

                # I- should follow B- or I- of same type
                if i == 0 or not (labels[i - 1] == f"B-{entity_type}" or labels[i - 1] == f"I-{entity_type}"):
                    issues.append(f"I-{entity_type} at position {i} without proper B- prefix")

        return issues

    def print_split_analysis(self, analysis: dict, verbose: bool):
        """Print analysis results for a split"""

        self.stdout.write(f"  📝 Sentences: {analysis['total_sentences']:,}")
        self.stdout.write(f"  🔤 Tokens: {analysis['total_tokens']:,}")

        # Entity distribution
        if analysis["entity_stats"]:
            self.stdout.write("  🏷️  Entity Distribution:")
            total_entities = sum(analysis["entity_stats"].values())
            for entity_type, count in sorted(analysis["entity_stats"].items()):
                percentage = (count / total_entities) * 100 if total_entities > 0 else 0
                self.stdout.write(f"     {entity_type}: {count:,} ({percentage:.1f}%)")
        else:
            self.stdout.write("  🏷️  No entities found")

        # Issues summary
        issues = analysis["issues"]
        total_issues = (
            len(issues["inconsistent_bio"]) + len(issues["single_char_entities"]) + len(issues["suspicious_entities"])
        )

        if total_issues > 0:
            self.stdout.write(f"  ⚠️  Quality Issues: {total_issues}")
            if issues["inconsistent_bio"]:
                self.stdout.write(f"     BIO inconsistencies: {len(issues['inconsistent_bio'])}")
            if issues["single_char_entities"]:
                self.stdout.write(f"     Single-char entities: {len(issues['single_char_entities'])}")
            if issues["suspicious_entities"]:
                self.stdout.write(f"     Suspicious entities: {len(issues['suspicious_entities'])}")
        else:
            self.stdout.write("  ✅ No major quality issues detected")

        # Data composition
        genealogy_count = issues["genealogy_vs_conll"]["genealogy"]
        conll_count = issues["genealogy_vs_conll"]["conll"]
        total_sentences = genealogy_count + conll_count

        if total_sentences > 0:
            self.stdout.write("  📊 Data Composition:")
            self.stdout.write(
                f"     Genealogy-specific: {genealogy_count:,} ({genealogy_count/total_sentences*100:.1f}%)"
            )
            self.stdout.write(f"     CoNLL-2002-Dutch: {conll_count:,} ({conll_count/total_sentences*100:.1f}%)")

        # Verbose details
        if verbose and total_issues > 0:
            self.print_verbose_issues(issues)

    def print_verbose_issues(self, issues: dict):
        """Print detailed issue examples"""

        self.stdout.write("  🔍 Issue Details:")

        # Show some single character entities
        if issues["single_char_entities"]:
            self.stdout.write("     Single-char entities (first 5):")
            for issue in issues["single_char_entities"][:5]:
                self.stdout.write(f"       '{issue['text']}' ({issue['type']}) at line {issue['line_num']}")

        # Show suspicious entities
        if issues["suspicious_entities"]:
            self.stdout.write("     Suspicious entities (first 5):")
            for issue in issues["suspicious_entities"][:5]:
                self.stdout.write(f"       '{issue['text']}' ({issue['type']}) - {issue['reason']}")

        # Show BIO issues
        if issues["inconsistent_bio"]:
            self.stdout.write("     BIO inconsistencies (first 3):")
            for issue in issues["inconsistent_bio"][:3]:
                self.stdout.write(f"       {issue['issue']}")

    def analyze_overall(self, analyses: dict, verbose: bool):
        """Analyze patterns across all splits"""

        # Combine entity statistics
        all_entities = defaultdict(int)
        all_issues = defaultdict(int)

        for _split_name, analysis in analyses.items():
            for entity_type, count in analysis["entity_stats"].items():
                all_entities[entity_type] += count

            # Count issues
            issues = analysis["issues"]
            all_issues["bio_inconsistencies"] += len(issues["inconsistent_bio"])
            all_issues["single_char_entities"] += len(issues["single_char_entities"])
            all_issues["suspicious_entities"] += len(issues["suspicious_entities"])

        # Print overall entity distribution
        total_entities = sum(all_entities.values())
        if total_entities > 0:
            self.stdout.write("🏷️  Overall Entity Distribution:")
            for entity_type, count in sorted(all_entities.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_entities) * 100
                self.stdout.write(f"   {entity_type}: {count:,} ({percentage:.1f}%)")

        # Print quality assessment
        total_issues = sum(all_issues.values())
        self.stdout.write(f"\n⚠️  Total Quality Issues: {total_issues}")

        if total_issues > 0:
            for issue_type, count in all_issues.items():
                if count > 0:
                    self.stdout.write(f"   {issue_type.replace('_', ' ').title()}: {count}")

        # Recommendations
        self.print_recommendations(all_entities, all_issues)

    def print_recommendations(self, entities: dict, issues: dict):
        """Print recommendations based on analysis"""

        self.stdout.write("\n💡 RECOMMENDATIONS")
        self.stdout.write("-" * 20)

        total_entities = sum(entities.values())

        # Entity balance recommendations
        if total_entities > 0:
            genealogy_entities = sum(
                count for entity_type, count in entities.items() if entity_type in ["GENEALOGY_ID", "FAMILY_GROUP"]
            )
            general_entities = total_entities - genealogy_entities

            if genealogy_entities < total_entities * 0.05:  # Less than 5%
                self.stdout.write("📈 Consider generating more genealogy-specific training examples")

            # Check for imbalanced entity types
            max_count = max(entities.values()) if entities else 0
            min_count = min(entities.values()) if entities else 0

            if max_count > min_count * 10:  # 10x imbalance
                self.stdout.write("⚖️  Entity types are imbalanced - consider balancing the dataset")

        # Quality recommendations
        if issues["single_char_entities"] > 0:
            self.stdout.write("🔤 Remove or fix single-character entity annotations")

        if issues["bio_inconsistencies"] > 0:
            self.stdout.write("🏷️  Fix BIO tagging inconsistencies before training")

        if issues["suspicious_entities"] > 0:
            self.stdout.write("🔍 Review suspicious genealogy entities - may indicate pattern matching errors")

        if sum(issues.values()) == 0:
            self.stdout.write("✅ Training data quality looks good for training!")

    def export_quality_issues(self, data_dir: Path, analyses: dict):
        """Export problematic examples for manual review"""

        issues_dir = data_dir / "quality_issues"
        issues_dir.mkdir(exist_ok=True)

        self.stdout.write(f"\n📤 Exporting quality issues to {issues_dir}/")

        # Collect all issues
        all_issues = {"single_char_entities": [], "suspicious_entities": [], "bio_inconsistencies": []}

        for split_name, analysis in analyses.items():
            issues = analysis["issues"]
            for issue_type in all_issues:
                for issue in issues.get(issue_type, []):
                    issue["split"] = split_name
                    all_issues[issue_type].append(issue)

        # Export each type
        for issue_type, issue_list in all_issues.items():
            if issue_list:
                output_file = issues_dir / f"{issue_type}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(issue_list, f, indent=2, ensure_ascii=False)
                self.stdout.write(f"   {issue_type}: {len(issue_list)} issues → {output_file}")
