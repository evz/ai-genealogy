"""
Neural network-based genealogical entity extraction using trained models.

This module provides a production-ready interface to use trained NER models
for extracting genealogical entities from text chunks.
"""

import logging
from pathlib import Path

# Check if ML packages are available
try:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    HAS_ML_PACKAGES = True
except ImportError:
    HAS_ML_PACKAGES = False


logger = logging.getLogger(__name__)


class GenealogyNERExtractor:
    """Neural network-based genealogical entity extractor"""

    def __init__(self, model_path: str | None = None):
        """
        Initialize the NER extractor

        Args:
            model_path: Path to trained model directory. If None, will look for default location.
        """
        if not HAS_ML_PACKAGES:
            raise ImportError(
                "ML packages not installed. Install with: " "pip install torch transformers datasets scikit-learn numpy"
            )

        self.model_path = self._find_model_path(model_path)
        self.tokenizer = None
        self.model = None
        self.label_list = None
        self._loaded = False

    def _find_model_path(self, model_path: str | None) -> Path:
        """Find the model path, preferring the most recent if not specified"""
        if model_path:
            path = Path(model_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Look for models in default location
        models_dir = Path("models")
        if not models_dir.exists():
            raise FileNotFoundError(
                "No models directory found. Train a model first using: " "python manage.py train_genealogy_ner"
            )

        # Find the most recent genealogy_ner model
        model_dirs = sorted(models_dir.glob("genealogy_ner_*"), reverse=True)
        if not model_dirs:
            raise FileNotFoundError("No trained genealogy NER models found in models/ directory")

        return model_dirs[0]

    def load_model(self):
        """Load the tokenizer and model"""
        if self._loaded:
            return

        logger.info(f"Loading genealogy NER model from {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)

            # Extract label list from model config
            self.label_list = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

            # Set model to evaluation mode
            self.model.eval()

            self._loaded = True
            logger.info(f"Model loaded successfully. Entity types: {self._get_entity_types()}")

        except Exception as e:
            logger.exception(f"Failed to load model from {self.model_path}: {e}")
            raise

    def _get_entity_types(self) -> list[str]:
        """Get list of entity types that the model can extract"""
        if not self.label_list:
            return []

        entity_types = set()
        for label in self.label_list:
            if label.startswith("B-"):
                entity_types.add(label[2:])

        return sorted(entity_types)

    def extract_entities(self, text: str) -> dict[str, list[dict]]:
        """
        Extract genealogical entities from text

        Args:
            text: Input text to process

        Returns:
            Dictionary with entity types as keys and lists of extracted entities as values.
            Each entity is a dict with 'text', 'start', 'end', 'confidence' keys.
        """
        if not self._loaded:
            self.load_model()

        # Tokenize text
        tokens = text.split()  # Simple tokenization for now
        if not tokens:
            return {}

        # Tokenize with the model tokenizer
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(**tokenized)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)

        # Convert back to labels
        tokens_predictions = []
        word_ids = tokenized.word_ids(batch_index=0)
        previous_word_id = None

        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != previous_word_id:
                label_id = predicted_labels[0][i].item()
                confidence = predictions[0][i][label_id].item()
                label = self.label_list[label_id]
                tokens_predictions.append((word_id, tokens[word_id], label, confidence))
            previous_word_id = word_id

        # Group consecutive entities
        return self._group_entities(tokens_predictions, text)

    def _group_entities(self, tokens_predictions: list[tuple], original_text: str) -> dict[str, list[dict]]:
        """Group consecutive B- and I- tags into single entities"""
        entities = {}
        current_entity = None
        current_tokens = []
        current_confidences = []

        for word_id, token, label, confidence in tokens_predictions:
            if label.startswith("B-"):
                # Start new entity
                if current_entity:
                    # Finish previous entity
                    self._add_entity_to_dict(
                        entities,
                        current_entity,
                        current_tokens,
                        current_confidences,
                        original_text,
                    )

                current_entity = label[2:]  # Remove 'B-' prefix
                current_tokens = [(word_id, token)]
                current_confidences = [confidence]

            elif label.startswith("I-") and current_entity == label[2:]:
                # Continue current entity
                current_tokens.append((word_id, token))
                current_confidences.append(confidence)

            # End current entity (O tag or different entity type)
            elif current_entity:
                self._add_entity_to_dict(
                    entities,
                    current_entity,
                    current_tokens,
                    current_confidences,
                    original_text,
                )
                current_entity = None
                current_tokens = []
                current_confidences = []

        # Handle final entity
        if current_entity:
            self._add_entity_to_dict(
                entities,
                current_entity,
                current_tokens,
                current_confidences,
                original_text,
            )

        return entities

    def _add_entity_to_dict(
        self,
        entities_dict: dict,
        entity_type: str,
        tokens: list[tuple],
        confidences: list[float],
        original_text: str,
    ):
        """Add grouped entity to the entities dictionary"""
        if not tokens:
            return

        # Join tokens to form entity text
        entity_text = " ".join([token for _, token in tokens])

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences)

        # Find position in original text (approximate)
        start_pos = original_text.find(entity_text)
        end_pos = start_pos + len(entity_text) if start_pos != -1 else 0

        entity_info = {
            "text": entity_text,
            "start": start_pos,
            "end": end_pos,
            "confidence": avg_confidence,
            "tokens": len(tokens),
        }

        if entity_type not in entities_dict:
            entities_dict[entity_type] = []

        entities_dict[entity_type].append(entity_info)

    def extract_entities_batch(self, texts: list[str]) -> list[dict[str, list[dict]]]:
        """Extract entities from multiple texts in batch"""
        return [self.extract_entities(text) for text in texts]

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self._loaded:
            self.load_model()

        # Load training info if available
        training_info_path = self.model_path / "training_info.json"
        training_info = {}
        if training_info_path.exists():
            import json

            with open(training_info_path, encoding="utf-8") as f:
                training_info = json.load(f)

        return {
            "model_path": str(self.model_path),
            "entity_types": self._get_entity_types(),
            "num_labels": len(self.label_list),
            "training_info": training_info,
        }


def get_default_ner_extractor() -> GenealogyNERExtractor | None:
    """Get default NER extractor instance, returns None if no model available"""
    try:
        extractor = GenealogyNERExtractor()
        extractor.load_model()
        return extractor
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"NER extractor not available: {e}")
        return None
