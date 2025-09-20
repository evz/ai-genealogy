"""
Document layout detection using DocLayout-YOLO.

Handles document structure analysis, region detection, and overlap resolution
for genealogy documents.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

from django.conf import settings

from doclayout_yolo import YOLOv10
from PIL import Image
from shapely.geometry import box as shapely_box

logger = logging.getLogger(__name__)


class DocumentLayoutDetector:
    """
    Document layout detector using DocLayout-YOLO for genealogy documents.

    Handles region detection, confidence filtering, element type filtering,
    and overlap resolution strategies.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.1,
        iou_threshold: float = 0.5,
        device: str = "cuda:0",
    ) -> None:
        """
        Initialize the document layout detector.

        Args:
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on (cuda:0, cpu, etc.)
        """
        model_path = settings.DOCLAYOUT_MODEL_PATH

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading DocLayout-YOLO model from {model_path}")
        self.model = YOLOv10(str(self.model_path))

        # Element mapping from model output
        self.element_mapping = {
            0: "text",
            1: "title",
            2: "list",
            3: "table",
            4: "figure",
            5: "caption",
            6: "footnote",
            7: "formula",
            8: "reference",
        }

        # Elements to keep for genealogy processing
        self.desired_elements = {"text", "title", "list"}

        logger.info(f"DocumentLayoutDetector initialized - confidence: {confidence_threshold}, " f"device: {device}")

    def detect_regions(self, image: Image.Image) -> list[dict[str, Any]]:
        """
        Detect document regions in the given image.

        Args:
            image: PIL Image to analyze

        Returns:
            List of detected regions with bounding boxes and metadata
        """
        logger.info("Starting document layout detection...")

        # Create unique temporary file for this detection task
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save image to temporary file
            image.save(temp_path, quality=95)

            # Run inference
            logger.debug("Running DocLayout-YOLO inference...")
            detection_results = self.model.predict(
                temp_path,
                imgsz=1280,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                save=False,
                verbose=False,
            )

            # Extract detections
            detections = self._extract_detections(detection_results)
            logger.info(f"Raw detections: {len(detections)} regions")

            # Filter to desired element types
            filtered_detections = self._filter_detections(detections)
            logger.info(f"Filtered detections: {len(filtered_detections)} regions")

            # Resolve overlaps
            resolved_detections = self._resolve_overlaps(filtered_detections)
            logger.info(f"Resolved detections: {len(resolved_detections)} regions")

            return resolved_detections

        finally:
            # Clean up temporary file
            try:
                Path(temp_path).unlink()
            except OSError:
                logger.warning(f"Failed to clean up temporary file: {temp_path}")

    def _extract_detections(self, detection_results) -> list[dict[str, Any]]:
        """
        Extract detections from YOLO model output.

        Args:
            detection_results: Raw YOLO detection results

        Returns:
            List of detection dictionaries
        """
        detections = []

        if not detection_results or len(detection_results) == 0:
            logger.warning("No detection results returned from model")
            return detections

        result = detection_results[0]

        if not hasattr(result, "boxes") or result.boxes is None:
            logger.warning("No boxes found in detection results")
            return detections

        boxes = result.boxes

        if boxes.cls is None or boxes.conf is None:
            logger.warning("No class or confidence data in detection results")
            return detections

        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        bboxes = boxes.xyxy.cpu().numpy()

        for cls_id, conf, bbox in zip(classes, confidences, bboxes, strict=False):
            element_name = self.element_mapping.get(cls_id, f"unknown_{cls_id}")

            detections.append(
                {
                    "element": element_name,
                    "confidence": float(conf),
                    "bbox": [float(x) for x in bbox],
                }
            )

        return detections

    def _filter_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter detections to only include desired element types.

        Args:
            detections: List of detection dictionaries

        Returns:
            Filtered list of detections
        """
        filtered = [d for d in detections if d["element"] in self.desired_elements]

        logger.debug(
            f"Filtered from {len(detections)} to {len(filtered)} detections " f"(kept: {self.desired_elements})"
        )

        return filtered

    def _calculate_overlap(self, box1: list[float], box2: list[float]) -> tuple[float, float, float]:
        """
        Calculate overlap between two bounding boxes.

        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]

        Returns:
            Tuple of (overlap_area, ratio1, ratio2)
        """
        # Convert to shapely boxes for precise geometry
        shape1 = shapely_box(box1[0], box1[1], box1[2], box1[3])
        shape2 = shapely_box(box2[0], box2[1], box2[2], box2[3])

        if not shape1.intersects(shape2):
            return 0, 0, 0

        intersection = shape1.intersection(shape2)
        overlap_area = intersection.area

        overlap_ratio_1 = overlap_area / shape1.area
        overlap_ratio_2 = overlap_area / shape2.area

        return overlap_area, overlap_ratio_1, overlap_ratio_2

    def _resolve_overlaps(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Resolve overlapping bounding boxes using OCR-friendly strategy.

        Strategy:
        1. Complete containment (>90% overlap) - keep larger region only
        2. All other overlaps - keep both regions, deduplicate text later in OCR

        Args:
            detections: List of detection dictionaries

        Returns:
            List of detections with overlaps resolved
        """
        logger.debug(f"Resolving overlaps for {len(detections)} detections...")

        # Find all overlapping pairs
        overlaps = []
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections):
                if i >= j:  # Avoid duplicates
                    continue

                overlap_area, ratio1, ratio2 = self._calculate_overlap(det1["bbox"], det2["bbox"])

                if overlap_area > 0:
                    overlaps.append(
                        {
                            "det1_idx": i,
                            "det2_idx": j,
                            "det1": det1,
                            "det2": det2,
                            "overlap_area": overlap_area,
                            "ratio1": ratio1,  # How much of det1 is covered
                            "ratio2": ratio2,  # How much of det2 is covered
                        }
                    )

        logger.debug(f"Found {len(overlaps)} overlapping pairs")

        # Process overlaps - only remove completely contained regions
        indices_to_remove: set[int] = set()

        for overlap in overlaps:
            i, j = overlap["det1_idx"], overlap["det2_idx"]
            det1, det2 = overlap["det1"], overlap["det2"]
            ratio1, ratio2 = overlap["ratio1"], overlap["ratio2"]

            logger.debug(
                f"Processing overlap: {det1['element']} "
                f"(conf:{det1['confidence']:.2f}) vs "
                f"{det2['element']} (conf:{det2['confidence']:.2f})"
            )
            logger.debug(f"Overlap ratios: {ratio1:.2f}, {ratio2:.2f}")

            # Only remove completely contained regions (>90% coverage)
            if ratio1 > 0.9:  # det1 is mostly covered by det2
                logger.debug("Strategy: det1 entirely contained -> remove det1, keep det2")
                indices_to_remove.add(i)
            elif ratio2 > 0.9:  # det2 is mostly covered by det1
                logger.debug("Strategy: det2 entirely contained -> remove det2, keep det1")
                indices_to_remove.add(j)
            else:
                # Keep both overlapping regions - will deduplicate text in OCR
                logger.debug("Strategy: partial overlap -> keep both regions, deduplicate OCR text")

        # Apply removals (no modifications, just remove completely contained regions)
        final_detections = []
        for idx, detection in enumerate(detections):
            if idx in indices_to_remove:
                logger.debug(f"Removing detection {idx}: {detection['element']} (contained)")
                continue
            final_detections.append(detection)

        logger.debug(f"Overlap resolution: {len(detections)} -> {len(final_detections)} regions")
        return final_detections
