import logging
import math

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class RotationDetector:
    """Rotation detection using multiple methods for fine-angle precision"""

    def __init__(self):
        self.min_line_length = 100
        self.max_line_gap = 20
        self.angle_threshold = 0.5  # degrees
        self.confidence_threshold = 0.7

    def detect_rotation(self, image: Image.Image) -> tuple[float, float]:
        """
        Detect rotation angle using two-stage approach:
        1. Major rotation detection (0°, 90°, 180°, 270°)
        2. Fine-angle refinement

        Args:
            image: PIL Image to analyze

        Returns:
            Tuple of (rotation_angle, confidence_score)
            rotation_angle: angle in degrees (positive = clockwise)
            confidence_score: 0.0 to 1.0
        """
        try:
            # Stage 1: Detect major rotation using projection profiles
            major_angle, major_confidence = self._detect_major_rotation(image)

            # Apply major rotation correction
            if abs(major_angle) > 5:
                corrected_image = image.rotate(-major_angle, expand=True, fillcolor="white")
                logger.debug(f"Applied major rotation correction: {major_angle}°")
            else:
                corrected_image = image
                major_angle = 0

            # Stage 2: Fine-angle detection on corrected image
            fine_angle, fine_confidence = self._detect_fine_angle(corrected_image)

            # Combine angles
            total_angle = major_angle + fine_angle
            combined_confidence = (major_confidence + fine_confidence) / 2

            logger.info(
                f"Rotation detection: {total_angle:.2f}° (major: {major_angle}°, "
                f"fine: {fine_angle:.2f}°, confidence: {combined_confidence:.2f})"
            )

            return total_angle, combined_confidence

        except Exception as e:
            logger.exception(f"Rotation detection failed: {e}")
            return 0.0, 0.0

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format"""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        cv_image = np.array(pil_image)
        return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    def _detect_with_hough_lines(self, cv_image: np.ndarray) -> tuple[float, float]:
        """Detect rotation using Hough Line Transform - best for text documents"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Edge detection with careful parameter tuning
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Apply morphological operations to connect text characters
            kernel = np.ones((1, 5), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=80,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap,
            )

            if lines is None or len(lines) < 5:
                return 0.0, 0.0

            # Calculate angles of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Skip vertical lines (they don't help with text rotation)
                if abs(x2 - x1) < 5:
                    continue

                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)

            if not angles:
                return 0.0, 0.0

            # Use histogram to find most common angle
            hist, bins = np.histogram(angles, bins=180, range=(-90, 90))
            peak_bin = np.argmax(hist)
            most_common_angle = bins[peak_bin] + (bins[1] - bins[0]) / 2

            # Calculate confidence based on how many lines agree
            angle_tolerance = 2.0  # degrees
            agreeing_lines = sum(1 for a in angles if abs(a - most_common_angle) < angle_tolerance)
            confidence = min(agreeing_lines / max(len(angles), 1), 1.0)

            return most_common_angle, confidence

        except Exception as e:
            logger.warning(f"Hough line rotation detection failed: {e}")
            return 0.0, 0.0

    def _detect_with_osd(self, image: Image.Image) -> tuple[float, float]:
        """Use Tesseract OSD for major rotation detection"""
        try:
            osd_data = pytesseract.image_to_osd(image, config="--psm 0")

            rotation_angle = 0.0
            rotation_confidence = 0.0

            for line in osd_data.split("\n"):
                if "Rotate:" in line:
                    rotation_angle = float(line.split(":")[1].strip())
                elif "Orientation confidence:" in line:
                    rotation_confidence = float(line.split(":")[1].strip()) / 100.0

            # OSD returns angles as 0, 90, 180, 270, but we want signed angles
            if rotation_angle == 90:
                rotation_angle = -90  # Counter-clockwise
            elif rotation_angle == 270:
                rotation_angle = 90  # Clockwise

            return rotation_angle, rotation_confidence

        except Exception as e:
            logger.warning(f"OSD rotation detection failed: {e}")
            return 0.0, 0.0

    def _detect_with_projection(self, cv_image: np.ndarray) -> tuple[float, float]:
        """Detect rotation using horizontal projection profiles"""
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Try different angles and find the one with maximum horizontal projection variance
            best_angle = 0.0
            max_variance = 0.0

            # Test angles from -10 to +10 degrees in 0.25 degree steps
            test_angles = np.arange(-10, 10.25, 0.25)

            for angle in test_angles:
                # Rotate image
                rows, cols = gray.shape
                rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                rotated = cv2.warpAffine(gray, rotation_matrix, (cols, rows))

                # Calculate horizontal projection (sum of pixels in each row)
                h_projection = np.sum(rotated, axis=1)

                # Calculate variance of projection (higher = more aligned text lines)
                variance = np.var(h_projection)

                if variance > max_variance:
                    max_variance = variance
                    best_angle = angle

            # Simple confidence based on projection variance
            confidence = min(max_variance / 1000000.0, 1.0)  # Normalize roughly

            return best_angle, confidence

        except Exception as e:
            logger.warning(f"Projection rotation detection failed: {e}")
            return 0.0, 0.0

    def _detect_major_rotation(self, image: Image.Image) -> tuple[float, float]:
        """Detect major rotations (0°, 90°, 180°, 270°) using projection analysis"""
        try:
            cv_image = self._pil_to_cv2(image)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Test each major rotation and find the one with best text alignment
            rotations = [0, 90, 180, 270]
            variances = {}

            for angle in rotations:
                # Rotate image
                if angle == 0:
                    test_img = gray
                else:
                    rows, cols = gray.shape
                    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
                    test_img = cv2.warpAffine(gray, rotation_matrix, (cols, rows))

                # Calculate horizontal projection variance (higher = better text alignment)
                h_projection = np.sum(test_img, axis=1)
                variance = np.var(h_projection)
                variances[angle] = variance

            # Find best angle
            best_angle = max(variances, key=variances.get)
            max_variance = variances[best_angle]

            # Calculate confidence based on how much better the best angle is vs second best
            sorted_variances = sorted(variances.values(), reverse=True)
            if len(sorted_variances) > 1 and sorted_variances[1] > 0:
                improvement_ratio = sorted_variances[0] / sorted_variances[1]
                confidence = min((improvement_ratio - 1.0) * 0.5, 1.0)
            else:
                confidence = 0.5

            # Additional check: if 0° and 180° are very close, prefer 0° (no rotation)
            if best_angle == 180 and abs(variances[180] - variances[0]) / variances[0] < 0.1:
                best_angle = 0
                confidence *= 0.7  # Lower confidence due to ambiguity

            logger.debug(
                f"Major rotation detection: {best_angle}° (variances: {variances}, confidence: {confidence:.2f})"
            )
            return best_angle, confidence

        except Exception as e:
            logger.warning(f"Major rotation detection failed: {e}")
            return 0.0, 0.0

    def _detect_fine_angle(self, image: Image.Image) -> tuple[float, float]:
        """Detect fine angle corrections (within ±10°) using Hough lines"""
        try:
            cv_image = self._pil_to_cv2(image)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Edge detection optimized for text
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Morphological operations to connect text elements
            kernel = np.ones((1, 5), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Detect lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=100,
                maxLineGap=20,
            )

            if lines is None or len(lines) < 3:
                # Fallback to projection method for fine tuning
                return self._detect_fine_with_projection(gray)

            # Calculate angles of horizontal lines only
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10:  # Skip vertical lines
                    continue

                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                # Only consider small angles (fine corrections)
                if abs(angle) <= 10:
                    angles.append(angle)

            if not angles:
                return 0.0, 0.0

            # Find most common angle using median (robust to outliers)
            median_angle = np.median(angles)

            # Calculate confidence based on angle consistency
            angle_std = np.std(angles) if len(angles) > 1 else 0
            confidence = max(0, 1.0 - angle_std / 5.0)  # Lower std = higher confidence

            logger.debug(f"Fine angle detection: {median_angle:.2f}° from {len(angles)} lines")
            return median_angle, confidence

        except Exception as e:
            logger.warning(f"Fine angle detection failed: {e}")
            return 0.0, 0.0

    def _detect_fine_with_projection(self, gray: np.ndarray) -> tuple[float, float]:
        """Fallback fine angle detection using projection profiles"""
        try:
            best_angle = 0.0
            max_variance = 0.0

            # Test small angles from -5 to +5 degrees
            test_angles = np.arange(-5, 5.5, 0.5)

            for angle in test_angles:
                rows, cols = gray.shape
                rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                rotated = cv2.warpAffine(gray, rotation_matrix, (cols, rows))

                # Calculate horizontal projection variance
                h_projection = np.sum(rotated, axis=1)
                variance = np.var(h_projection)

                if variance > max_variance:
                    max_variance = variance
                    best_angle = angle

            confidence = 0.5  # Medium confidence for projection method
            return best_angle, confidence

        except Exception as e:
            logger.warning(f"Projection fine angle detection failed: {e}")
            return 0.0, 0.0

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-180, 180] range"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
