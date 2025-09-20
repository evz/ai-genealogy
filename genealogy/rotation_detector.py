#!/usr/bin/env python3
"""
GPU-accelerated rotation detection for document images.

Combines coarse rotation detection (Tesseract OSD) with fine rotation detection
(GPU projection profiles) for accurate document deskewing.
"""

import logging

import cv2
import kornia
import numpy as np
import pytesseract
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class RotationDetector:
    """
    GPU-accelerated rotation detector for document images.

    Uses a two-step approach:
    1. Coarse rotation detection using Tesseract OSD (0 or 180 degree rotations)
    2. Fine rotation detection using GPU projection profiles (-3 to +3 degrees)
    """

    def __init__(self, use_gpu: bool = True) -> None:
        """
        Initialize the rotation detector.

        Args:
            use_gpu: Whether to use GPU acceleration for fine rotation.
                    Falls back to CPU if GPU is not available.
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if self.use_gpu:
            self.device = torch.device("cuda:0")
            logger.info(f"RotationDetector: Using GPU acceleration on {self.device}")
        else:
            self.device = torch.device("cpu")
            logger.info("RotationDetector: Using CPU processing")

    def detect_coarse_rotation(self, image_pil: Image.Image) -> int:
        """
        Detect coarse rotation (0 or 180 degrees) using Tesseract OSD.

        Args:
            image_pil: Input image

        Returns:
            Rotation angle in degrees (0 or 180)
        """
        try:
            # Convert to format suitable for Tesseract
            if image_pil.mode == "RGBA":
                image_pil = image_pil.convert("RGB")

            # Run OSD (Orientation and Script Detection)
            osd_result = pytesseract.image_to_osd(image_pil)

            # Parse rotation from OSD output
            for line in osd_result.split("\n"):
                if "Rotate:" in line:
                    rotation_str = line.split("Rotate:")[1].strip()
                    rotation = int(rotation_str)

                    # Only handle 0 or 180 degree rotations
                    if rotation == 180:
                        logger.info(f"Coarse rotation detected: {rotation}°")
                        return 180
                    logger.info("No coarse rotation needed")
                    return 0

            logger.info("No coarse rotation detected")
            return 0

        except Exception as e:
            logger.warning(f"Coarse rotation detection failed: {e}")
            return 0

    def detect_fine_rotation_gpu(self, image_tensor: torch.Tensor) -> float:
        """
        GPU-accelerated fine rotation detection using projection profiles.

        Args:
            image_tensor: Image tensor on GPU [1, H, W]

        Returns:
            Fine rotation angle in degrees (-3 to +3)
        """
        device = image_tensor.device
        angles = torch.linspace(-3, 3, 61, device=device)  # 0.1° precision

        best_angle = 0.0
        max_variance = 0.0

        for angle in angles:
            # Rotate image
            rotated = kornia.geometry.transform.rotate(
                image_tensor, torch.tensor([angle], device=device, dtype=torch.float32)
            )

            # Invert colors (black text on white background becomes white text on black)
            inverted = 1.0 - rotated.squeeze()

            # Calculate horizontal projection (sum along width dimension)
            horizontal_proj = torch.sum(inverted, dim=1)

            # Calculate variance - higher variance indicates better alignment
            variance = torch.var(horizontal_proj)

            if variance > max_variance:
                max_variance = variance
                best_angle = angle.item()

        # Only return meaningful corrections (> 0.05 degrees)
        return best_angle if abs(best_angle) > 0.05 else 0.0

    def detect_fine_rotation_cpu(self, image_pil: Image.Image) -> float:
        """
        CPU fallback for fine rotation detection.

        Args:
            image_pil: Input image

        Returns:
            Fine rotation angle in degrees (-3 to +3)
        """
        # Convert to grayscale numpy array
        if image_pil.mode != "L":
            image_pil = image_pil.convert("L")
        image_np = np.array(image_pil)

        angles = np.linspace(-3, 3, 61)  # 0.1° precision
        best_angle = 0.0
        max_variance = 0.0

        height, width = image_np.shape
        center = (width // 2, height // 2)

        for angle in angles:
            # Rotate image
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image_np, rotation_matrix, (width, height))

            # Invert (assuming black text on white background)
            inverted = 255 - rotated

            # Calculate horizontal projection
            horizontal_proj = np.sum(inverted, axis=1)

            # Calculate variance
            variance = np.var(horizontal_proj)

            if variance > max_variance:
                max_variance = variance
                best_angle = angle

        return best_angle if abs(best_angle) > 0.05 else 0.0

    def apply_rotation(self, image_pil: Image.Image, total_angle: float) -> Image.Image:
        """
        Apply rotation correction to image.

        Args:
            image_pil: Input image
            total_angle: Total rotation angle to apply

        Returns:
            Corrected image
        """
        if abs(total_angle) < 0.01:  # No meaningful rotation
            return image_pil

        if self.use_gpu:
            # GPU rotation using Kornia (like original implementation)
            # Convert to grayscale if needed
            if image_pil.mode != "L":
                image_pil = image_pil.convert("L")

            # Convert to tensor using torch.from_numpy (like original)
            full_img_tensor = torch.from_numpy(np.array(image_pil)).float().unsqueeze(0).unsqueeze(0) / 255.0
            full_img_tensor = full_img_tensor.to(self.device)

            corrected_tensor = kornia.geometry.transform.rotate(
                full_img_tensor, torch.tensor([total_angle], device=self.device, dtype=torch.float32)
            )

            # Convert back to PIL Image (like original)
            corrected_np = (corrected_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(corrected_np)
        # CPU rotation using PIL
        return image_pil.rotate(-total_angle, expand=True, fillcolor="white")

    def detect_and_correct(self, image_pil: Image.Image) -> tuple[Image.Image, float]:
        """
        Full rotation detection and correction pipeline.

        Args:
            image_pil: Input image

        Returns:
            Tuple of (corrected_image, total_rotation_applied)
        """
        logger.info("Starting rotation detection...")

        # Step 1: Coarse rotation detection
        logger.debug("Detecting coarse rotation...")
        coarse_rotation = self.detect_coarse_rotation(image_pil)

        # Apply coarse rotation if needed
        if coarse_rotation != 0:
            logger.info(f"Applying coarse rotation: {coarse_rotation}°")
            image_pil = image_pil.rotate(-coarse_rotation, expand=True)

        # Step 2: Fine rotation detection
        logger.debug("Detecting fine rotation...")
        if self.use_gpu:
            # Convert to grayscale for GPU processing (like original implementation)
            if image_pil.mode != "L":
                image_pil_gray = image_pil.convert("L")
            else:
                image_pil_gray = image_pil

            # Downscale for speed (like original implementation)
            original_size = image_pil_gray.size
            downscale_factor = 200 / 600
            new_size = (int(original_size[0] * downscale_factor), int(original_size[1] * downscale_factor))
            image_small = image_pil_gray.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to tensor using torch.from_numpy (like original)
            img_tensor = torch.from_numpy(np.array(image_small)).float().unsqueeze(0).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self.device)
            fine_rotation = self.detect_fine_rotation_gpu(img_tensor)
        else:
            fine_rotation = self.detect_fine_rotation_cpu(image_pil)

        logger.debug(f"Fine rotation detected: {fine_rotation:.2f}°")

        # Step 3: Apply fine rotation
        total_rotation = coarse_rotation + fine_rotation
        if abs(fine_rotation) > 0.01:
            logger.debug(f"Applying fine rotation: {fine_rotation:.2f}°")
            corrected_image = self.apply_rotation(image_pil, fine_rotation)
        else:
            corrected_image = image_pil

        logger.info(f"Total rotation applied: {total_rotation:.2f}°")

        return corrected_image, total_rotation
