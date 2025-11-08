#!/usr/bin/env python3
"""
DocStructBench production implementation with overlap resolution strategies.
Uses 0.1 confidence threshold and smart overlap handling.
"""

from pathlib import Path

import kornia
import numpy as np
import pytesseract
import torch
from doclayout_yolo import YOLOv10
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box as shapely_box


def detect_orientation(image_pil):
    """Detect page orientation using Tesseract PSM 0 (OSD only)"""
    try:
        osd_result = pytesseract.image_to_osd(image_pil, lang="eng+nld", config="--oem 2 --psm 0")

        for line in osd_result.split("\n"):
            if "Rotate:" in line:
                rotation_str = line.split("Rotate:")[1].strip()
                return int(rotation_str)
        return 0
    except Exception as e:
        print(f"  Orientation detection failed: {e}")
        return 0


def gpu_projection_profile_rotation(image_tensor):
    """GPU-native projection profile method for fine rotation detection"""
    device = image_tensor.device
    angles = torch.linspace(-3, 3, 61, device=device)

    best_angle = 0.0
    max_variance = 0.0

    for angle in angles:
        rotated = kornia.geometry.transform.rotate(
            image_tensor, torch.tensor([angle], device=device, dtype=torch.float32)
        )

        inverted = 1.0 - rotated.squeeze()
        horizontal_proj = torch.sum(inverted, dim=1)
        variance = torch.var(horizontal_proj)

        if variance > max_variance:
            max_variance = variance
            best_angle = angle.item()

    return best_angle if abs(best_angle) > 0.05 else 0.0


def gpu_deskew_image(image_pil):
    """GPU-accelerated fine rotation detection"""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available for accelerated rotation detection")

    device = torch.device("cuda:0")

    if image_pil.mode != "L":
        image_pil = image_pil.convert("L")

    # Downscale for speed
    original_size = image_pil.size
    downscale_factor = 200 / 600
    new_size = (int(original_size[0] * downscale_factor), int(original_size[1] * downscale_factor))
    image_small = image_pil.resize(new_size, Image.Resampling.LANCZOS)

    img_tensor = torch.from_numpy(np.array(image_small)).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    correction_angle = gpu_projection_profile_rotation(img_tensor)

    if abs(correction_angle) > 0.05:
        print(f"  GPU detected {correction_angle:.2f}° skew, correcting...")

        full_img_tensor = torch.from_numpy(np.array(image_pil)).float().unsqueeze(0).unsqueeze(0) / 255.0
        full_img_tensor = full_img_tensor.to(device)

        corrected_tensor = kornia.geometry.transform.rotate(
            full_img_tensor, torch.tensor([correction_angle], device=device, dtype=torch.float32)
        )

        corrected_np = (corrected_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        corrected_image = Image.fromarray(corrected_np)
        return corrected_image, correction_angle
    return image_pil, 0.0


def filter_detections(detections):
    """Filter detections to only include desired element types"""
    # Only keep text, title, and list elements (exclude table/figure/etc)
    desired_elements = {"text", "title", "list"}

    filtered = [d for d in detections if d["element"] in desired_elements]

    print(f"  Filtered from {len(detections)} to {len(filtered)} detections (kept: {desired_elements})")
    return filtered


def calculate_overlap(box1, box2):
    """Calculate overlap between two boxes [x1,y1,x2,y2]"""
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


def resolve_overlaps(detections):
    """
    Resolve overlapping bounding boxes using OCR-friendly strategy:
    1. Complete containment (>90% overlap) - keep larger region only
    2. All other overlaps - keep both regions, deduplicate text later in OCR

    This avoids chopping text lines and preserves all data for deduplication.
    """

    print(f"  Resolving overlaps for {len(detections)} detections...")

    # First, identify all overlaps
    overlaps = []
    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections):
            if i >= j:  # Avoid duplicates
                continue

            overlap_area, ratio1, ratio2 = calculate_overlap(det1["bbox"], det2["bbox"])

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

    print(f"    Found {len(overlaps)} overlapping pairs")

    # Process overlaps by type
    resolved_detections = detections.copy()
    indices_to_remove = set()
    indices_to_modify = {}  # idx -> new_bbox

    for overlap in overlaps:
        i, j = overlap["det1_idx"], overlap["det2_idx"]
        det1, det2 = overlap["det1"], overlap["det2"]
        ratio1, ratio2 = overlap["ratio1"], overlap["ratio2"]

        print(
            f"    Processing overlap: {det1['element']} (conf:{det1['confidence']:.2f}) vs "
            f"{det2['element']} (conf:{det2['confidence']:.2f})"
        )
        print(f"      Overlap ratios: {ratio1:.2f}, {ratio2:.2f}")

        # Scenario 1: One area entirely covers another (>90% coverage)
        if ratio1 > 0.9:  # det1 is mostly covered by det2
            print("      Strategy: det1 entirely covered by det2 -> keep det2, remove det1")
            indices_to_remove.add(i)
        elif ratio2 > 0.9:  # det2 is mostly covered by det1
            print("      Strategy: det2 entirely covered by det1 -> keep det1, remove det2")
            indices_to_remove.add(j)

        # Scenario 2: All other overlaps - keep both regions for OCR deduplication
        else:
            print("      Strategy: partial/minor overlap -> keep both regions, deduplicate OCR text")

    # Apply removals (no modifications, just remove completely contained regions)
    final_detections = []
    for idx, detection in enumerate(resolved_detections):
        if idx in indices_to_remove:
            print(f"    Removing detection {idx}: {detection['element']} (completely contained)")
            continue
        final_detections.append(detection)

    print(f"  Overlap resolution: {len(detections)} -> {len(final_detections)} regions")
    return final_detections


def chop_overlap_from_box(large_box, small_box):
    """
    Remove the overlapping area from large_box by finding the largest
    non-overlapping rectangular piece.
    """
    x1, y1, x2, y2 = large_box
    sx1, sy1, sx2, sy2 = small_box

    # Create candidate rectangles by chopping at the overlap boundaries
    candidates = []

    # Left piece (everything to the left of the overlapping area)
    if sx1 > x1:
        candidates.append([x1, y1, sx1, y2])

    # Right piece (everything to the right of the overlapping area)
    if sx2 < x2:
        candidates.append([sx2, y1, x2, y2])

    # Top piece (everything above the overlapping area)
    if sy1 > y1:
        candidates.append([x1, y1, x2, sy1])

    # Bottom piece (everything below the overlapping area)
    if sy2 < y2:
        candidates.append([x1, sy2, x2, y2])

    if not candidates:
        return None

    # Return the largest candidate by area
    def area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    largest = max(candidates, key=area)

    # Verify the result doesn't overlap with small_box
    large_shape = shapely_box(largest[0], largest[1], largest[2], largest[3])
    small_shape = shapely_box(sx1, sy1, sx2, sy2)

    if large_shape.intersects(small_shape):
        print("      Warning: chopped box still overlaps!")
        return None

    return largest


def run_ocr_on_regions(image, regions):
    """Run OCR on resolved regions and stitch results together"""

    print(f"  Running OCR on {len(regions)} regions...")

    ocr_results = []

    for idx, region in enumerate(regions):
        bbox = region["bbox"]
        element_type = region["element"]
        confidence = region["confidence"]

        # Extract region from image
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        region_image = image.crop((x1, y1, x2, y2))

        # Use consistent OCR settings for all element types
        # PSM 6 (single uniform block) works best for genealogy text
        config = "--oem 2 --psm 6"

        try:
            # Run OCR
            text = pytesseract.image_to_string(region_image, lang="eng+nld", config=config)

            # Get confidence score
            tsv_data = pytesseract.image_to_data(
                region_image, lang="eng+nld", config=config, output_type=pytesseract.Output.DICT
            )
            ocr_confidences = [int(conf) for conf in tsv_data["conf"] if int(conf) > 0]
            avg_ocr_conf = np.mean(ocr_confidences) if ocr_confidences else 0

            ocr_results.append(
                {
                    "region_idx": idx,
                    "element_type": element_type,
                    "detection_confidence": confidence,
                    "ocr_confidence": avg_ocr_conf,
                    "bbox": bbox,
                    "text": text.strip(),
                    "modified": region.get("modified", False),
                }
            )

            print(f"    Region {idx} ({element_type}): {len(text.strip())} chars, OCR conf: {avg_ocr_conf:.1f}")

        except Exception as e:
            print(f"    Error OCR'ing region {idx}: {e}")
            ocr_results.append(
                {
                    "region_idx": idx,
                    "element_type": element_type,
                    "detection_confidence": confidence,
                    "bbox": bbox,
                    "error": str(e),
                    "modified": region.get("modified", False),
                }
            )

    return ocr_results


def deduplicate_text_lines(text_lines):
    """Remove duplicate and near-duplicate text lines from OCR results"""
    if not text_lines:
        return text_lines

    # Remove exact duplicates while preserving order
    seen = set()
    unique_lines = []

    for line in text_lines:
        line_stripped = line.strip()
        if line_stripped and line_stripped not in seen:
            seen.add(line_stripped)
            unique_lines.append(line)

    # Remove near-duplicates (lines that are substrings of others)
    filtered_lines = []

    for i, line1 in enumerate(unique_lines):
        is_substring = False
        line1_clean = line1.strip()

        for j, line2 in enumerate(unique_lines):
            if i != j:
                line2_clean = line2.strip()
                # If line1 is a substring of line2 and significantly shorter
                if line1_clean in line2_clean and len(line1_clean) < len(line2_clean) * 0.8:
                    is_substring = True
                    break

        if not is_substring:
            filtered_lines.append(line1)

    return filtered_lines


def stitch_text_results(ocr_results):
    """Stitch OCR results back together in reading order with deduplication"""

    if not ocr_results:
        return "", []

    # Get image width to determine what's "mid-page"
    # Use the rightmost x-coordinate as a rough page width estimate
    max_x = max(result["bbox"][2] for result in ocr_results if "text" in result)
    page_width_threshold = max_x * 0.2  # 20% from left edge

    # Separate main content from inset content
    main_content = []
    inset_content = []

    for result in ocr_results:
        if "text" not in result:
            continue

        bbox = result["bbox"]
        x_start = bbox[0]

        if x_start <= page_width_threshold:
            main_content.append(result)
        else:
            inset_content.append(result)

    print(f"    Split content: {len(main_content)} main regions, {len(inset_content)} inset regions")

    # Sort main content by reading order (top-to-bottom, left-to-right)
    def reading_order_key(result):
        bbox = result["bbox"]
        return (bbox[1], bbox[0])

    main_sorted = sorted(main_content, key=reading_order_key)
    inset_sorted = sorted(inset_content, key=reading_order_key)

    # Combine: main content first, then inset content
    sorted_results = main_sorted + inset_sorted

    # Extract all text lines first
    all_text_lines = []
    for result in sorted_results:
        text = result.get("text", "").strip()
        if text:
            # Split into lines and add each line
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            all_text_lines.extend(lines)

    # Deduplicate text lines
    deduplicated_lines = deduplicate_text_lines(all_text_lines)

    # Group by approximate row (y-position) for the remaining display logic
    rows = []
    current_row = []
    row_threshold = 50  # pixels

    for result in sorted_results:
        if not current_row:
            current_row = [result]
        else:
            # Check if this result is in the same row as the previous
            prev_y = current_row[-1]["bbox"][1]
            curr_y = result["bbox"][1]

            if abs(curr_y - prev_y) <= row_threshold:
                current_row.append(result)
            else:
                # Start new row
                rows.append(current_row)
                current_row = [result]

    if current_row:
        rows.append(current_row)

    # Use deduplicated text for final output
    final_text_parts = deduplicated_lines

    print(f"    Text deduplication: {len(all_text_lines)} -> {len(deduplicated_lines)} lines")

    return "\n".join(final_text_parts), rows


def process_page_with_overlap_resolution(page_num, file_path):
    """Full pipeline: rotation -> detection -> overlap resolution -> OCR -> stitching"""

    print(f"\\n{'='*70}")
    print(f"PROCESSING PAGE {page_num} WITH OVERLAP RESOLUTION")
    print(f"{'='*70}")

    try:
        # Step 1: Load and rotate
        print("1. Loading and correcting rotation...")
        images = convert_from_path(file_path, first_page=1, dpi=600)
        image = images[0]

        coarse_rotation = detect_orientation(image)
        if coarse_rotation != 0:
            image = image.rotate(coarse_rotation, expand=True)

        image, fine_rotation = gpu_deskew_image(image)
        total_rotation = coarse_rotation + fine_rotation
        print(f"   Total rotation: {total_rotation:.2f}°")

        # Step 2: Run DocStructBench detection
        print("2. Running DocStructBench 1280 detection...")
        model_path = "/app/models/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
        model = YOLOv10(model_path)

        temp_image_path = f"/tmp/page_{page_num}_processed.jpg"
        image.save(temp_image_path, quality=95)

        detection_results = model.predict(
            temp_image_path,
            imgsz=1280,
            conf=0.1,  # Your optimal confidence threshold
            iou=0.5,
            device="cuda:0",
            save=False,
            verbose=False,
        )

        # Step 3: Extract and filter detections
        print("3. Extracting and filtering detections...")
        detections = []

        if detection_results and len(detection_results) > 0:
            result = detection_results[0]

            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes

                if boxes.cls is not None and boxes.conf is not None:
                    classes = boxes.cls.cpu().numpy().astype(int)
                    confidences = boxes.conf.cpu().numpy()
                    bboxes = boxes.xyxy.cpu().numpy()

                    elements = {
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

                    for cls_id, conf, bbox in zip(classes, confidences, bboxes, strict=False):
                        element_name = elements.get(cls_id, f"unknown_{cls_id}")

                        detections.append(
                            {"element": element_name, "confidence": float(conf), "bbox": [float(x) for x in bbox]}
                        )

        # Filter to desired element types
        filtered_detections = filter_detections(detections)

        # Step 4: Resolve overlaps
        print("4. Resolving overlapping regions...")
        resolved_detections = resolve_overlaps(filtered_detections)

        # Step 5: Run OCR on resolved regions
        print("5. Running OCR on resolved regions...")
        ocr_results = run_ocr_on_regions(image, resolved_detections)

        # Step 6: Stitch results together
        print("6. Stitching results in reading order...")
        final_text, sorted_results = stitch_text_results(ocr_results)

        # Step 7: Create bounding box visualization
        print("7. Creating bounding box visualization...")
        create_resolution_visualization(image, resolved_detections, page_num)

        # Step 8: Write final OCR text to file
        print("8. Writing final OCR text...")
        final_output_path = f"/app/page_{page_num:03d}_final_text.txt"
        with open(final_output_path, "w", encoding="utf-8") as f:
            f.write(f"PAGE {page_num} - FINAL OCR TEXT\n")
            f.write("=" * 60 + "\n\n")
            f.write(final_text)

        print(f"    Final OCR text saved to: {final_output_path}")

        print("\\n🎉 PROCESSING COMPLETE!")
        print(f"   📄 Final text: {len(final_text)} characters")
        print(f"   📊 Regions processed: {len(resolved_detections)}")
        print(f"   🔄 Total rotation: {total_rotation:.2f}°")

        return {
            "page_num": page_num,
            "total_rotation": total_rotation,
            "region_count": len(resolved_detections),
            "final_text_length": len(final_text),
            "output_file": final_output_path,
        }

    except Exception as e:
        print(f"❌ ERROR processing page {page_num}: {e}")
        return {"page_num": page_num, "error": str(e)}


def create_resolution_visualization(image, detections, page_num):
    """Create visualization showing resolved regions"""

    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)

    colors = {"text": "blue", "title": "red", "list": "green"}

    for idx, detection in enumerate(detections):
        element_type = detection["element"]
        bbox = detection["bbox"]
        conf = detection["confidence"]
        modified = detection.get("modified", False)

        color = colors.get(element_type, "gray")

        # Use thicker border for modified regions
        width = 5 if modified else 3

        # Draw bounding box
        draw.rectangle(bbox, outline=color, width=width)

        # Add label
        label = f"{element_type}: {conf:.2f}"
        if modified:
            label += " [CHOPPED]"

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Draw label
        text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bbox[0], bbox[1] - 20), label, fill="white", font=font)

        # Add region number
        draw.text((bbox[0] + 5, bbox[1] + 5), str(idx), fill=color, font=font)

    # Save
    output_path = f"/app/page_{page_num:03d}_overlap_resolved.jpg"
    vis_image.save(output_path, quality=90)
    print(f"   🖼️  Visualization saved: {output_path}")


def main():
    """Test the overlap resolution pipeline"""

    print("DocStructBench Production with Overlap Resolution")
    print("=" * 60)

    # Test on your key pages
    test_pages = [
        (7, "/app/media/document_pages/007.pdf"),
        (11, "/app/media/document_pages/011.pdf"),
        (20, "/app/media/document_pages/020_E6b7PAp.pdf"),
        (21, "/app/media/document_pages/021_KEogAn7.pdf"),
        (22, "/app/media/document_pages/022_du2mr2N.pdf"),
        (27, "/app/media/document_pages/027_fP0mKja.pdf"),
        (30, "/app/media/document_pages/030_NIbGR1V.pdf"),
        (83, "/app/media/document_pages/083.pdf"),
        (100, "/app/media/document_pages/100.pdf"),
    ]

    results = {}

    for page_num, file_path in test_pages:
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            continue

        result = process_page_with_overlap_resolution(page_num, file_path)
        results[page_num] = result

    print(f"\\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")

    for page_num, result in results.items():
        if "error" not in result:
            print(f"📄 Page {page_num}: {result['final_text_length']} characters extracted")
            print(f"   📊 {result['region_count']} regions processed")
            print(f"   📝 Output: {result['output_file']}")
        else:
            print(f"❌ Page {page_num}: {result['error']}")

    print("\\n✅ Processing complete! Check page_XXX_final_text.txt files for extracted text.")


if __name__ == "__main__":
    main()
