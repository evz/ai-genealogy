# DocLayout-YOLO: A Smarter Way to Read Complex Documents

## The Problem We Were Trying to Solve

Our genealogy documents were defeating us. We had a working OCR system, but the results were frustratingly inconsistent. Sometimes we'd get beautiful, clean text extraction. Other times, entire sections would vanish, genealogical IDs would get jumbled with biographical text, or margin notes would appear in the middle of family records.

The core issue was that we were treating document analysis like a black box. Feed an image to Tesseract, hope for the best. But genealogy documents aren't simple - they have complex layouts with inset boxes, margin annotations, mixed text densities, and spatial relationships that matter for meaning.

We needed to get smarter about understanding document structure before trying to read the text.

## Why the Old Approach Wasn't Working

Our original monolithic `OCRProcessor` class had grown into a monster trying to solve every problem at once:

- Rotation detection mixed with text extraction
- Layout analysis buried inside OCR logic
- No separation between "finding text regions" and "reading those regions"
- Hard to test individual components
- Difficult to debug when things went wrong

When we hit a problematic page, we couldn't tell if the issue was rotation detection, layout analysis, or OCR configuration. Everything was tangled together.

## Research Journey: What We Tried Before DocLayout-YOLO

Before settling on DocLayout-YOLO, we explored several different approaches to document layout analysis:

### CombiSeg: Morphological + Histogram Approach

Our first attempt was based on research by Ptak, Zygadlo, and Unold¹, combined with work by Dos Santos et al.² on morphological operations and histogram projections for text line segmentation.

**The CombiSeg approach:**

1. **Morphological preprocessing** - Remove borders and noise using opening operations
2. **Text region dilation** - Horizontally dilate text to connect characters in the same line
3. **Connected component analysis** - Find potential text line regions
4. **Histogram projection** - Use horizontal projections to split merged lines

**Why it didn't work for us:**
- Designed for single-column newspaper text, not complex genealogical layouts
- Struggled with mixed content (text + margin annotations + inset boxes)
- Required extensive parameter tuning for different document types
- No semantic understanding of document structure
- Processing time was 45x faster than our benchmark, but accuracy was inconsistent on complex layouts

### OCR Quality Assessment: Text-First vs Image-First

We also researched approaches from Schneider and Maurer³ on OCR quality assessment, which gave us insights into different strategies:

**Text-based quality features we tested:**
- Dictionary mapping ratios
- N-gram frequency analysis
- Garbage token detection
- Publication year considerations

**Image-based approaches we considered:**
- Blur estimation and image degradation metrics
- Physical distortion analysis
- Font classification (Antiqua vs Fraktur)

**What we learned:**
- Text-based features were computationally efficient but couldn't solve layout problems
- Image-based quality metrics helped identify problematic pages but didn't improve extraction
- The fundamental issue was layout understanding, not quality assessment

### Image-First Approach: Remove Pictures, Then OCR

Our next experiment tried to solve the "phantom column" problem by removing images before they could confuse Tesseract:

**The pipeline:**
1. **Normalize first** - 400 DPI, deskew, Sauvola binarization, light despeckling
2. **Remove image regions** - Fast variance/entropy detection to identify photos and halftones
3. **Segment into controlled regions** - Connected components + morphology to find text blocks
4. **OCR with appropriate PSM** - PSM 6 for body, PSM 7 for margin bullets, PSM 4 for captions
5. **Maintain reading order** - Sort by position, attach marginalia to nearest lines

**Why it wasn't enough:**
- Image detection was brittle - variance thresholds worked for photos but missed diagrams
- Still required extensive parameter tuning per document type
- Complex pipeline with many failure points
- Didn't solve the core semantic understanding problem

### Text-First Approach: Find Text, Everything Else is Images

We then flipped the problem entirely - instead of detecting images, detect text reliably and treat everything else as image:

**Option A - Tesseract sparse detection:**
```bash
tesseract page.png stdout --psm 11 --oem 1 -l eng tsv
```
- PSM 11 = "sparse text" mode for finding word boxes without heavy layout
- Filter words by confidence (≥60-70), dilate boxes to form text mask
- Image mask = NOT(text mask)

**Option B - Dedicated text detectors:**
- EAST/CRAFT/DB models for robust text detection regardless of background
- More accurate on captions and wrap-around text
- Requires shipping model files

**Why this was better but still not enough:**
- Much more robust than image-first detection
- Handled white-on-black text and complex backgrounds well
- Still needed different PSM modes for different text types
- Lacked semantic understanding of document structure - couldn't distinguish titles from body text

### The Breakthrough: Semantic Document Understanding

Both approaches were still **reactive** - working around Tesseract's limitations rather than giving it properly structured input. What we needed was semantic document layout analysis.

¹ *Ptak, R., Zygadlo, B., Unold, O. "Projection–Based Text Line Segmentation with a Variable Threshold." International Journal of Applied Mathematics and Computer Science, 27:195-206, 2017.*

² *Dos Santos, R., Clemente, G., Ren, T., Calvalcanti, G. "Text Line Segmentation Based on Morphology and Histogram Projection." 2009.*

³ *Schneider, P., Maurer, Y. "Rerunning OCR: A Machine Learning Approach to Quality Assessment and Enhancement Prediction." National Library of Luxembourg, 2020.*

## The DocLayout-YOLO Solution

We broke the problem into three clean steps, each handled by its own component:

### Step 1: Get the Document Oriented Correctly

**Component**: `RotationDetector` ([genealogy/rotation_detector.py](../genealogy/rotation_detector.py))

Before we can analyze layout, we need the document right-side-up. Our rotation detector uses a two-stage approach:

**Coarse Detection**: Use Tesseract's built-in Orientation and Script Detection (OSD) to catch major rotations (0° or 180°):

```python
def detect_coarse_rotation(self, image_pil: Image.Image) -> int:
    # Let Tesseract handle the heavy lifting for major rotations
    try:
        osd = pytesseract.image_to_osd(image_pil, output_type=pytesseract.Output.DICT)
        angle = osd['rotate']
        # We only care about 0 or 180 degree rotations
        return angle if angle in [0, 180] else 0
    except Exception:
        return 0  # Fallback to no rotation
```

**Fine Correction**: Use GPU-accelerated projection profiles to detect and correct small angle adjustments (-3° to +3° in 0.1° increments):

```python
def detect_fine_rotation_gpu(self, image_tensor: torch.Tensor) -> float:
    # Test 61 angles from -3 to +3 degrees (0.1° precision)
    angles = torch.linspace(-3, 3, 61, device=device)

    best_angle = 0.0
    max_variance = 0.0

    for angle in angles:
        # Rotate and calculate horizontal projection variance
        rotated = kornia.geometry.transform.rotate(image_tensor, angle)
        inverted = 1.0 - rotated.squeeze()  # Invert for projection
        horizontal_proj = torch.sum(inverted, dim=1)
        variance = torch.var(horizontal_proj)

        if variance > max_variance:
            max_variance = variance
            best_angle = angle.item()

    # Only return meaningful corrections (> 0.05 degrees)
    return best_angle if abs(best_angle) > 0.05 else 0.0
```

This gives us the best of both worlds: reliable major angle detection from Tesseract's proven algorithms, plus fine-tuned correction using modern GPU techniques.

### Step 2: Understand the Document Layout

**Component**: `DocumentLayoutDetector` ([genealogy/document_layout_detector.py](../genealogy/document_layout_detector.py))

Once we have a properly oriented document, we need to understand its structure. This is where DocLayout-YOLO comes in - a deep learning model specifically trained for document layout analysis.

Instead of making Tesseract guess where text regions are, we use a computer vision model that can intelligently identify:

- Text blocks (paragraphs, genealogical entries)
- Titles and headers
- Tables and structured data
- Figures and images
- Captions and annotations

The model gives us precise bounding boxes for each detected element:

```python
def detect_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
    # Save image to temporary file for YOLO processing
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        rgb_image = image.convert('RGB')
        rgb_image.save(temp_file.name, 'JPEG')

        # Run DocLayout-YOLO inference
        results = self.model(temp_file.name)

        regions = []
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                element_type = self.model.names[class_id]

                regions.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'element': element_type
                })

        return regions
```

The beauty of this approach is that we get semantic understanding of the document structure. We know which regions are main text versus marginalia, which makes a huge difference for genealogical documents.

### Step 3: Smart Text Extraction and Ordering

**Component**: `RegionOCRProcessor` ([genealogy/region_ocr_processor.py](../genealogy/region_ocr_processor.py))

Now that we know where the text regions are and what type they are, we can run OCR on each region individually and then intelligently stitch the results together.

**Individual Region OCR**: Each detected region gets processed with Tesseract using consistent settings:

```python
def _extract_text_from_regions(self, image: Image.Image, regions: List[Dict[str, Any]]):
    ocr_results = []

    for idx, region in enumerate(regions):
        bbox = region['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        region_image = image.crop((x1, y1, x2, y2))

        # Run OCR with consistent configuration
        text = pytesseract.image_to_string(
            region_image,
            lang=self.tesseract_language,  # "eng+nld" for English+Dutch
            config="--oem 2 --psm 6"       # LSTM+legacy, single block
        ).strip()

        # Get confidence scores
        tsv_data = pytesseract.image_to_data(region_image, ...)
        confidences = [int(conf) for conf in tsv_data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        ocr_results.append({
            'text': text,
            'ocr_confidence': avg_confidence,
            'bbox': bbox,
            'element_type': region['element']
        })
```

**Smart Text Ordering**: Here's where we get clever about genealogical document structure. We separate main content from inset content based on horizontal position:

```python
def _stitch_text_results(self, ocr_results: List[Dict[str, Any]]) -> str:
    # Calculate page width threshold (20% from left edge)
    max_x = max(result['bbox'][2] for result in ocr_results)
    page_width_threshold = max_x * 0.2  # 20% of page width

    main_content = []
    inset_content = []

    for result in ocr_results:
        x_start = result['bbox'][0]
        if x_start <= page_width_threshold:
            main_content.append(result)  # Main text, genealogical entries
        else:
            inset_content.append(result)  # Margin notes, annotations

    # Sort by reading order (top-to-bottom, left-to-right)
    def reading_order_key(result):
        bbox = result['bbox']
        return (bbox[1], bbox[0])  # (y1, x1)

    main_sorted = sorted(main_content, key=reading_order_key)
    inset_sorted = sorted(inset_content, key=reading_order_key)

    # Process main content first, then inset content
    return self._combine_and_deduplicate(main_sorted + inset_sorted)
```

**Text Deduplication**: Sometimes regions overlap or contain similar text. We intelligently remove duplicates:

```python
def _deduplicate_text_lines(self, text_lines: List[str]) -> List[str]:
    # Remove exact duplicates
    unique_lines = []
    seen = set()

    for line in text_lines:
        line_stripped = line.strip()
        if line_stripped and line_stripped not in seen:
            seen.add(line_stripped)
            unique_lines.append(line)

    # Remove near-duplicates (substrings of other lines)
    filtered_lines = []
    for i, line1 in enumerate(unique_lines):
        is_substring = False
        line1_clean = line1.strip()

        for j, line2 in enumerate(unique_lines):
            if i != j:
                line2_clean = line2.strip()
                # If line1 is a substring of line2 and significantly shorter
                if (line1_clean in line2_clean and
                    len(line1_clean) < len(line2_clean) * 0.8):
                    is_substring = True
                    break

        if not is_substring:
            filtered_lines.append(line1)

    return filtered_lines
```

## Putting It All Together

The three components work together in our Celery task to create a robust OCR pipeline:

```python
def process_page_ocr(page_id: str):
    # Get the page and validate
    page = DocumentPage.objects.get(id=page_id)
    language = page.document.languages or "eng+nld"

    # Step 1: Rotation correction
    rotation_detector = RotationDetector()
    image = Image.open(page.image_file.path)
    corrected_image, rotation_applied = rotation_detector.detect_and_correct(image)

    # Step 2: Layout detection
    layout_detector = DocumentLayoutDetector()
    regions = layout_detector.detect_regions(corrected_image)

    # Step 3: Smart OCR processing
    ocr_processor = RegionOCRProcessor(tesseract_language=language)
    text, confidence = ocr_processor.process_regions(corrected_image, regions)

    # Save results
    page.ocr_text = text
    page.ocr_confidence = confidence
    page.rotation_applied = rotation_applied
    page.ocr_completed = True
    page.save()

    return {
        "success": True,
        "text": text,
        "confidence": confidence,
        "rotation_applied": rotation_applied
    }
```

## Why This Works Better

**Modularity**: Each component has a single responsibility and can be tested independently. When something goes wrong, we know exactly where to look.

**Semantic Understanding**: DocLayout-YOLO gives us intelligent layout detection instead of blind text region guessing.

**Smart Ordering**: We understand genealogical document structure and process main content separately from annotations.

**GPU Acceleration**: Fine rotation detection uses modern hardware for speed and accuracy.

**Confidence Tracking**: Each step provides confidence metrics, weighted by text content volume.

## Configuration

The pipeline requires one environment variable:

```bash
# Path to DocLayout-YOLO model file
DOCLAYOUT_MODEL_PATH=/app/models/doclayout_yolo_docstructbench_imgsz1280_2501.pt
```

Component defaults work well for genealogical documents:
- **Tesseract Language**: `"eng+nld"` (English + Dutch)
- **Tesseract Config**: `"--oem 2 --psm 6"` (LSTM + legacy, single block)
- **Page Width Threshold**: `0.2` (20% from left edge separates main vs inset content)

## The Results

This modular approach solved our original problems:

- **Consistent text extraction**: Layout-aware processing eliminates structural confusion
- **Better debugging**: Component isolation makes issues easy to track down
- **Improved accuracy**: Semantic document understanding beats blind text detection
- **Maintainable code**: Each component has clear responsibilities and interfaces

Instead of fighting with a monolithic black box, we now have a pipeline that understands genealogical documents and processes them intelligently.
