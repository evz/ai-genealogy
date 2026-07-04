# Morphological OCR: A Better Way to Read Genealogy Documents

> **⚠️ OBSOLETE DOCUMENTATION**
> This document describes the morphological OCR approach that was replaced by the DocLayout-YOLO pipeline in September 2025. The current implementation uses three modular components (RotationDetector, DocumentLayoutDetector, RegionOCRProcessor) with deep learning layout detection.
>
> See [DocLayout-YOLO Pipeline Documentation](./doclayout_yolo_pipeline.md) for the current architecture.
>
> This document is preserved for historical reference only.

---

## The Problem with Standard OCR

When you run standard OCR on a genealogical document, you typically get disappointing results. The text might be readable, but the structure gets mangled. Genealogical IDs like "II.1.a" get separated from their biographical entries, margin notes disappear, and sometimes entire columns of text vanish.

Why does this happen? Standard OCR tools like Tesseract use a "one size fits all" approach. They assume your document is either a single column of text, or they try to detect columns automatically. But genealogy documents are more complex - they have margin annotations, mixed text densities, reversed text regions, and spatial relationships that matter for meaning.

We built a different approach that treats OCR more like reading a complex layout: first understand the page structure, then read each part with the right technique.

## How Morphological Segmentation Works

Instead of throwing the whole page at Tesseract and hoping for the best, we break the process into clear steps. The main implementation is in [`genealogy/ocr_processor.py`](../genealogy/ocr_processor.py), specifically the `_process_with_segmentation()` method.

### Step 1: Clean Up the Image

Before we try to understand the layout, we need a clean image to work with. This happens in the [`_normalize_image()`](../genealogy/ocr_processor.py#L253) method.

**Deskewing**: Documents are often scanned at slight angles. We detect skew by testing different rotation angles and finding the one that creates the most consistent horizontal lines of text:

```python
def _deskew_image(self, img_array: np.ndarray) -> np.ndarray:
    # Test angles from -2 to +2 degrees in 0.1° increments
    best_angle = 0
    max_variance = 0

    for angle in np.arange(-2, 2.1, 0.1):
        rotated = ndimage.rotate(img_array, angle, reshape=False, cval=255)
        proj = np.sum(255 - rotated, axis=1)  # Horizontal projection
        variance = np.var(proj)

        if variance > max_variance:
            max_variance = variance
            best_angle = angle

    # Only rotate if significant skew detected
    if abs(best_angle) > 0.1:
        return ndimage.rotate(img_array, best_angle, reshape=False, cval=255)
    return img_array
```

**Adaptive Binarization**: Instead of using a single threshold for the whole page, we use Sauvola thresholding in [`_sauvola_binarization()`](../genealogy/ocr_processor.py#L363):

```python
def _sauvola_binarization(self, img_array: np.ndarray) -> np.ndarray:
    # Use scikit-image's threshold_sauvola
    threshold = filters.threshold_sauvola(img_array, window_size=25, k=0.2)
    binary = img_array > threshold
    return (binary * 255).astype(np.uint8)
```

This analyzes each pixel's neighborhood and sets an adaptive threshold. It handles variations in lighting and paper quality much better than global methods.

**Image Region Detection**: Photos and graphics can confuse text detection algorithms. Our [`_remove_image_regions()`](../genealogy/ocr_processor.py#L390) method identifies high-variance regions and masks them out:

```python
def _remove_image_regions(self, binary_img: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    # Compute local variance using a 25x25 kernel
    kernel = np.ones((25, 25)) / (25 ** 2)
    mean = cv2.filter2D(original_img.astype(np.float32), -1, kernel)
    sqr_mean = cv2.filter2D((original_img.astype(np.float32)) ** 2, -1, kernel)
    variance = sqr_mean - mean ** 2

    # Mask out high variance regions (photos vs text)
    var_threshold = np.percentile(variance, 85)  # Top 15% variance
    high_var_mask = variance > var_threshold

    result = binary_img.copy()
    result[high_var_mask > 0] = 255  # Set to white (background)
    return result
```

### Step 2: Find Text Regions Using Connected Components

This is where the "morphological" part comes in, implemented in [`_segment_page_regions()`](../genealogy/ocr_processor.py#L280). We use mathematical morphology to analyze the shape and structure of objects in the image.

```python
def _segment_page_regions(self, binary_img: np.ndarray) -> list:
    # Horizontal closing: connect words into lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    h_closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, h_kernel)

    # Vertical dilation: connect lines into paragraphs
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    v_dilated = cv2.morphologyEx(h_closed, cv2.MORPH_DILATE, v_kernel)

    # Find contours for regions
    contours, _ = cv2.findContours(255 - v_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**Horizontal Closing**: The `(15, 1)` kernel connects nearby letters into words, and words into lines. Think of it as intelligent connection - it bridges small gaps between text elements horizontally.

**Vertical Dilation**: The `(1, 15)` kernel connects related lines into paragraph-like blocks. Lines that are close vertically get grouped together.

The key insight here is that we're letting the actual layout of the text determine our regions, rather than making assumptions about column structure.

### Step 3: Classify Each Region

Now that we have distinct text regions, we need to figure out what kind of text each one contains. This logic is in [`_classify_region()`](../genealogy/ocr_processor.py#L420):

```python
def _classify_region(self, x: int, y: int, w: int, h: int, binary_img: np.ndarray, margin_threshold: float) -> str:
    page_width = binary_img.shape[1]

    # Check if region is in margin area
    is_left_margin = x < margin_threshold  # margin_threshold = page_width * 0.15
    is_right_margin = (x + w) > (page_width - margin_threshold)

    # Check if region is narrow (likely marginalia)
    is_narrow = w < (page_width * 0.3)

    # Check mean intensity (for reversed text detection)
    region = binary_img[y:y+h, x:x+w]
    mean_intensity = np.mean(region)
    is_reversed = mean_intensity < 128  # Dark background

    if is_reversed:
        return 'reversed_block'
    elif (is_left_margin or is_right_margin) and is_narrow:
        return 'marginalia'  # This is our genealogical IDs!
    elif w < (page_width * 0.5) and h < 100:
        return 'caption_sidebar'
    else:
        return 'main_body'
```

**Marginalia Detection**: We identify genealogical IDs and margin notes by looking for regions that are:
- Located in the outer 15% of the page width (left or right margins)
- Narrow (less than 30% of the page width)
- Both conditions must be true to avoid false positives

### Step 4: OCR Each Region with Optimal Settings

Here's where we get technical about Tesseract configuration in [`_ocr_region()`](../genealogy/ocr_processor.py#L144). Different types of content need different PSM (Page Segmentation Mode) settings:

```python
def _ocr_region(self, normalized_img: np.ndarray, region: dict) -> tuple[str, float]:
    region_type = region['type']

    # Select PSM mode based on region type
    if region_type == 'marginalia':
        config = "--oem 2 --psm 7 -c preserve_interword_spaces=1"  # Single text line
    elif region_type == 'caption_sidebar':
        config = "--oem 2 --psm 4"  # Single column, variable size
    elif region_type == 'reversed_block':
        config = "--oem 2 --psm 6 -c tessedit_do_invert=0"  # Uniform block, no auto-invert
    else:  # main_body
        config = "--oem 2 --psm 6 -c preserve_interword_spaces=1 -c textord_tablefind_recognize_tables=0 -c textord_tabfind_find_tables=0"

    # Extract text with the optimal configuration
    text = pytesseract.image_to_string(pil_region, lang=self.language, config=config)
    return text, confidence
```

**Marginalia gets PSM 7** - "Single text line mode". Perfect for genealogical IDs like "II.1.a" or "b." because it doesn't try to find multiple columns.

**Main body text gets PSM 6** - "Single uniform block of text". We also disable table detection because Tesseract sometimes sees consistent paragraph structure as table columns.

**All regions use OEM 2** - This combines legacy Tesseract with newer LSTM neural networks for more deterministic results.

### Step 5: Put It Back Together Intelligently

The final challenge is reconstructing the reading order in [`_combine_regions_in_reading_order()`](../genealogy/ocr_processor.py#L193). We can't just concatenate regions randomly - we need to understand spatial relationships.

```python
def _combine_regions_in_reading_order(self, region_results: list) -> str:
    # Separate marginalia from other regions
    marginalia = [r for r in region_results if r['region']['type'] == 'marginalia']
    other_regions = [r for r in region_results if r['region']['type'] != 'marginalia']

    # Sort main regions by top (y), then left (x)
    other_regions.sort(key=lambda r: (r['region']['bbox'][1], r['region']['bbox'][0]))

    combined_parts = []

    for region_result in other_regions:
        region_bbox = region_result['region']['bbox']
        region_y = region_bbox[1]
        region_height = region_bbox[3]
        region_y_end = region_y + region_height

        # Find marginalia that vertically aligns with this region
        attached_marginalia = []
        for marg in marginalia:
            marg_y = marg['region']['bbox'][1]
            marg_height = marg['region']['bbox'][3]
            marg_y_center = marg_y + marg_height // 2

            # Check for y-overlap or proximity (50px tolerance)
            if region_y <= marg_y_center <= region_y_end or abs(marg_y_center - region_y) < 50:
                attached_marginalia.append(marg)

        # Sort marginalia by x position (left to right)
        attached_marginalia.sort(key=lambda m: m['region']['bbox'][0])

        # Combine: marginalia first, then main content
        if attached_marginalia:
            marg_texts = [m['text'].strip() for m in attached_marginalia if m['text'].strip()]
            if marg_texts and region_text:
                combined_text = " ".join(marg_texts) + " " + region_text
```

**Marginalia Association**: For each main content region, we look for marginalia that vertically aligns with it. We check if the center of a margin note falls within the vertical bounds of a main content region, or within 50 pixels of it.

**Text Assembly**: When we find marginalia that associates with main content, we put the margin text first: `"II.1.a Johan van der Berg, geboren 15 maart 1654..."` This preserves the genealogical structure.

## Why This Should Work Better

### The Column Problem

Standard Tesseract PSM modes often interpret margin text as separate columns. When you have text like:

```
II.1.a    Johan van der Berg, born March 15, 1654...
b.        Maria Janssen, born December 22, 1658...
```

Standard OCR sees this as three columns: the genealogical IDs, whitespace, and the biographical text. Our morphological approach should keep related text together during the connected components analysis phase.

### The Confidence Problem

Standard OCR gives you one confidence score for the whole page. We extract confidence scores for each region separately using [`_get_confidence_from_tsv()`](../genealogy/ocr_processor.py#L81):

```python
def _get_confidence_from_tsv(self, tsv_data: str) -> float:
    lines = tsv_data.strip().split('\n')[1:]  # Skip header
    confidences = []

    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 11 and parts[10].strip():  # conf column
            conf = int(parts[10])
            if conf > 0:  # Only valid confidence scores
                confidences.append(conf)

    return sum(confidences) / len(confidences) if confidences else 0.0
```

This should let us identify which parts of the document might need manual review. Marginalia processed with PSM 7 should theoretically have higher confidence than the same text processed with PSM 1.

## Testing the Approach

We have a test command [`test_ocr_configurations.py`](../genealogy/management/commands/test_ocr_configurations.py) that can compare different approaches:

```python
# Test different PSM modes
all_configs = {
    'current': ("Current (PSM 1)", "--oem 3 --psm 1"),
    'default': ("PSM 3 (Default)", "--oem 3 --psm 3"),
    'column': ("PSM 4 (Single column)", "--oem 3 --psm 4"),
    'uniform': ("PSM 6 (Uniform block)", "--oem 3 --psm 6"),
}

# The new morphological approach is now the default when using OCRProcessor()
processor = OCRProcessor(language="eng+nld")
```

To test the new approach against the old ones:

```bash
python manage.py test_ocr_configurations --pages 7 11 --verbose
```

This will show you how many genealogical markers each approach detects and the confidence scores for comparison.

## Technical Requirements

This approach requires additional image processing libraries, specified in our [`requirements.txt`](../requirements.txt):

```
opencv-python==4.10.0.84  # Morphological operations and contour detection
scipy==1.14.1             # Image transformations like rotation
scikit-image==0.24.0      # Advanced thresholding algorithms
```

The Docker setup in [`Dockerfile`](../Dockerfile) includes the necessary system libraries:

```dockerfile
# OS-level dependencies for OpenCV and scientific computing
RUN apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev
```

## How to Use It

The new OCR processor is integrated into the existing pipeline in [`genealogy/tasks.py`](../genealogy/tasks.py):

```python
def process_page_ocr(self, page_id: str):
    # Get language from document
    language = page.document.languages

    # Initialize OCR processor with morphological segmentation
    processor = OCRProcessor(language=language)

    # Process the image file - this now uses morphological segmentation automatically
    text, confidence, rotation = processor.process_file(file_path)
```

The system automatically determines the optimal processing approach based on the detected layout. No configuration needed!

## Edge Cases and Robustness

The system is designed to fail gracefully:

**No marginalia present**: The system just classifies everything as main body text and processes it normally with PSM 6.

**False marginalia detection**: Incorrectly identified marginalia still appears in the output, just potentially in the wrong order. The text remains readable.

**Complex layouts**: The morphological approach should handle irregular layouts better than column-assumption methods because it works from the actual text positioning rather than predetermined grid structures.

**Segmentation failures**: The system raises exceptions rather than falling back to potentially poor results, so issues can be identified and fixed.

You can test edge cases using the admin interface or the [`test_ocr_configurations.py`](../genealogy/management/commands/test_ocr_configurations.py) management command.

## Advanced OCR Enhancement: Lessons from NautilusOCR

Our research into the National Library of Luxembourg's NautilusOCR project reveals significant opportunities to enhance our current morphological approach. NautilusOCR is the production implementation by the same researchers who developed CombiSeg, providing authoritative insights into advanced OCR processing for historical documents.

### Key Findings from NautilusOCR Research

**Machine Learning Quality Assessment**: The Luxembourg team developed regression models that can:
- Predict which text blocks will benefit most from reprocessing
- Assess OCR quality without ground truth data
- Estimate enhancement potential of different OCR engines
- Process 566,000 newspaper pages in 15 days with <5% overhead

**Advanced Feature Analysis**: Their quality classifier uses sophisticated text analysis:
- **Dictionary Mapping**: Weighted character-based matching against language dictionaries
- **N-gram Analysis**: Tri-gram frequency analysis based on Zipfian distribution
- **Garbage Token Detection**: 9-rule system identifying problematic OCR outputs
- **Publication Year Consideration**: Time-based quality adjustments

**Multi-Engine Training**: They train models on outputs from multiple OCR engines, creating robust quality prediction across different processing approaches.

### Current Limitations in Our Image Masking

While our morphological segmentation works well for text layout, we still struggle with **image region detection**. The NautilusOCR research reveals that they don't focus heavily on image masking because they work at the "text block" level - assuming layout analysis has already separated text from images.

However, their **quality-based processing approach** could solve our image masking problems:

```python
# Current approach: uniform text masking
if confidence >= 60:
    mask_as_text(region)
else:
    mask_as_image(region)

# NautilusOCR-inspired approach: quality-guided decisions
quality_score = assess_region_quality(region)
content_type = classify_content_type(region)

if content_type == 'pure_text' and quality_score > 0.8:
    minimal_masking(region)
elif content_type == 'mixed_content':
    selective_masking(region, quality_score)
elif content_type == 'pure_image':
    complete_masking(region)
else:
    adaptive_masking(region, quality_score)
```

### Proposed Enhancements

**1. Content-Type Classification Model**
Train a classifier to distinguish:
- Pure text regions (genealogical entries, main body text)
- Mixed text/image regions (newspaper clippings, annotated photos)
- Pure image regions (photographs, graphics)
- Degraded/noisy regions (faded text, artifacts)

**2. Quality-Guided Processing Pipeline**
```python
def enhanced_region_processing(self, region, binary_img):
    # Step 1: Content type classification
    content_type = self.classify_content_type(region, binary_img)

    # Step 2: Quality assessment using multiple features
    quality_features = self.extract_quality_features(region)
    quality_score = self.quality_model.predict(quality_features)

    # Step 3: Processing strategy selection
    if content_type == 'marginalia' and quality_score > 0.7:
        return self.process_high_quality_marginalia(region)
    elif content_type == 'main_text' and quality_score < 0.5:
        return self.process_degraded_text(region)
    else:
        return self.process_standard_text(region)
```

**3. Multi-Feature Text Detection**
Beyond our current sparse text detection, implement:
- Language-aware dictionary matching
- Character pattern analysis (identifying genealogical markers vs. noise)
- Spatial relationship analysis (margin text vs. body text)
- Historical document specialization (adapting to document age/quality)

**4. Training Data Generation**
Following NautilusOCR's approach:
- Create "ground truth" examples of well-processed vs. poorly-processed regions
- Generate augmented training data with different quality levels
- Build language-specific dictionaries for genealogical terminology
- Develop genealogy-specific "garbage token" detection rules

### Implementation Priority

**Phase 1: Quality Assessment Integration**
- Implement basic quality scoring for text regions
- Use quality scores to guide masking aggressiveness
- Test on problematic pages (Pages 27, 30 with newspaper images)

**Phase 2: Content-Type Classification**
- Train a model to distinguish text blocks from image blocks
- Implement selective masking based on content type
- Handle mixed content regions more intelligently

**Phase 3: Advanced Feature Analysis**
- Add dictionary-based quality assessment for genealogical terms
- Implement n-gram analysis for Dutch/German historical text
- Develop genealogy-specific garbage token detection

### Benchmarking Against Production Systems

The NautilusOCR implementation processes **102,000 historical newspaper issues** with:
- Multiple languages (German, French, Luxembourgish)
- Mixed typography (Antiqua and Fraktur fonts)
- Variable quality (1841-1954 publication dates)
- 33k+ transcribed training examples

This scale demonstrates that machine learning approaches to OCR quality assessment are production-ready for cultural heritage institutions.

### Integration with Current System

Our morphological segmentation provides an excellent foundation for NautilusOCR-inspired enhancements:

```python
# Enhanced OCR processor workflow
def _process_with_enhanced_segmentation(self, binary_img):
    # Step 1: Current morphological segmentation
    regions = self._segment_page_regions(binary_img)

    # Step 2: Enhanced classification with quality assessment
    enhanced_regions = []
    for region in regions:
        region_type = self._classify_region_enhanced(region, binary_img)
        quality_score = self._assess_region_quality(region, binary_img)
        processing_strategy = self._select_processing_strategy(region_type, quality_score)

        enhanced_regions.append({
            'region': region,
            'type': region_type,
            'quality': quality_score,
            'strategy': processing_strategy
        })

    # Step 3: Quality-guided OCR processing
    results = []
    for enhanced_region in enhanced_regions:
        if enhanced_region['quality'] > 0.8:
            # High quality: minimal preprocessing, optimal PSM
            result = self._ocr_high_quality_region(enhanced_region)
        elif enhanced_region['quality'] < 0.3:
            # Low quality: aggressive preprocessing, fallback methods
            result = self._ocr_degraded_region(enhanced_region)
        else:
            # Standard quality: current approach
            result = self._ocr_region(enhanced_region)

        results.append(result)

    return results
```

## Future Directions

This research opens up several sophisticated enhancement opportunities:

- **Machine Learning Quality Assessment**: Implement NautilusOCR-inspired quality prediction models
- **Advanced Image/Text Classification**: Move beyond simple confidence thresholds to content-type classification
- **Multi-Language Dictionary Integration**: Leverage genealogical terminology for quality assessment
- **Historical Document Specialization**: Adapt processing based on document age, font type, and degradation patterns
- **Production-Scale Processing**: Follow Luxembourg's approach to handle large document collections efficiently
- **Enhancement Prediction Models**: Predict which processing strategies will yield the best results for specific content types

## Implementation Notes

The complete implementation is in [`genealogy/ocr_processor.py`](../genealogy/ocr_processor.py). Key methods:

- [`_process_with_segmentation()`](../genealogy/ocr_processor.py#L103): Main orchestration method
- [`_normalize_image()`](../genealogy/ocr_processor.py#L253): Image preprocessing pipeline
- [`_segment_page_regions()`](../genealogy/ocr_processor.py#L280): Morphological segmentation
- [`_classify_region()`](../genealogy/ocr_processor.py#L420): Region type detection
- [`_ocr_region()`](../genealogy/ocr_processor.py#L144): Region-specific OCR processing
- [`_combine_regions_in_reading_order()`](../genealogy/ocr_processor.py#L193): Spatial text reconstruction

The approach replaces the previous two-pass OCR system and is automatically used for all new document processing. The old `tesseract_config` field has been removed from the DocumentPage model since configuration is now determined automatically based on detected regions.

## Next Steps

To validate this approach:

1. **Run comparative tests** using the test command on your existing genealogy documents
2. **Compare genealogical marker detection** between old and new approaches
3. **Analyze confidence scores** to see if region-specific processing improves reliability
4. **Check text quality** to ensure no regressions in readability

The theoretical advantages should translate to practical improvements, but testing on real documents will confirm the actual performance gains.
