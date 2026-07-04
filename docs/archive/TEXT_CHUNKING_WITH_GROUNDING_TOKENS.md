# Text Chunking Strategy with DeepSeek-OCR Grounding Tokens

## Overview

DeepSeek-OCR with `preserve_layout=True` provides rich structural information via:
1. **Grounding tokens**: `<|ref|>type<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>`
2. **Markdown formatting**: Headers (`##`), lists, tables, emphasis, etc.

This is a HUGE improvement over the fragile Tesseract-based chunking we had before.

## Grounding Token Types Found

From analyzing pages 20, 21, 22, 27, 30:

### Content Types
- `<|ref|>text<|/ref|>` - Regular paragraph text
- `<|ref|>sub_title<|/ref|>` - Section headers (often marked with `##` in markdown)
- `<|ref|>image<|/ref|>` - Images/photos
- `<|ref|>image_caption<|/ref|>` - Image captions (often wrapped in `<center>` tags)
- `<|ref|>table<|/ref|>` - Tables (with full HTML structure)

### Bounding Boxes
Format: `<|det|>[[x1, y1, x2, y2]]<|/det|>`
- Coordinates appear to be in pixels
- Can be used to determine:
  - Vertical ordering (y1, y2)
  - Horizontal position (x1, x2)
  - Column detection (for multi-column layouts)
  - Inset content vs main content

## Genealogical Entry Patterns

### Pattern 1: List Entry (Child in family group)
```
<|ref|>text<|/ref|><|det|>[[161, 417, 686, 434]]<|/det|>
f. Arie van Zanten, * Naarden 3.10.1829, † Naarden 13.10.1829,
```

**Characteristics:**
- Starts with letter + period (e.g., "f.", "g.", "h.")
- Contains vital statistics (birth *, death †, marriage x)
- Usually single line or short paragraph
- Contains genealogical ID references

### Pattern 2: Long Biographical Entry
```
<|ref|>text<|/ref|><|det|>[[161, 697, 905, 820]]<|/det|>
n. Bessel van Zanten, * Naarden 17.8.1841, † Haarlem 16.2.1911, boerenknecht (1860),
opzichter der fortificatiën, werkman (1872), spoorwegarbeider (1895), spoorwegbearbe (1900),
x 1. Weesp 10.5.1872 Geertruij Voorhaar wed Pieter van den Bergh, schilder, ...
```

**Characteristics:**
- Starts with letter + period
- Very dense biographical facts
- Multiple marriages (x 1., x 2.)
- Occupations in parentheses with years
- Parent references (zv = zoon van, dv = dochter van)
- May span multiple grounding token blocks

### Pattern 3: Narrative Context
```
<|ref|>text<|/ref|><|det|>[[182, 80, 900, 216]]<|/det|>
Beide gasthuizen waren in de 19e eeuw berucht om hun deplorabele toestanden. Met de medische
behandeling was het ronduit slecht gesteld. De Oostenrijkse arts Joseph Speilt, die het Binnen-
en Buitengasthuis in 1852 bezocht, schreef in zijn verslag...
```

**Characteristics:**
- Full sentences, narrative flow
- Historical context, explanations
- No genealogical markers (*, †, x)
- No letter prefixes
- Often multiple paragraphs

### Pattern 4: Section Headers
```
<|ref|>sub_title<|/ref|><|det|>[[184, 607, 285, 621]]<|/det|>
## Bronverwijzing
```

```
<|ref|>sub_title<|/ref|><|det|>[[159, 151, 440, 168]]<|/det|>
## Opzichters der Fortificatien
```

**Characteristics:**
- Marked with `<|ref|>sub_title<|/ref|>`
- Often has `##` markdown prefix
- Introduces new section
- Short, capitalized

### Pattern 5: Source Citations
```
<|ref|>text<|/ref|><|det|>[[184, 620, 605, 634]]<|/det|>
RGV SSANO 16.3 Bevolkingsregister Naarden 1850-1862 dl 1, bl 156
```

**Characteristics:**
- Follows "Bronverwijzing" section
- Archive codes (RGV, SSANO, etc.)
- Document names with dates
- Page/volume references (bl, dl)

### Pattern 6: Images & Captions
```
<|ref|>image<|/ref|><|det|>[[211, 85, 760, 710]]<|/det|>


<|ref|>image_caption<|/ref|><|det|>[[320, 714, 644, 725]]<|/det|>
<center>Bessel van Zanten en Geertruij Voorhaar. 1872.</center>
```

**Characteristics:**
- Image block (often just whitespace in text)
- Caption with `<center>` tags
- Names, dates, descriptions

## Chunking Strategy

### Strategy 1: Grounding-Token-Based Chunking
**Best for:** Preserving document structure

```python
def chunk_by_grounding_tokens(ocr_text: str) -> List[Chunk]:
    """
    Split text into chunks at grounding token boundaries.
    Each chunk = one grounding token block + its content.
    """
    pattern = r'<\|ref\|>([^<]+)<\|/ref\|><\|det\|>\[\[([^\]]+)\]\]<\|/det\|>\s*\n(.*?)(?=<\|ref\||$)'

    chunks = []
    for match in re.finditer(pattern, ocr_text, re.DOTALL):
        element_type = match.group(1)  # 'text', 'sub_title', 'image', etc.
        bbox = parse_bbox(match.group(2))  # [x1, y1, x2, y2]
        content = match.group(3).strip()

        chunks.append({
            'type': element_type,
            'bbox': bbox,
            'content': content,
            'y_position': bbox[1]  # For vertical ordering
        })

    # Sort by vertical position
    chunks.sort(key=lambda c: c['y_position'])
    return chunks
```

### Strategy 2: Semantic Chunking with Type Detection
**Best for:** Entity extraction

```python
def chunk_by_genealogy_type(chunks: List[Chunk]) -> List[SemanticChunk]:
    """
    Combine grounding token chunks into semantic units:
    - FAMILY_GROUP: Header + list of children
    - GENEALOGY_ENTRY: Single biographical entry
    - NARRATIVE_CONTEXT: Explanatory text
    - SOURCE_CITATION: Archive references
    """
    semantic_chunks = []
    current_group = None

    for chunk in chunks:
        if chunk['type'] == 'sub_title':
            # Start new section
            if current_group:
                semantic_chunks.append(current_group)
            current_group = {
                'type': 'SECTION',
                'header': chunk['content'],
                'content_chunks': []
            }

        elif is_genealogy_entry(chunk['content']):
            # Detect list items (e.g., "f. Name, * date...")
            if current_group:
                current_group['content_chunks'].append(chunk)
            else:
                semantic_chunks.append({
                    'type': 'GENEALOGY_ENTRY',
                    'content_chunks': [chunk]
                })

        elif is_narrative(chunk['content']):
            # Narrative context
            if current_group:
                current_group['content_chunks'].append(chunk)
            else:
                semantic_chunks.append({
                    'type': 'NARRATIVE_CONTEXT',
                    'content_chunks': [chunk]
                })

    return semantic_chunks
```

### Strategy 3: Multi-Column Detection
**Best for:** Index pages (pages 94-101)

```python
def detect_columns(chunks: List[Chunk], page_width: int = 1000) -> List[List[Chunk]]:
    """
    Use bbox x-coordinates to detect columns.
    Useful for 6-column index pages.
    """
    # Cluster by x1 position
    columns = []
    current_column = []
    last_x1 = None

    for chunk in sorted(chunks, key=lambda c: (c['bbox'][0], c['bbox'][1])):
        x1 = chunk['bbox'][0]

        if last_x1 is None or abs(x1 - last_x1) < 50:  # Same column
            current_column.append(chunk)
        else:  # New column
            columns.append(current_column)
            current_column = [chunk]

        last_x1 = x1

    if current_column:
        columns.append(current_column)

    return columns
```

## Detection Heuristics

### Is Genealogy Entry?
```python
def is_genealogy_entry(text: str) -> bool:
    """Detect if text is a genealogical list entry."""
    # Starts with letter + period
    if re.match(r'^[a-z]\.\s+[A-Z]', text):
        return True

    # Contains birth/death/marriage markers
    if any(marker in text for marker in ['*', '†', 'x 1.', 'x 2.', 'zv', 'dv']):
        return True

    return False
```

### Is Narrative Context?
```python
def is_narrative(text: str) -> bool:
    """Detect if text is narrative/explanatory."""
    # Full sentences, no genealogical markers
    if '.' in text and not any(marker in text for marker in ['*', '†', 'x 1.', 'zv', 'dv']):
        # Check if it's not a source citation
        if not re.match(r'^[A-Z]{2,}\s+[A-Z]{2,}', text):
            return True
    return False
```

### Is Source Citation?
```python
def is_source_citation(text: str) -> bool:
    """Detect archive/source references."""
    # Starts with archive codes
    if re.match(r'^[A-Z]{2,}\s+[A-Z]{2,}', text):
        return True

    # Contains typical source keywords
    source_keywords = ['Bevolkingsregister', 'Weesregister', 'DTB', 'BS', 'Archiefnummer']
    if any(kw in text for kw in source_keywords):
        return True

    return False
```

## Advantages Over Old Approach

### Before (Tesseract):
- No structural information
- OCR artifacts (random line breaks, spacing issues)
- Had to use fragile heuristics (character position, line counting)
- Lost table structure completely
- Couldn't distinguish image captions from body text

### Now (DeepSeek-OCR):
- ✅ Explicit element types (text, image, table, subtitle)
- ✅ Bounding boxes for precise positioning
- ✅ Markdown formatting preserved (headers, tables, emphasis)
- ✅ Clean text output (no OCR artifacts)
- ✅ Can reconstruct document flow accurately
- ✅ Column detection trivial (x-coordinates)
- ✅ Can merge related chunks intelligently

## Integration with Existing Code

Current code in `genealogy/prompts/examples_extraction.txt` expects:
- `FAMILY GROUP:` header
- `CONTENT:` section with raw text
- LLM extracts entities

New approach:
1. Use grounding tokens to identify section boundaries
2. Classify chunks as GENEALOGY_ENTRY vs NARRATIVE_CONTEXT
3. Send appropriately sized chunks to LLM
4. Include bbox information for provenance tracking
5. Link chunks back to specific regions on page

## Next Steps

1. Implement grounding token parser
2. Build chunk classifier (genealogy entry vs narrative vs source)
3. Test on pages 20-30 (we have ground truth from examples_extraction.txt)
4. Handle edge cases:
   - Multi-paragraph biographical entries
   - Entries that span multiple grounding token blocks
   - Page continuations
5. Build index parser for pages 94-101 (6-column tables with genealogical IDs)

## Notes

- Don't filter out grounding tokens - they're valuable structural metadata!
- Bounding boxes can be used for provenance ("This fact came from coordinates [x,y] on page N")
- The `<|ref|>image<|/ref|>` blocks might be useful for detecting photo pages vs text pages
- `sub_title` blocks are perfect section dividers for family groups
