# Book Structure Notes

## Jan van Bulhius Book Structure

This genealogical book follows a specific structure that's important for understanding the content types and OCR challenges:

### Page Structure:
- **Pages 1-6**: Front matter (foreword, introduction, methodology, abbreviations)
- **Pages 7-78**: Main genealogical content (biographical entries + narrative context)
- **Pages 79-84**: Family trees for selected female lines
- **Pages 85-92**: Single narrative story (English translation of Dutch emigration story about a half-brother of the great-grandfather who left for Canada/USA)
- **Page 93**: Glossary with cultural details
- **Pages 94-101**: 6-column index with genealogical IDs

### Content Types in Main Section (Pages 7-78):
- **GENEALOGY_ENTRY**: Dense biographical facts (birth, death, marriage, occupation)
- **NARRATIVE_CONTEXT**: Explanatory stories, source discussions, family context
- **CONTENT**: Should be minimal - mostly parsing errors or structural content

### Index Structure Challenge (Pages 94-101):
The index is arranged in 6 columns of tiny text:
```
[Surname] [Given name + tussenvoegsel] [ID] | [Surname] [Given name + tussenvoegsel] [ID]
```

Readers scan down the first 3 columns, then jump back to the top for columns 4-6 before moving to the next page. Each person has their unique genealogical identifier (e.g., VIII.3.d).

### Future OCR Improvements:
1. **6-column index parsing**: Table detection, column structure recognition, genealogical ID extraction
2. **Narrative continuation**: Fix chunks that span multiple segments due to 2000-char limits
3. **Generalization**: Avoid hardcoding page numbers - structure should be detected dynamically

### Notes:
- Don't assume page number ranges in code - need to generalize for other genealogical documents
- Index OCR would provide valuable validation data for genealogical ID linking
- Current spaCy-based narrative detection works well for distinguishing content types
