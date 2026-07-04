# Coding Guidelines for genealogy_extractor

## Fail-Fast Philosophy

**Core Principle**: Code should raise exceptions immediately when data quality is compromised. Never silently degrade or return partial results.

## Critical Rules

### 1. Data Pipeline Validation

Always validate data at pipeline boundaries to detect silent data loss:

```python
# BAD - No validation
def process_text(text: str) -> str:
    result = expensive_operation(text)
    return result

# GOOD - Validate output
def process_text(text: str) -> str:
    result = expensive_operation(text)

    # Sanity check: output shouldn't be dramatically smaller than input
    if len(result) < len(text) * 0.5:
        raise ValueError(
            f"Suspicious data loss detected: input={len(text)} chars, "
            f"output={len(result)} chars"
        )

    return result
```

### 2. Never Swallow Exceptions in Critical Paths

```python
# BAD - Silent failure returns partial results
def clean_text(text: str) -> str:
    try:
        cleaned = llm_cleanup(text)
        return cleaned
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return text[:1000]  # Returns truncated data!

# GOOD - Fail loudly
def clean_text(text: str) -> str:
    try:
        cleaned = llm_cleanup(text)
        return cleaned
    except Exception as e:
        # Log for debugging, but always re-raise
        logger.error(f"Cleanup failed: {e}")
        raise RuntimeError(f"Critical cleanup operation failed: {e}") from e
```

### 3. Optional Components Must Not Degrade Quality

If an optional component fails, either skip it entirely or fail the whole operation:

```python
# BAD - Continues with degraded quality
def process_page(image: Image) -> str:
    text = ocr_extract(image)

    try:
        text = optional_cleanup(text)  # Fails but returns partial results
    except Exception as e:
        logger.warning(f"Optional cleanup failed: {e}")
        # Continues with corrupted text!

    return text

# GOOD - Skip optional step entirely if it fails
def process_page(image: Image) -> str:
    text = ocr_extract(image)

    try:
        text = optional_cleanup(text)
    except Exception as e:
        logger.warning(f"Optional cleanup failed, skipping: {e}")
        # Return original text, not partial results
        pass

    return text

# EVEN BETTER - Fail fast if cleanup is actually critical
def process_page(image: Image) -> str:
    text = ocr_extract(image)

    # If cleanup is important enough to include, it's important enough to fail on
    text = optional_cleanup(text)  # Let exceptions propagate

    return text
```

### 4. Validate Assumptions with Assertions

Use assertions to catch unexpected states early:

```python
def chunk_text(text: str, tokens: List[GroundingToken]) -> List[TextChunk]:
    chunks = []

    for token in tokens:
        chunk = create_chunk(token)
        chunks.append(chunk)

    # Validate we didn't lose data
    assert len(chunks) > 0, "No chunks created from tokens"
    assert sum(len(c.content) for c in chunks) >= len(text) * 0.9, \
        "Chunks account for less than 90% of original text - data loss detected"

    return chunks
```

### 5. Logging vs Exceptions

- **Logging**: For informational messages, debugging, performance metrics
- **Exceptions**: For data quality issues, failures, unexpected states

```python
# Use logging for info
logger.info(f"Processing page {page_num}")
logger.debug(f"Image size: {image.size}")

# Use exceptions for problems
if not text:
    raise ValueError(f"OCR returned empty text for page {page_num}")

if len(text) < 100:
    raise ValueError(
        f"OCR returned suspiciously short text ({len(text)} chars) "
        f"for page {page_num}"
    )
```

### 6. External Service Failures

When external services (Ollama, DeepSeek-OCR, etc.) fail, always raise exceptions:

```python
# BAD - Returns None or empty results
def call_llm(prompt: str) -> str | None:
    try:
        return client.generate(prompt)
    except ConnectionError:
        logger.error("LLM service unavailable")
        return None  # Caller might not check!

# GOOD - Fail explicitly
def call_llm(prompt: str) -> str:
    try:
        result = client.generate(prompt)
        if not result:
            raise ValueError("LLM returned empty response")
        return result
    except ConnectionError as e:
        raise RuntimeError(
            f"LLM service unavailable at {client.host}:{client.port}"
        ) from e
```

### 7. Unit Test for Failure Modes

Test that your code raises exceptions when it should:

```python
def test_process_page_detects_data_loss():
    """Test that processing detects when OCR returns truncated data"""
    image = create_test_image()

    # Mock OCR to return suspiciously short text
    with mock.patch('genealogy.ocr.extract_text', return_value="short"):
        with pytest.raises(ValueError, match="data loss detected"):
            process_page(image)
```

## Example: The Small Model Cleanup Bug

**What happened**: The `clean_ocr_genealogy_ids()` function caught an Ollama connection error, logged it, but returned partial results instead of raising an exception. This caused page 25 to lose 60% of its content (6162 chars → 2441 chars) silently.

**What we should have done**:

```python
# Original (BAD)
def clean_ocr_genealogy_ids(text: str) -> str:
    try:
        cleaned = ollama_client.generate(...)
        return cleaned
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return text[:2441]  # Silently returns truncated data!

# Fixed (GOOD)
def clean_ocr_genealogy_ids(text: str) -> str:
    try:
        cleaned = ollama_client.generate(...)

        # Validate we didn't lose data
        if len(cleaned) < len(text) * 0.9:
            raise ValueError(
                f"Cleanup lost data: {len(text)} → {len(cleaned)} chars"
            )

        return cleaned
    except Exception as e:
        # Don't continue with degraded data - fail loudly
        raise RuntimeError(f"Critical cleanup failed: {e}") from e
```

## Checklist for New Code

Before committing code that processes genealogical data:

- [ ] Does this code validate that data hasn't been lost?
- [ ] If an exception occurs, does it propagate up or get re-raised?
- [ ] Are there any `except: pass` or `except: logger.error()` blocks that might hide failures?
- [ ] Does this code return partial/truncated results on failure?
- [ ] Are there assertions checking critical assumptions?
- [ ] If an external service fails, does the code fail fast?

## When to Use Try/Except

Use try/except only when you can actually handle the error:

```python
# GOOD - Actually handles the error by retrying
def fetch_with_retry(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            return requests.get(url).text
        except requests.ConnectionError:
            if attempt == retries - 1:
                raise  # Re-raise on final attempt
            time.sleep(2 ** attempt)

# GOOD - Provides context then re-raises
def process_document(doc_id: str) -> None:
    try:
        doc = Document.objects.get(id=doc_id)
    except Document.DoesNotExist as e:
        raise ValueError(f"Document {doc_id} not found") from e

# BAD - Catches but can't actually handle it
def process_text(text: str) -> str:
    try:
        return expensive_operation(text)
    except Exception as e:
        logger.error(f"Failed: {e}")
        return ""  # What now? This is wrong!
```

## Summary

**Fail fast. Fail loudly. Never silently lose data.**

When in doubt, raise an exception. It's easier to add error handling later than to debug silent data corruption.
