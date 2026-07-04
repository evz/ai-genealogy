# Status Report: Text Chunk Summarization Feature
**Date:** January 10, 2026
**Status:** Implemented but likely pending data backfill.

## Feature Overview
To optimize the Agent's context window usage, a system was implemented to pre-calculate concise summaries of long genealogical text chunks (narratives, biographies). The search tools prefer these summaries over full text when available.

## Implementation Details

### 1. Database Schema
- **File:** `genealogy/models.py`
- **Change:** Added `text_summary` (TextField) to `TextChunk` model.
- **Migration:** `0046_add_text_summary_to_textchunk.py` (Applied).

### 2. Processing Logic
- **File:** `genealogy/tasks/summarization.py`
- **Engine:** Celery task using Ollama (`llama3.1:8b`).
- **Logic:**
  - Targets chunks > 1000 characters.
  - Generates a summary (~50% reduction).
  - Skips chunks already summarized.

### 3. Consumption
- **File:** `genealogy/services/genealogy_tools.py`
- **Method:** `search_source_text`
- **Behavior:** Returns `text_summary` if populated; falls back to `text_content`. Logs the reduction ratio achieved.

## Outstanding Items
1.  **Data Backfill:** The Celery task `summarize_all_chunks` needs to be executed against the existing database. This requires the Ollama service to be running with `llama3.1:8b` pulled.
2.  **Verification:** No dedicated unit tests were found for the summarization task logic.
3.  **UI:** Frontend likely displays the summary without explicitly labeling it as such (though the agent knows).
