"""Genealogical entity extraction prompts and parsers"""

import os

# Dutch abbreviations reference for extraction
DUTCH_ABBREVIATIONS = """DUTCH GENEALOGICAL ABBREVIATIONS:
* or geb. = geboren (born)
~ or ged. = gedoopt (baptized)
+ or † or overl. = overleden (died)
begr. = begraven (buried)
x or tr. or ondertr. = getrouwd/ondertrouwd (married)
wednr. or wedn. = weduwnaar (widower)
wed. = weduwe (widow)
dv. = dochter van (daughter of)
zv. = zoon van (son of)
get. or getuige or Doopgetuige = witness (NOT a parent or spouse!)
"""

EVENT_TYPE_CODES = """EVENT TYPE CODES:
BIRT = Birth
DEAT = Death
MARR = Marriage
DIVR = Divorce
BAPT = Baptism
BURI = Burial
RESI = Residence
OCCU = Occupation
EDUC = Education
IMMI = Immigration
EMIG = Emigration
OTHER = Other events
"""


def load_examples():
    """Load extraction examples from text file"""
    examples_path = os.path.join(os.path.dirname(__file__), 'examples_extraction.txt')
    with open(examples_path, 'r', encoding='utf-8') as f:
        return f.read()


def build_extraction_prompt(chunk, examples=None):
    """Build focused event extraction prompt

    NOTE: We no longer extract people or relationships via LLM since we get those
    from genealogical identifiers (more reliable). This prompt focuses ONLY on events.

    Args:
        chunk: TextChunk model instance
        examples: Optional examples text (NOT USED - kept for backward compatibility)

    Returns:
        str: Complete prompt for LLM extraction
    """
    # Build genealogical context
    generation_info = f"Generation {chunk.generation_number}" if chunk.generation_number else "None"
    family_group = chunk.family_groups[0] if chunk.family_groups else "None"

    # Primary person this chunk is about
    primary_person = chunk.subject if chunk.subject else "Unknown"

    return f"""Extract life events from genealogical text.

{DUTCH_ABBREVIATIONS}

{EVENT_TYPE_CODES}

======================================================================
YOUR TASK
======================================================================

CONTEXT:
- Primary Person: {primary_person}
- Generation: {generation_info}
- Family Group: {family_group}

CONTENT TO EXTRACT FROM:
{chunk.text_content}

INSTRUCTIONS:
1. Extract ONLY life events explicitly stated in the text
2. Focus on events for the primary person: {primary_person}
3. Event types to extract:
   - BIRTH: Birth events (*, geb.)
   - DEATH: Death events (+, †, overl.)
   - BAPT: Baptism events (~, ged.)
   - BURI: Burial events (begr.)
   - MARR: Marriage events (x, tr., ondertr.)
   - OCCU: Occupations/professions
   - RESI: Residence/living places
   - EDUC: Education (degrees, schools)
   - IMMI/EMIG: Immigration/Emigration
   - OTHER: Other significant life events
4. For each event, extract:
   - Person name (usually the primary person)
   - Event type code
   - Date (as written in text, or empty if not stated)
   - Place (as written in text, or empty if not stated)
5. If NO events are mentioned, return "None"
6. Do NOT make up information - extract only what is explicitly stated

CRITICAL WARNING - AVOID HALLUCINATION:
- Do NOT copy events from the examples above
- Do NOT infer events that are not in the text
- If the person died as an infant, extract ONLY the birth and death events shown
- Even if a person has the same name as someone in the examples, extract ONLY what is in THIS text
- Short entries (one line) usually mean limited life events - extract only what you see

OUTPUT FORMAT:

EVENTS:
- PersonName|EVENT_CODE|Date|Place|Description (pipe-delimited, or "None")

Field definitions:
- Date: The date of the event (can be full date like "15.3.1850", year like "1860", or empty)
- Place: Geographic location ONLY (city, address) - NOT for occupations
- Description: Additional details - USE THIS for occupations

CRITICAL for OCCU events:
- Put the YEAR in the Date field (3rd position)
- Leave Place EMPTY (4th position)
- Put the OCCUPATION in Description field (5th position)

Example output:
EVENTS:
- Jan van Zanten|BIRTH|15.3.1850|Amsterdam|
- Jan van Zanten|MARR|12.6.1875|Rotterdam|
- Jan van Zanten|OCCU|1860||boerenknecht
- Jan van Zanten|OCCU|1872||werkman
- Jan van Zanten|RESI|1875|lepenlaan 2, Bussum|
- Jan van Zanten|DEATH|3.11.1920|Utrecht|

Extract now:
"""


def parse_extraction_output(output_text):
    """Parse the pipe-delimited output into structured data

    NOTE: We now only extract events via LLM. People and relationships come from
    genealogical identifiers during chunking (more reliable).

    Args:
        output_text: Raw text output from LLM

    Returns:
        dict: Parsed extraction with keys: events (people, parent_child, partnerships are empty)
    """
    lines = output_text.strip().split('\n')

    result = {
        'people': [],
        'parent_child': [],
        'partnerships': [],
        'events': []
    }

    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section headers
        if line.startswith('EVENTS:'):
            current_section = 'events'
            continue
        elif line.startswith('INTERPRETATION:') or line.startswith('PEOPLE:') or line.startswith('PARENT_CHILD:') or line.startswith('PARTNERSHIPS:'):
            current_section = None
            continue

        # Skip "None" entries
        if line.lower() == 'none' or line.lower() == '- none':
            continue

        # Parse events only
        if current_section == 'events':
            if line.startswith('-'):
                parts = line[1:].strip().split('|')
                if len(parts) >= 2:
                    result['events'].append({
                        'person': parts[0].strip() if len(parts) > 0 else '',
                        'event_type': parts[1].strip() if len(parts) > 1 else '',
                        'date': parts[2].strip() if len(parts) > 2 else '',
                        'place': parts[3].strip() if len(parts) > 3 else '',
                        'description': parts[4].strip() if len(parts) > 4 else ''
                    })

    return result
