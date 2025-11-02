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
    """Build the unified example-based extraction prompt

    Args:
        chunk: TextChunk model instance
        examples: Optional examples text (will load from file if not provided)

    Returns:
        str: Complete prompt for LLM extraction
    """
    if examples is None:
        examples = load_examples()

    # Build genealogical context
    generation_info = f"Generation {chunk.generation_number}" if chunk.generation_number else "None"
    family_group = chunk.family_groups[0] if chunk.family_groups else "None"

    # Show Phase 1 extracted data (deterministic extraction from chunking)
    phase1_people = chunk.extracted_people if chunk.extracted_people else []
    phase1_relationships = chunk.extracted_relationships if chunk.extracted_relationships else []

    # Format Phase 1 data for display
    if phase1_people:
        phase1_people_str = ", ".join(phase1_people)
    else:
        phase1_people_str = "None"

    if phase1_relationships:
        phase1_rels_str = f"{len(phase1_relationships)} parent-child relationships"
    else:
        phase1_rels_str = "None"

    return f"""Extract genealogical information from Dutch genealogy text.

{DUTCH_ABBREVIATIONS}

{EVENT_TYPE_CODES}

EXAMPLES:

{examples}

======================================================================
YOUR TASK
======================================================================

CONTEXT:
- Generation: {generation_info}
- Family Group: {family_group}

ALREADY EXTRACTED (Phase 1):
- People: {phase1_people_str}
- Relationships: {phase1_rels_str}

CONTENT TO EXTRACT FROM:
{chunk.text_content}

INSTRUCTIONS:
1. Extract ONLY information explicitly stated in the content above
2. Focus on:
   - Life events: births, deaths, baptisms, burials, marriages
   - Occupations: any profession, job, or occupation mentioned (use OCCU event type)
   - Residences: places where people lived (use RESI event type)
   - Partnerships: marriages and relationships
   - Additional people mentioned
3. If the content has NO genealogical facts (just narrative/acknowledgments), return "None" for all sections
4. Do NOT copy the examples or Phase 1 data
5. For occupations, extract the occupation as an OCCU event with the person's name

OUTPUT FORMAT:

PEOPLE:
- List people mentioned (one per line, or "None")

PARENT_CHILD:
- PersonA|child|PersonB (or "None")

PARTNERSHIPS:
- PersonA|spouse|PersonB (or "None")

EVENTS:
- PersonName|EVENT_CODE|Date|Place (or "None")

Extract now:
"""


def parse_extraction_output(output_text):
    """Parse the pipe-delimited output into structured data

    Args:
        output_text: Raw text output from LLM

    Returns:
        dict: Parsed extraction with keys: people, parent_child, partnerships, events
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
        if line.startswith('PEOPLE:'):
            current_section = 'people'
            continue
        elif line.startswith('PARENT_CHILD:'):
            current_section = 'parent_child'
            continue
        elif line.startswith('PARTNERSHIPS:'):
            current_section = 'partnerships'
            continue
        elif line.startswith('EVENTS:'):
            current_section = 'events'
            continue
        elif line.startswith('INTERPRETATION:'):
            current_section = None
            continue

        # Skip "None" entries
        if line.lower() == 'none' or line.lower() == '- none':
            continue

        # Parse based on current section
        if current_section == 'people':
            if line.startswith('-'):
                result['people'].append(line[1:].strip())

        elif current_section == 'parent_child':
            if line.startswith('-'):
                parts = line[1:].strip().split('|')
                if len(parts) == 3:
                    result['parent_child'].append({
                        'person1': parts[0].strip(),
                        'relationship_type': parts[1].strip(),
                        'person2': parts[2].strip()
                    })

        elif current_section == 'partnerships':
            if line.startswith('-'):
                parts = line[1:].strip().split('|')
                if len(parts) == 3:
                    # Format: PersonA|spouse|PersonB
                    result['partnerships'].append({
                        'person1': parts[0].strip(),
                        'relationship_type': parts[1].strip(),
                        'person2': parts[2].strip()
                    })

        elif current_section == 'events':
            if line.startswith('-'):
                parts = line[1:].strip().split('|')
                if len(parts) >= 2:
                    result['events'].append({
                        'person': parts[0].strip() if len(parts) > 0 else '',
                        'event_type': parts[1].strip() if len(parts) > 1 else '',
                        'date': parts[2].strip() if len(parts) > 2 else '',
                        'place': parts[3].strip() if len(parts) > 3 else ''
                    })

    return result
