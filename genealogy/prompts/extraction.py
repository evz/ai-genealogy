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

    family_group = chunk.family_groups[0] if chunk.family_groups else "None"

    return f"""You are extracting genealogical information from Dutch genealogy text.

{DUTCH_ABBREVIATIONS}

{EVENT_TYPE_CODES}

Here are examples of how to extract information:

{examples}

======================================================================
NOW EXTRACT FROM THIS TEXT
======================================================================

CONTENT TO EXTRACT FROM:
{chunk.text_content}

FAMILY GROUP CONTEXT (for inferring parent relationships only): {family_group}

Extract the information in the same format as the examples above:

PEOPLE:
- List all people mentioned (one per line)

PARENT_CHILD:
- PersonA|child|PersonB (PersonA is child of PersonB)
- PersonA|parent|PersonB (PersonA is parent of PersonB)

PARTNERSHIPS:
- PersonA|spouse|PersonB

EVENTS:
- PersonName|EVENT_CODE|Date|Place
- Use event codes: BIRT, DEAT, MARR, BAPT, BURI, RESI, OCCU, EDUC, IMMI, EMIG, OTHER
- Date format: YYYY-MM-DD (or YYYY if only year known, or date ranges like "1847-1852")
- Place can be empty if not mentioned

CRITICAL RULES - READ CAREFULLY:
- ONLY extract information that is EXPLICITLY STATED in the "CONTENT TO EXTRACT FROM" section above
- DO NOT invent, assume, or hallucinate any information not present in the content
- DO NOT extract information from the FAMILY GROUP CONTEXT - use it ONLY to infer parent names when you see "dv" or "zv" without explicit parent names
- If the content is very short or incomplete, extract only what is explicitly there - DO NOT fill in missing information
- Do NOT extract from speculative text (words like "Misschien", "mogelijk", "vermoedelijk")
- Do NOT extract witnesses (Doopgetuige, getuige) as family members
- When you see "dv ParentA en ParentB", create TWO separate child relationships
- When you see "Kinderen van ParentA en ParentB", create child relationships to BOTH parents
- If a section has no data, write "None"
- If the content only mentions a name and birth symbol, only extract that person and birth event - nothing more

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
