"""Descendant genealogy extraction strategy

Two-phase extraction approach:
- Phase 1 (during chunking): Deterministic extraction of people and parent-child relationships
- Phase 2 (LLM): Extract events, partnerships, and additional people
"""

import logging
import os
from typing import Dict, Any

from .base import ExtractionStrategy

logger = logging.getLogger(__name__)

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


class DescendantGenealogyStrategy(ExtractionStrategy):
    """
    Strategy for descendant genealogy sections.

    Extracts genealogical facts from formatted genealogical entries using a
    complementary two-phase approach.
    """

    def __init__(self):
        """Initialize strategy and load examples"""
        self._examples = None

    def _load_examples(self) -> str:
        """Load extraction examples from text file"""
        if self._examples is None:
            examples_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'prompts',
                'examples_extraction.txt'
            )
            with open(examples_path, 'r', encoding='utf-8') as f:
                self._examples = f.read()
        return self._examples

    @property
    def strategy_name(self) -> str:
        return "Descendant Genealogy Extraction"

    def should_process(self, chunk) -> bool:
        """Process GENEALOGY_ENTRY chunks that have genealogical context"""
        return chunk.chunk_type == "GENEALOGY_ENTRY"

    def get_chunk_filter(self) -> Dict[str, Any]:
        """Filter for GENEALOGY_ENTRY chunks only"""
        return {"chunk_type": "GENEALOGY_ENTRY"}

    def build_prompt(self, chunk) -> str:
        """Build extraction prompt with Phase 1 context"""
        examples = self._load_examples()

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
     * Look for occupations in the genealogical entry line (between dates/events)
     * Look for occupations in narrative text and witness descriptions
     * Extract each occupation as a separate OCCU event, even if multiple occupations are listed
   - Residences: places where people lived (use RESI event type)
   - Partnerships: marriages and relationships
   - Additional people mentioned
3. If the content has NO genealogical facts (just narrative/acknowledgments), return "None" for all sections
4. Do NOT copy the examples or Phase 1 data
5. For occupations with dates like "metselaar (1895)", include the date in the event

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

    def parse_output(self, output_text: str) -> Dict[str, Any]:
        """Parse the pipe-delimited output into structured data"""
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

    def extract(self, chunk, ollama, model: str) -> Dict[str, Any]:
        """
        Extract entities from a descendant genealogy chunk using LLM.

        Phase 1 data (people, parent-child relationships) is already in the chunk.
        Phase 2 (this method) extracts events, partnerships, and additional people.
        """
        try:
            logger.info(f"Extracting from chunk {chunk.sequence_number} (Descendant Genealogy)")

            # Build extraction prompt
            prompt = self.build_prompt(chunk)

            # Calculate required context window based on chunk size
            estimated_tokens = len(chunk.text_content) // 4 + 4000
            num_ctx = max(4096, min(131072, 2 ** (estimated_tokens - 1).bit_length()))

            if num_ctx > 8192:
                logger.info(f"Large chunk: using context window {num_ctx} tokens")

            # Query LLM
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    'num_ctx': num_ctx,
                    'temperature': 0.0,
                }
            )

            if not response:
                logger.error(f"No response from LLM for chunk {chunk.sequence_number}")
                return {
                    'success': False,
                    'error': 'No response from LLM'
                }

            # Parse output
            extracted_data = self.parse_output(response)

            # MERGE Phase 2 (LLM) data WITH Phase 1 (deterministic) data
            phase1_people = chunk.extracted_people or []
            phase2_people = extracted_data['people']
            all_people = phase1_people.copy()
            for person in phase2_people:
                if person not in all_people:
                    all_people.append(person)

            # Merge relationships
            phase1_relationships = chunk.extracted_relationships or []
            phase2_parent_child = extracted_data['parent_child']
            phase2_partnerships = extracted_data['partnerships']

            all_relationships = phase1_relationships.copy()
            for rel in phase2_parent_child + phase2_partnerships:
                # Check for duplicates
                is_duplicate = any(
                    existing['person1'] == rel['person1'] and
                    existing['relationship_type'] == rel['relationship_type'] and
                    existing['person2'] == rel['person2']
                    for existing in all_relationships
                )
                if not is_duplicate:
                    all_relationships.append(rel)

            # Events are new (Phase 1 doesn't extract events)
            all_events = extracted_data['events']

            # Save merged data
            chunk.extracted_people = all_people
            chunk.extracted_relationships = all_relationships
            chunk.extracted_events = all_events
            chunk.entities_extracted = True
            chunk.save(update_fields=[
                'extracted_people',
                'extracted_relationships',
                'extracted_events',
                'entities_extracted'
            ])

            result = {
                'success': True,
                'people_count': len(all_people),
                'relationships_count': len(all_relationships),
                'events_count': len(all_events),
                'phase1_people': len(phase1_people),
                'phase2_people_added': len(all_people) - len(phase1_people),
                'phase1_relationships': len(phase1_relationships),
                'phase2_relationships_added': len(all_relationships) - len(phase1_relationships),
            }

            logger.info(
                f"Chunk {chunk.sequence_number}: "
                f"{result['people_count']} people (P1:{result['phase1_people']} + P2:{result['phase2_people_added']}), "
                f"{result['relationships_count']} rels (P1:{result['phase1_relationships']} + P2:{result['phase2_relationships_added']}), "
                f"{result['events_count']} events"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk.sequence_number}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
