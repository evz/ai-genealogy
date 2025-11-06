"""Test helper functions and fixtures for genealogy tests"""

from typing import Optional, List, Dict, Any
from genealogy.chunking.models import GroundingToken, BoundingBox


def create_grounding_token(
    content: str,
    element_type: str = 'text',
    bbox: Optional[tuple] = None,
    is_inverted: bool = False
) -> GroundingToken:
    """
    Factory for creating test GroundingToken objects.

    Args:
        content: Text content of the token
        element_type: Element type (text, sub_title, image, etc.)
        bbox: Bounding box coordinates as (x1, y1, x2, y2). Defaults to (0, 0, 100, 20)
        is_inverted: Whether token is inverted (white text on black background)

    Returns:
        GroundingToken instance
    """
    if bbox is None:
        bbox = (0, 0, 100, 20)

    bbox_obj = BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])

    # Generate raw_match format that matches DeepSeek-OCR output
    inverted_tag = '<|inverted|>true<|/inverted|>' if is_inverted else ''
    raw_match = (
        f'<|ref|>{element_type}<|/ref|>'
        f'<|det|>[[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]]<|/det|>'
        f'{inverted_tag}\n{content}'
    )

    return GroundingToken(
        element_type=element_type,
        bbox=bbox_obj,
        content=content,
        raw_match=raw_match,
        is_inverted=is_inverted
    )


def create_token_sequence(
    tokens: List[Dict[str, Any]]
) -> List[GroundingToken]:
    """
    Create a sequence of grounding tokens from simplified specs.

    Args:
        tokens: List of dicts with keys: content, element_type, bbox (optional), is_inverted (optional)

    Returns:
        List of GroundingToken instances

    Example:
        tokens = create_token_sequence([
            {'content': 'Tweede generatie', 'element_type': 'sub_title'},
            {'content': 'II.1. Kinderen van...', 'element_type': 'sub_title'},
            {'content': 'a. Pieter van Zanten', 'element_type': 'text'},
        ])
    """
    result = []
    y_offset = 0

    for spec in tokens:
        bbox = spec.get('bbox')
        if bbox is None:
            # Auto-generate bounding boxes stacked vertically
            bbox = (10, y_offset, 500, y_offset + 20)
            y_offset += 25

        token = create_grounding_token(
            content=spec['content'],
            element_type=spec.get('element_type', 'text'),
            bbox=bbox,
            is_inverted=spec.get('is_inverted', False)
        )
        result.append(token)

    return result


def create_mock_ollama_response(
    people: List[str],
    relationships: Optional[List[tuple]] = None,
    partnerships: Optional[List[tuple]] = None,
    events: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate mock LLM output in the expected format.

    Args:
        people: List of person names
        relationships: List of (parent, child, relationship_type) tuples
        partnerships: List of (person1, person2) tuples
        events: List of event dicts with keys: person, event_type, date, place

    Returns:
        Formatted string matching LLM extraction output

    Example:
        output = create_mock_ollama_response(
            people=['Pieter van Zanten', 'Maria Jansen'],
            relationships=[('Pieter van Zanten', 'Jan van Zanten', 'father')],
            events=[{
                'person': 'Pieter van Zanten',
                'event_type': 'BIRT',
                'date': '1850-01-15',
                'place': 'Amsterdam'
            }]
        )
    """
    lines = []

    # PEOPLE section
    lines.append('PEOPLE:')
    for person in people:
        lines.append(f'{person}')
    lines.append('')

    # PARENT_CHILD section
    lines.append('PARENT_CHILD:')
    if relationships:
        for parent, child, rel_type in relationships:
            lines.append(f'{parent}|{child}|{rel_type}')
    lines.append('')

    # PARTNERSHIPS section
    lines.append('PARTNERSHIPS:')
    if partnerships:
        for person1, person2 in partnerships:
            lines.append(f'{person1}|{person2}')
    lines.append('')

    # EVENTS section
    lines.append('EVENTS:')
    if events:
        for event in events:
            person = event.get('person', '')
            event_type = event.get('event_type', '')
            date = event.get('date', '')
            place = event.get('place', '')
            lines.append(f'{person}|{event_type}|{date}|{place}')
    lines.append('')

    return '\n'.join(lines)


def create_ocr_text(tokens: List[Dict[str, Any]]) -> str:
    """
    Create OCR text in DeepSeek format from token specs.

    Args:
        tokens: List of dicts with keys: content, element_type, bbox (optional), is_inverted (optional)

    Returns:
        OCR text string with grounding tokens in DeepSeek format

    Example:
        ocr_text = create_ocr_text([
            {'content': 'Tweede generatie', 'element_type': 'sub_title'},
            {'content': 'a. Pieter van Zanten', 'element_type': 'text'},
        ])
    """
    token_objs = create_token_sequence(tokens)
    return '\n\n'.join(token.raw_match for token in token_objs)


def load_fixture(fixture_name: str) -> str:
    """
    Load a test fixture file from genealogy/tests/fixtures/

    Args:
        fixture_name: Relative path within fixtures/ directory
                     e.g., 'ocr_samples/page_77_generation_12.txt'

    Returns:
        File contents as string
    """
    import os

    fixtures_dir = os.path.join(
        os.path.dirname(__file__),
        'fixtures'
    )
    fixture_path = os.path.join(fixtures_dir, fixture_name)

    with open(fixture_path, 'r', encoding='utf-8') as f:
        return f.read()
