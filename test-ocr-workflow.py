#!/usr/bin/env python
"""
Test script for OCR workflow in Docker environment
"""

import os
from io import BytesIO

import django
from django.core.files.uploadedfile import SimpleUploadedFile

from PIL import Image, ImageDraw, ImageFont

from genealogy.models import Document, DocumentPage
from genealogy.tasks import process_page_ocr

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "genealogy_extractor.settings")
django.setup()


def create_test_image():
    """Create a simple test image with English and Dutch text"""
    # Create a white image
    img = Image.new("RGB", (500, 300), color="white")
    draw = ImageDraw.Draw(img)

    # Add multilingual genealogy text
    text = """Familie Register / Family Record

    Naam: Johannes van der Berg
    Name: Johannes van der Berg

    Geboren: 15 maart 1892, Amsterdam
    Born: March 15, 1892, Amsterdam

    Vader: Pieter van der Berg
    Father: Pieter van der Berg

    Moeder: Maria de Jong
    Mother: Maria de Jong

    Getrouwd met: Elisabeth Bakker
    Married to: Elisabeth Bakker"""

    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None

    draw.text((20, 20), text, fill="black", font=font)

    # Save to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return img_bytes.getvalue()


def test_ocr_workflow():
    """Test the complete OCR workflow"""
    print("Testing OCR workflow with English + Dutch text...")

    # Create test image
    print("1. Creating test image with multilingual genealogy content...")
    image_data = create_test_image()

    # Create document and page
    print("2. Creating document and page...")
    test_file = SimpleUploadedFile(
        "test_genealogy_doc.png",
        image_data,
        content_type="image/png",
    )

    document = Document.objects.create(
        title="Test Dutch Family Record",
        languages="en+nl",  # Test multilingual OCR
    )

    page = DocumentPage.objects.create(
        document=document,
        page_number=1,
        image_file=test_file,
        original_filename="test_genealogy_doc.png",
    )

    print(f"Created document: {document.id}")
    print(f"Created page: {page.id}")
    print(f"Language setting: {document.languages}")

    # Test OCR processing
    print("3. Testing OCR processing...")
    try:
        result = process_page_ocr(str(page.id))
        print(f"OCR Result: {result}")

        if result.get("success"):
            print("✓ OCR processing successful!")
            extracted_text = result.get("text", "No text")
            print(f"Extracted text preview: {extracted_text[:200]}...")
            print(f"Confidence: {result.get('confidence', 0):.1f}%")
            print(f"Rotation applied: {result.get('rotation_applied', 0)}°")

            # Check for Dutch words
            dutch_words = ["Familie", "Geboren", "Vader", "Moeder", "Getrouwd"]
            found_dutch = [word for word in dutch_words if word in extracted_text]
            print(f"Dutch words detected: {found_dutch}")

            # Check for English words
            english_words = ["Family", "Born", "Father", "Mother", "Married"]
            found_english = [word for word in english_words if word in extracted_text]
            print(f"English words detected: {found_english}")

        else:
            print(f"✗ OCR processing failed: {result.get('error')}")

    except Exception as e:
        print(f"✗ OCR processing error: {e}")
        import traceback

        traceback.print_exc()

    # Check final status
    print("4. Checking final status...")
    page.refresh_from_db()
    document.refresh_from_db()

    print(f"Page OCR completed: {page.ocr_completed}")
    print(f"Document OCR completed: {document.ocr_completed}")
    if page.ocr_text:
        print(f"Full extracted text:\n{page.ocr_text}")

    # Cleanup
    print("5. Cleaning up...")
    document.delete()  # This will cascade delete the page

    print("OCR workflow test completed!")


if __name__ == "__main__":
    test_ocr_workflow()
