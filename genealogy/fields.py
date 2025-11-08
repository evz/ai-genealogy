"""
Custom field types and utilities for the genealogy app
"""

from django import forms
from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ValidationError


class CommaSeparatedArrayField(ArrayField):
    """
    ArrayField that accepts comma-separated strings in admin and converts them to arrays
    """

    def to_python(self, value):
        """Convert from database or form input to Python"""
        # If it's already a list, return as-is (from database)
        if isinstance(value, list):
            return value

        # If it's None or empty string, return empty list
        if not value:
            return []

        # If it's a string, split by comma and clean up
        if isinstance(value, str):
            # Split by comma and strip whitespace
            return [item.strip() for item in value.split(",") if item.strip()]

        # Fallback to parent implementation
        return super().to_python(value)

    def get_prep_value(self, value):
        """Convert from Python to database storage"""
        # If it's a string (from form input), convert to list first
        if isinstance(value, str):
            value = self.to_python(value)

        # Use parent implementation for array storage
        return super().get_prep_value(value)

    def validate(self, value, model_instance):
        """Validate the field value"""
        # Convert string to list for validation if needed
        if isinstance(value, str):
            value = self.to_python(value)

        # Use parent validation
        super().validate(value, model_instance)

    def formfield(self, **kwargs):
        """Return the form field for admin interface"""
        # Create a custom form field that handles the conversion
        class CommaSeparatedFormField(forms.CharField):
            def prepare_value(self, value):
                """Convert list to comma-separated string for display"""
                if isinstance(value, list):
                    return ", ".join(str(item) for item in value if item)
                return value or ""

            def to_python(self, value):
                """Convert comma-separated string to list"""
                if not value:
                    return []
                if isinstance(value, list):
                    return value
                # Split by comma and clean up whitespace
                return [item.strip() for item in value.split(",") if item.strip()]

        # Use our custom form field
        kwargs.setdefault("form_class", CommaSeparatedFormField)
        kwargs.setdefault(
            "help_text",
            f'{kwargs.get("help_text", "")} (Enter comma-separated values)'.strip(),
        )

        # Remove ArrayField-specific arguments that CharField doesn't understand
        kwargs.pop("base_field", None)
        kwargs.pop("size", None)

        return super(ArrayField, self).formfield(**kwargs)


def clean_comma_separated_string(value, field_name="field"):
    """
    Utility function to clean and validate comma-separated strings

    Args:
        value: The input value (string or list)
        field_name: Name of field for error messages

    Returns:
        list: Cleaned list of strings

    Raises:
        ValidationError: If the input is invalid
    """
    if not value:
        return []

    if isinstance(value, list):
        # Already a list, just clean the strings
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        # Split by comma and clean
        return [item.strip() for item in value.split(",") if item.strip()]

    raise ValidationError(f"{field_name} must be a comma-separated string or list")
