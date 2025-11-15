# Import all admin modules to register them with Django admin
from .book_section import BookSectionAdmin
from .document import DocumentAdmin
from .document_page import DocumentPageAdmin
from .event import EventAdmin
from .partnership_admin import PartnershipAdmin
from .person import PersonAdmin
from .place import PlaceAdmin
from .relationship_admin import RelationshipAdmin
from .textchunk import TextChunkAdmin

__all__ = [
    'BookSectionAdmin',
    'DocumentAdmin',
    'DocumentPageAdmin',
    'EventAdmin',
    'PartnershipAdmin',
    'PersonAdmin',
    'PlaceAdmin',
    'RelationshipAdmin',
    'TextChunkAdmin',
]
