# Import all admin modules to register them with Django admin
from .archive import ArchiveAdmin
from .book_section import BookSectionAdmin
from .conversation import ConversationAdmin, MessageAdmin
from .document import DocumentAdmin
from .document_page import DocumentPageAdmin
from .event import EventAdmin
from .partnership_admin import PartnershipAdmin
from .person import PersonAdmin
from .place import PlaceAdmin
from .prompt_log import PromptLogAdmin
from .prompt_template import PromptTemplateAdmin
from .relationship_admin import RelationshipAdmin
from .textchunk import TextChunkAdmin

__all__ = [
    'ArchiveAdmin',
    'BookSectionAdmin',
    'ConversationAdmin',
    'DocumentAdmin',
    'DocumentPageAdmin',
    'EventAdmin',
    'MessageAdmin',
    'PartnershipAdmin',
    'PersonAdmin',
    'PlaceAdmin',
    'PromptLogAdmin',
    'PromptTemplateAdmin',
    'RelationshipAdmin',
    'TextChunkAdmin',
]
