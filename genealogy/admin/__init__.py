# Import all admin modules to register them with Django admin
from .document import DocumentAdmin
from .document_page import DocumentPageAdmin
from .book_section import BookSectionAdmin
from .person_mention import PersonMentionAdmin
from .identity import IdentityAdmin
from .place import PlaceAdmin
from .partnership import PartnershipMentionAdmin
from .event import EventAdmin
from .relationship import RelationshipMentionAdmin
from .textchunk import TextChunkAdmin
from .duplicate_clusters import PotentialDuplicateAdmin, MatchReasonFilter

__all__ = [
    'DocumentAdmin',
    'DocumentPageAdmin',
    'BookSectionAdmin',
    'PersonMentionAdmin',
    'IdentityAdmin',
    'PlaceAdmin',
    'PartnershipMentionAdmin',
    'EventAdmin',
    'RelationshipMentionAdmin',
    'TextChunkAdmin',
    'PotentialDuplicateAdmin',
    'MatchReasonFilter',
]
