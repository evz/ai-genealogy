# Import all admin modules to register them with Django admin
from .book_section import BookSectionAdmin
from .document import DocumentAdmin
from .document_page import DocumentPageAdmin
from .duplicate_clusters import MatchReasonFilter, PotentialDuplicateAdmin
from .event import EventAdmin
from .identity import IdentityAdmin
from .partnership import PartnershipMentionAdmin
from .person_mention import PersonMentionAdmin
from .place import PlaceAdmin
from .relationship import RelationshipMentionAdmin
from .textchunk import TextChunkAdmin

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
