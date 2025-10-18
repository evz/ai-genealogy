# Import all admin modules to register them with Django admin
from .document import DocumentAdmin
from .document_page import DocumentPageAdmin
from .person import PersonAdmin
from .place import PlaceAdmin
from .partnership import PartnershipAdmin
from .event import EventAdmin
from .relationship import ParentChildRelationshipAdmin
from .textchunk import TextChunkAdmin
from .duplicate_clusters import PotentialDuplicateAdmin, MatchReasonFilter

__all__ = [
    'DocumentAdmin',
    'DocumentPageAdmin',
    'PersonAdmin',
    'PlaceAdmin',
    'PartnershipAdmin',
    'EventAdmin',
    'ParentChildRelationshipAdmin',
    'TextChunkAdmin',
    'PotentialDuplicateAdmin',
    'MatchReasonFilter',
]
