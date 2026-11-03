"""Import every model so `Base.metadata` is complete for Alembic and create_all."""

from app.models.activity import ActivityLog, Correction, Extraction, Message
from app.models.audit import AuditLog
from app.models.base import Base
from app.models.bills import Bill, BillLine
from app.models.core import Client, InviteToken, Tenant, User
from app.models.policy import ApprovalRequest, BenchmarkRate, PolicyRule
from app.models.state import ConversationState, ProcessedEvent
from app.models.sync import SyncJob

__all__ = [
    "ActivityLog",
    "ApprovalRequest",
    "AuditLog",
    "Base",
    "BenchmarkRate",
    "Bill",
    "BillLine",
    "Client",
    "ConversationState",
    "Correction",
    "Extraction",
    "InviteToken",
    "Message",
    "PolicyRule",
    "ProcessedEvent",
    "SyncJob",
    "Tenant",
    "User",
]
