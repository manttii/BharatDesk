"""Closed vocabularies stored as strings in the database."""

from __future__ import annotations

import enum


class Role(enum.StrEnum):
    BOSS = "boss"
    MANAGER = "manager"
    WORKER = "worker"


class ProcessingState(enum.StrEnum):
    RECEIVED = "received"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"


class ActivityStatus(enum.StrEnum):
    CONFIRMED = "confirmed"
    PENDING_CONFIRMATION = "pending_confirmation"
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    SYNCED = "synced"


class ApprovalStatus(enum.StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SyncStatus(enum.StrEnum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"  # gave up after max attempts; a boss is told once
    CANCELLED = "cancelled"  # the record was no longer eligible (never sync unconfirmed data)
