"""Voice-note intake: validate what the sender claims, hash and store what we received.

Telegram's `duration`, `mime_type` and `file_size` are sender-supplied, so they gate cheap
early rejections but the real checks run on the downloaded bytes (size cap, magic bytes).
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path

from app.config import Settings
from app.schemas.events import IncomingEvent


class MediaError(Exception):
    """`code` is a pack string key suffix: too_short | too_long | bad_type | too_large."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def validate_voice_event(event: IncomingEvent, settings: Settings) -> None:
    if event.file_id is None:
        raise MediaError("bad_type")
    if event.duration_s is not None:
        if event.duration_s * 1000 < settings.min_voice_ms:
            raise MediaError("too_short")
        if event.duration_s > settings.max_voice_seconds:
            raise MediaError("too_long")
    if event.file_size is not None and event.file_size > settings.max_media_bytes:
        raise MediaError("too_large")
    if event.mime_type is not None and event.mime_type not in settings.allowed_voice_mimes:
        raise MediaError("bad_type")


def sniff_audio_extension(data: bytes) -> str | None:
    """Identify the container from magic bytes; None if it is not a recognised audio type."""
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if data[4:8] == b"ftyp":
        return "m4a"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    if data[:4] == b"fLaC":
        return "flac"
    return None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_SAFE = re.compile(r"[^A-Za-z0-9]")  # extensions are plain alphanumerics: no dots, no slashes


class MediaStore:
    """Local content-addressed storage: <base>/<tenant_id>/<sha256>.<ext>."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base = Path(base_dir)

    def path_for(self, tenant_id: uuid.UUID, sha256: str, ext: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("bad sha256")
        return self.base / str(tenant_id) / f"{sha256}.{_SAFE.sub('', ext) or 'bin'}"

    def delete(self, tenant_id: uuid.UUID, sha256: str) -> int:
        """Remove every stored file for this content hash (any extension). Returns the count.
        Missing files are fine, so a purge that was interrupted can simply be re-run."""
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("bad sha256")
        folder = self.base / str(tenant_id)
        removed = 0
        for path in folder.glob(f"{sha256}.*") if folder.is_dir() else ():
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def save(self, tenant_id: uuid.UUID, sha256: str, ext: str, data: bytes) -> Path:
        path = self.path_for(tenant_id, sha256, ext)
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # atomic: a reader never sees a half-written file
        return path


def validate_photo_event(event: IncomingEvent, settings: Settings) -> None:
    """Cheap early checks on what Telegram reports; the real checks run on the bytes."""
    if event.file_id is None:
        raise MediaError("bad_type")
    if event.file_size is not None and event.file_size > settings.max_photo_bytes:
        raise MediaError("too_large")
    sides = [d for d in (event.width, event.height) if d is not None]
    if sides and min(sides) < settings.min_photo_dimension:
        raise MediaError("too_small")


def sniff_image(data: bytes) -> tuple[str, str] | None:
    """(extension, mime) from magic bytes, or None if it is not a supported image."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpg", "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png", "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    return None
