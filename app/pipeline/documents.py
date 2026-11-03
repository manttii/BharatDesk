"""Uploaded PDF or photo -> one input per page, routed by what the page really contains.

A page with a real text layer is read as TEXT (exact, cheap). A scanned page has none, so it
is rendered to an image for a vision model (OCR). The route is chosen per page, so a PDF that
mixes digital and scanned pages works. Uploads are untrusted: the file type comes from the
bytes (never the filename), size and page count are capped, and encrypted or malformed files
are refused with a clear reason rather than a crash.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import pypdfium2 as pdfium
import structlog
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.pipeline.media import sniff_image
from app.providers.llm import ImagePart

log = structlog.get_logger()

MIN_TEXT_CHARS = 60  # fewer characters than this means "scanned": use the image
MAX_TEXT_CHARS = 20_000
MAX_RENDER_PX = 2200  # longest side of a rendered page


class DocumentError(Exception):
    """The upload cannot be processed; the message is safe to show the user."""


@dataclass(frozen=True)
class PageInput:
    page_no: int  # 1-based
    route: Literal["text", "image"]
    text: str | None = None
    image: ImagePart | None = None


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def read_document(data: bytes, *, max_pages: int, max_bytes: int) -> list[PageInput]:
    """Blocking (CPU) work: call it in a worker thread."""
    if len(data) > max_bytes:
        raise DocumentError("file is too large")
    if is_pdf(data):
        return _read_pdf(data, max_pages)
    sniffed = sniff_image(data)
    if sniffed:
        return [PageInput(1, "image", image=ImagePart(data, sniffed[1]))]
    raise DocumentError("only PDF files and photos (JPG/PNG) are supported")


def _read_pdf(data: bytes, max_pages: int) -> list[PageInput]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentError("this PDF is password-protected; upload an unlocked copy")
        count = len(reader.pages)
        if count == 0:
            raise DocumentError("this PDF has no pages")
        if count > max_pages:
            raise DocumentError(f"this PDF has {count} pages; the limit is {max_pages}")
        texts = []
        for page in reader.pages:
            try:
                texts.append((page.extract_text() or "").strip())
            except Exception:  # a broken text layer is treated as 'scanned'
                texts.append("")
    except DocumentError:
        raise
    except (PyPdfError, ValueError, OSError, RecursionError) as exc:
        raise DocumentError("this PDF is damaged or unreadable") from exc

    pages: list[PageInput] = []
    renderer: pdfium.PdfDocument | None = None
    try:
        for i, text in enumerate(texts):
            if len(text) >= MIN_TEXT_CHARS:
                pages.append(PageInput(i + 1, "text", text=text[:MAX_TEXT_CHARS]))
                continue
            if renderer is None:
                renderer = pdfium.PdfDocument(data)
            pages.append(PageInput(i + 1, "image", image=_render(renderer[i])))
    except pdfium.PdfiumError as exc:
        raise DocumentError("a page of this PDF could not be rendered") from exc
    finally:
        if renderer is not None:
            renderer.close()
    return pages


def _render(page: pdfium.PdfPage) -> ImagePart:
    width, height = page.get_size()
    scale = min(2.0, MAX_RENDER_PX / max(width, height, 1.0))
    image = page.render(scale=scale).to_pil().convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return ImagePart(buf.getvalue(), "image/jpeg")
