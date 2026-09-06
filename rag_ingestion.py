"""Document extraction without model downloads or network calls at import time."""
from __future__ import annotations
import io
import logging
import os
import re
import unicodedata
from typing import List, Tuple
from langchain_core.messages import HumanMessage
logger = logging.getLogger(__name__)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv", ".txt", ".md"}
MAX_FILE_BYTES = 25 * 1024 * 1024

def _clean_text(text: str) -> str:
    """
    CLEANING stage:
    - Normalise unicode (NFC)
    - Remove null bytes and non-printable control characters
    - Collapse excessive whitespace / blank lines
    - Strip page-header/footer artifacts (lines that are purely numbers or short repeated tokens)
    """
    text = unicodedata.normalize("NFC", text)
    text = ''.join(c for c in text if c in '\n\r\t' or not unicodedata.category(c).startswith('C'))
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Numeric lines may be grades, amounts, dates, or table values. Keep them.
    return text.strip()

def _format_pdf_pages(page_texts: List[Tuple[int, str]]) -> str:
    parts = []
    for page_num, text in page_texts:
        cleaned = _clean_text(text)
        if cleaned:
            parts.append(f"## Page {page_num}\n\n{cleaned}")
    return "\n\n---\n\n".join(parts)

def _pdf_text_with_pypdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    if getattr(reader, "is_encrypted", False):
        try:
            if not reader.decrypt(""):
                raise ValueError("This PDF is password-protected and cannot be read.")
        except Exception as exc:
            raise ValueError("This PDF is password-protected and cannot be read.") from exc

    page_texts = []
    for i, page in enumerate(reader.pages, 1):
        text = ""
        for mode in ("plain", "layout"):
            try:
                text = page.extract_text(extraction_mode=mode) or ""
            except Exception:
                text = ""
            if text.strip():
                break
        if text.strip():
            page_texts.append((i, text))
    return _format_pdf_pages(page_texts)

def _pdf_text_with_pymupdf(file_bytes: bytes) -> str:
    try:
        import fitz
    except ImportError:
        return ""

    page_texts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, 1):
            text = page.get_text("text") or ""
            if not text.strip():
                blocks = page.get_text("blocks") or []
                text = "\n".join(str(block[4]) for block in blocks if len(block) > 4)
            if text.strip():
                page_texts.append((i, text))
    return _format_pdf_pages(page_texts)

def _pdf_text_with_groq_vision(file_bytes: bytes, user_id: str = "", page_numbers=None) -> str:
    try:
        import base64
        import fitz
    except ImportError:
        return ""

    from rag_providers import get_vision_llm, MissingAPIKeyError
    try:
        vision_llm = get_vision_llm(user_id)
    except MissingAPIKeyError as exc:
        raise ValueError("This PDF contains scanned pages. Add a Groq key in Settings, or upload a text-based version.") from exc

    max_pages = int(os.getenv("PDF_VISION_OCR_MAX_PAGES", "12"))
    dpi = int(os.getenv("PDF_VISION_OCR_DPI", "160"))
    scale = dpi / 72
    page_texts = []

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        selected = set(page_numbers) if page_numbers is not None else set(range(1, len(doc) + 1))
        if len(selected) > max_pages:
            raise ValueError(
                f"This PDF has {len(selected)} scanned pages. "
                f"Automatic OCR is limited to {max_pages} pages; set "
                "PDF_VISION_OCR_MAX_PAGES higher or upload a text-based PDF."
            )

        for i, page in enumerate(doc, 1):
            if i not in selected:
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode()
            msg = HumanMessage(content=[
                {"type": "text", "text": (
                    "Transcribe every readable word from this PDF page. Preserve tables "
                    "as markdown tables when possible. Do not summarize. If there is no "
                    "readable text, reply with an empty string."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ])
            response = vision_llm.invoke([msg])
            from rag_providers import content_text
            text = content_text(response.content).strip()
            if text:
                page_texts.append((i, text))
            else:
                raise ValueError(f"OCR could not read page {i}. Upload a clearer or text-based version.")

    return _format_pdf_pages(page_texts)

def _pdf_to_markdown(file_bytes: bytes, filename: str = "", user_id: str = "") -> str:
    # Inspect every page: mixed text/scanned PDFs must not silently lose pages.
    try:
        import fitz
        with fitz.open(stream=file_bytes, filetype='pdf') as doc:
            if doc.needs_pass and not doc.authenticate(''):
                raise ValueError('This PDF is password-protected and cannot be read.')
            if len(doc) > 200:
                raise ValueError('PDFs are limited to 200 pages. Split this document into smaller files.')
            pages, scanned = [], []
            for i, page in enumerate(doc, 1):
                text = page.get_text('text', sort=True) or ''
                if text.strip():
                    pages.append((i, text))
                elif page.get_images() or page.get_drawings():
                    scanned.append(i)
        if scanned:
            ocr = _pdf_text_with_groq_vision(file_bytes, user_id, scanned)
            parts = re.split(r'^## Page (\d+)\s*\n', ocr, flags=re.MULTILINE)
            pages.extend((int(parts[i]), parts[i + 1]) for i in range(1, len(parts), 2))
        result = _format_pdf_pages(sorted(pages))
        if result.strip():
            return result
    except ImportError:
        text = _pdf_text_with_pypdf(file_bytes)
        if text.strip():
            return text
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Could not read this PDF. Check that it is valid and that OCR is configured for scanned pages.') from exc

    raise ValueError(
        "This PDF has no extractable text. It may be a scanned/image-only PDF. "
        "Install the updated requirements and save a Groq API key to enable "
        "vision OCR, or upload a text-based PDF, Word, CSV, or TXT version."
    )

def _docx_to_markdown(file_bytes: bytes) -> str:
    from docx import Document as DocxDoc
    from docx.text.paragraph import Paragraph
    doc = DocxDoc(io.BytesIO(file_bytes))
    lines = []
    for item in doc.iter_inner_content():
        if isinstance(item, Paragraph):
            style = item.style.name.lower() if item.style else ''
            heading = re.match(r'heading ([1-6])', style)
            lines.append(('#' * int(heading[1]) + ' ' if heading else '') + item.text.strip())
        elif item.rows:
            rows = [[c.text.strip().replace('|', '\\|').replace('\n', '<br>') for c in r.cells] for r in item.rows]
            lines.append('| ' + ' | '.join(rows[0]) + ' |')
            lines.append('| ' + ' | '.join(['---'] * len(rows[0])) + ' |')
            lines.extend('| ' + ' | '.join(r) + ' |' for r in rows[1:])
    return _clean_text("\n".join(lines))

def _pptx_to_markdown(file_bytes: bytes) -> str:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(file_bytes))
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        lines = [f"## Slide {i}"]
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip(): lines.append(shape.text.strip())
            if shape.has_table:
                lines.extend(' | '.join(c.text for c in row.cells) for row in shape.table.rows)
        if len(lines) > 1:
            parts.append("\n\n".join(lines))
    return _clean_text("\n\n---\n\n".join(parts))

def _xlsx_to_markdown(file_bytes: bytes) -> str:
    import pandas as pd
    with pd.ExcelFile(io.BytesIO(file_bytes)) as xl:
        parts = []
        for sheet in xl.sheet_names:
            frame = xl.parse(sheet, dtype=str, keep_default_na=False)
            if not frame.empty:
                parts.append(f"## Sheet: {sheet}\n\n{frame.to_markdown(index=False, disable_numparse=True)}")
        return '\n\n---\n\n'.join(parts)

def _csv_to_markdown(file_bytes: bytes) -> str:
    import pandas as pd
    return pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False).to_markdown(index=False, disable_numparse=True)

def _txt_to_markdown(file_bytes: bytes) -> str:
    encoding = 'utf-16' if file_bytes.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    try:
        text = file_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError('Save this text file as UTF-8 or UTF-16 and upload it again.') from exc
    if '\x00' in text:
        raise ValueError('This file contains binary data, not readable text.')
    return _clean_text(text)

def convert_file_to_markdown(file_bytes: bytes, filename: str, user_id: str = "") -> Tuple[str, str]:
    ext = os.path.splitext(filename.lower())[1]
    if not file_bytes:
        raise ValueError('The file is empty.')
    if len(file_bytes) > MAX_FILE_BYTES:
        raise ValueError('Each document must be 25 MB or smaller.')
    # CHANGED: reject images explicitly instead of mis-decoding them as text.
    if ext in IMAGE_EXTS:
        raise ValueError(
            "Images cannot be ingested as documents. Upload PDF, Word, PowerPoint, "
            "Excel, CSV, text, or Markdown files."
        )
    if ext not in SUPPORTED_EXTS:
        raise ValueError('Unsupported file type. Upload PDF, DOCX, PPTX, XLSX, XLS, CSV, TXT, or Markdown.')
    if ext in {'.docx', '.pptx', '.xlsx'}:
        import zipfile
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 100 * 1024 * 1024:
                    raise ValueError('The expanded document is too large. Split it into smaller files.')
        except zipfile.BadZipFile as exc:
            raise ValueError('This document is damaged or its extension does not match its contents.') from exc
    if ext == ".pdf": return _pdf_to_markdown(file_bytes, filename, user_id), "pdf"
    elif ext == ".docx": return _docx_to_markdown(file_bytes), "docx"
    elif ext == ".pptx": return _pptx_to_markdown(file_bytes), "pptx"
    elif ext in (".xlsx", ".xls"): return _xlsx_to_markdown(file_bytes), "xlsx"
    elif ext == ".csv": return _csv_to_markdown(file_bytes), "csv"
    else: return _txt_to_markdown(file_bytes), ext.lstrip('.')

# ── 3. CHUNKING ───────────────────────────────────────────────────────────────
