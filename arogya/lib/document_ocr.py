"""Read uploaded claim documents and OCR image-only pages without saving the files."""
from __future__ import annotations

from io import BytesIO
from pathlib import PurePath

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 30
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt"}


class DocumentReadError(ValueError):
    pass


def _ocr_image(image) -> str:
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError as exc:
        raise RuntimeError("OCR dependencies are unavailable; rebuild/install the project requirements.") from exc
    try:
        return pytesseract.image_to_string(image, lang="eng", timeout=30)
    except TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract OCR is not installed. Rebuild the Docker image or install Tesseract locally.") from exc
    except RuntimeError as exc:
        if "timeout" in str(exc).lower():
            raise DocumentReadError("OCR took too long; upload a clearer or smaller scan.") from exc
        raise


def read_document(filename: str, content: bytes) -> tuple[str, str]:
    extension = PurePath(filename.replace("\\", "/")).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentReadError(f"{filename}: use a PDF, PNG, JPG, TIFF, or plain-text file.")
    if not content:
        raise DocumentReadError(f"{filename}: the uploaded file is empty.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentReadError(f"{filename}: files must be 10 MB or smaller.")

    if extension == ".txt":
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise DocumentReadError(f"{filename}: plain-text files must use UTF-8 encoding.") from exc
        if not text:
            raise DocumentReadError(f"{filename}: no text was found.")
        return text, "text"

    if extension == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF reading dependencies are unavailable; install the project requirements.") from exc
        try:
            document = fitz.open(stream=content, filetype="pdf")
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            raise DocumentReadError(f"{filename}: could not read this PDF.") from exc
        with document:
            if document.page_count > MAX_PDF_PAGES:
                raise DocumentReadError(f"{filename}: PDFs are limited to {MAX_PDF_PAGES} pages.")
            if document.page_count == 0:
                raise DocumentReadError(f"{filename}: the PDF has no pages.")
            pages: list[str] = []
            used_ocr = False
            for page in document:
                page_text = page.get_text().strip()
                if len(page_text) < 20:
                    if page.rect.width * page.rect.height * 4 > MAX_IMAGE_PIXELS:
                        raise DocumentReadError(f"{filename}: scanned PDF pages must be 40 megapixels or smaller.")
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    from PIL import Image
                    page_text = _ocr_image(Image.open(BytesIO(pixmap.tobytes("png")))).strip()
                    used_ocr = True
                pages.append(page_text)
        text = "\n\n".join(filter(None, pages)).strip()
        if not text:
            raise DocumentReadError(f"{filename}: no text could be extracted; check scan quality and OCR language.")
        return text, "OCR" if used_ocr else "PDF text"

    try:
        from PIL import Image, UnidentifiedImageError
        image = Image.open(BytesIO(content))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise DocumentReadError(f"{filename}: images must be 40 megapixels or smaller.")
        image.load()
    except DocumentReadError:
        raise
    except (ImportError, UnidentifiedImageError, OSError) as exc:
        raise DocumentReadError(f"{filename}: could not read this image.") from exc
    text = _ocr_image(image).strip()
    if not text:
        raise DocumentReadError(f"{filename}: OCR found no text; check scan quality and OCR language.")
    return text, "OCR"
