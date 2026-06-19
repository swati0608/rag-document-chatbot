"""
document_processor.py
---------------------
Handles parsing and cleaning of uploaded documents.

Supported formats:
    - PDF  (via PyMuPDF / fitz)
    - TXT  (plain UTF-8 text)

Public API:
    DocumentProcessor.process(file_path) -> dict
        Returns:
            {
                "text":       <str>  cleaned full text,
                "page_count": <int>  number of pages (1 for .txt),
                "file_name":  <str>  basename of the file,
                "file_type":  <str>  "pdf" or "txt",
            }
"""

from __future__ import annotations

import os
import re
from typing import Dict

import fitz  # PyMuPDF


class UnsupportedFileTypeError(ValueError):
    """Raised when the uploaded file has an extension we don't handle."""


class EmptyDocumentError(ValueError):
    """Raised when a document parses successfully but contains no readable text."""


class DocumentProcessor:
    """Parse PDF / TXT files into clean plain text for the RAG pipeline."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def process(self, file_path: str) -> Dict[str, object]:
        """
        Parse a document and return its cleaned text plus metadata.

        Args:
            file_path: Absolute or relative path to a .pdf or .txt file.

        Returns:
            A dict with keys: text, page_count, file_name, file_type.

        Raises:
            FileNotFoundError:        if the path does not exist.
            UnsupportedFileTypeError: if the extension is not .pdf or .txt.
            EmptyDocumentError:       if the file contains no readable text.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}'. Supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if ext == ".pdf":
            text, page_count = self._parse_pdf(file_path)
            file_type = "pdf"
        else:
            text, page_count = self._parse_txt(file_path)
            file_type = "txt"

        text = self._clean_text(text)
        if not text.strip():
            raise EmptyDocumentError("The document appears to be empty or unreadable.")

        return {
            "text": text,
            "page_count": page_count,
            "file_name": os.path.basename(file_path),
            "file_type": file_type,
        }

    # ------------------------------------------------------------------ #
    # Format-specific parsers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_pdf(file_path: str) -> tuple[str, int]:
        """Extract text from every page of a PDF using PyMuPDF."""
        parts: list[str] = []
        with fitz.open(file_path) as doc:
            page_count = doc.page_count
            for page in doc:
                parts.append(page.get_text("text"))
        return "\n".join(parts), page_count

    @staticmethod
    def _parse_txt(file_path: str) -> tuple[str, int]:
        """Read a UTF-8 (with fallback) text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # Fallback for files saved in legacy encodings.
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
        return text, 1

    # ------------------------------------------------------------------ #
    # Cleaning
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Light cleanup: collapse excessive whitespace, normalize line breaks,
        strip control characters. Preserves paragraph structure.
        """
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Strip non-printable control chars (keep \n and \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        # Collapse 3+ blank lines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse runs of spaces/tabs
        text = re.sub(r"[ \t]+", " ", text)
        # Trim each line
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text.strip()
