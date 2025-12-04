"""Concrete implementations for QA Generation Pipelines."""

# Core implementations that are always available
from .simple_text_chunker import SimpleTextChunker
from .simple_markdown_processor import SimpleMarkdownProcessor

# Optional implementations that may not be available in all environments
__all__ = [
    "SimpleTextChunker",
    "SimpleMarkdownProcessor"
]

# Try to import TesseractOCR, but don't fail if dependencies are missing
try:
    from .tesseract_ocr import TesseractOCR
    __all__.append("TesseractOCR")
except ImportError:
    # pytesseract or other dependencies not available
    TesseractOCR = None 