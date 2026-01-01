"""Concrete implementations for QA Generation Pipelines."""

# Core implementations that are always available
from .simple_text_chunker import SimpleTextChunker
from .simple_markdown_processor import SimpleMarkdownProcessor

# Optional implementations
__all__ = [
    "SimpleTextChunker",
    "SimpleMarkdownProcessor",
]

try:
    from .paddle_ocr import PaddleOCREngine
    __all__.append("PaddleOCREngine")
except ImportError:
    PaddleOCREngine = None