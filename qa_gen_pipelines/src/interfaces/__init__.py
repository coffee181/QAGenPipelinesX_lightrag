"""Abstract interfaces for QA Generation Pipelines."""

from .ocr_interface import OCRInterface
from .text_chunker_interface import TextChunkerInterface
from .question_generator_interface import QuestionGeneratorInterface
from .rag_interface import RAGInterface
from .markdown_processor_interface import MarkdownProcessorInterface

__all__ = [
    "OCRInterface",
    "TextChunkerInterface", 
    "QuestionGeneratorInterface",
    "RAGInterface",
    "MarkdownProcessorInterface"
] 