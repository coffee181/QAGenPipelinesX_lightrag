"""Data models for QA Generation Pipelines."""

from .document import Document, DocumentChunk
from .question import Question, QuestionSet
from .qa_pair import QAPair, QASet

__all__ = [
    "Document",
    "DocumentChunk", 
    "Question",
    "QuestionSet",
    "QAPair",
    "QASet"
] 