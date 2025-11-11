"""Question data models."""

from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class Question:
    """Represents a single question with context information."""
    
    question_id: str
    content: str
    source_document: str
    source_chunk_id: str
    question_index: int  # 在该文档中的问题索引
    created_at: datetime
    metadata: Dict[str, Any] = None
    
    # Enhanced context information (NEW)
    source_chunk_content: str = None  # 源chunk内容
    expected_context_keywords: List[str] = None  # 预期上下文关键词
    section_title: str = None  # 问题所属章节
    related_entities: List[str] = None  # 相关实体 (型号、术语等)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.metadata is None:
            self.metadata = {}
        if self.expected_context_keywords is None:
            self.expected_context_keywords = []
        if self.related_entities is None:
            self.related_entities = []
    
    @property
    def formatted_content(self) -> str:
        """Get formatted question content."""
        if not self.content.startswith("问题"):
            return f"问题{self.question_index}：{self.content}"
        return self.content
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get context information for RAG queries."""
        return {
            "section": self.section_title,
            "keywords": self.expected_context_keywords,
            "entities": self.related_entities,
            "source_chunk": self.source_chunk_id
        }


@dataclass 
class QuestionSet:
    """Represents a set of questions for a document."""
    
    document_id: str
    questions: List[Question]
    created_at: datetime
    total_questions: int = 0
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.total_questions:
            self.total_questions = len(self.questions)
    
    def add_question(self, question: Question) -> None:
        """Add a question to the set."""
        self.questions.append(question)
        self.total_questions = len(self.questions)
    
    def get_questions_as_list(self) -> List[str]:
        """Get all questions as a list of strings."""
        return [q.content for q in self.questions]
    
    def to_jsonl_format(self) -> Dict[str, Any]:
        """Convert to JSONL format for storage (with full metadata)."""
        return {
            "messages": [
                {
                    "question_id": q.question_id,
                    "content": q.content,
                    "source_document": q.source_document,
                    "source_chunk_id": q.source_chunk_id,
                    "question_index": q.question_index,
                    "metadata": q.metadata,  # 保存metadata（包含预生成的答案）
                    "created_at": q.created_at.isoformat() if q.created_at else None
                }
                for q in self.questions
            ]
        } 