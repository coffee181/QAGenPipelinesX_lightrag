"""QA pair data models."""

from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class QAPair:
    """Represents a question-answer pair."""
    
    question_id: str
    question: str
    answer: str
    source_document: str
    confidence_score: float = 0.0
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_message_format(self) -> List[Dict[str, str]]:
        """Convert to message format for JSONL."""
        return [
            {"role": "user", "content": self.question},
            {"role": "assistant", "content": self.answer}
        ]


@dataclass
class QASet:
    """Represents a set of QA pairs for a document."""
    
    document_id: str
    qa_pairs: List[QAPair]
    created_at: datetime
    total_pairs: int = 0
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.total_pairs:
            self.total_pairs = len(self.qa_pairs)
    
    def add_qa_pair(self, qa_pair: QAPair) -> None:
        """Add a QA pair to the set."""
        self.qa_pairs.append(qa_pair)
        self.total_pairs = len(self.qa_pairs)
    
    def to_jsonl_format(self) -> Dict[str, List[Dict[str, str]]]:
        """Convert to JSONL format for storage."""
        messages = []
        for qa_pair in self.qa_pairs:
            messages.extend(qa_pair.to_message_format())
        
        return {
            "messages": messages
        }
    
    def get_questions(self) -> List[str]:
        """Get all questions as a list."""
        return [qa.question for qa in self.qa_pairs]
    
    def get_answers(self) -> List[str]:
        """Get all answers as a list."""
        return [qa.answer for qa in self.qa_pairs] 