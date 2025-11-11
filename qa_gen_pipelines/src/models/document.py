"""Document data models."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from datetime import datetime


@dataclass
class Document:
    """Represents a document with its content and metadata."""
    
    file_path: Path
    content: str
    file_type: str
    file_size: int
    created_at: datetime
    processed_at: Optional[datetime] = None
    
    @property
    def name(self) -> str:
        """Get document name without extension."""
        return self.file_path.stem
    
    @property
    def extension(self) -> str:
        """Get file extension."""
        return self.file_path.suffix
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    
    document_id: str  # 原文档的ID或路径
    chunk_id: str
    content: str
    start_position: int
    end_position: int
    chunk_index: int
    total_chunks: int
    
    @property
    def length(self) -> int:
        """Get chunk content length."""
        return len(self.content)
    
    def __str__(self) -> str:
        return f"Chunk {self.chunk_index + 1}/{self.total_chunks} of {self.document_id}" 