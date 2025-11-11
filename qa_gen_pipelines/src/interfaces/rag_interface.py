"""RAG interface definition."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..models.document import Document
from ..models.question import Question
from ..models.qa_pair import QAPair, QASet


class RAGInterface(ABC):
    """Abstract interface for RAG implementations."""
    
    @abstractmethod
    def insert_document(self, document: Document) -> None:
        """
        Insert a single document into the knowledge base.
        
        Args:
            document: Document to insert
            
        Raises:
            RAGError: If insertion fails
        """
        pass
    
    @abstractmethod
    def insert_documents_batch(self, documents: List[Document]) -> None:
        """
        Insert multiple documents into the knowledge base.
        
        Args:
            documents: List of documents to insert
            
        Raises:
            RAGError: If batch insertion fails
        """
        pass
    
    @abstractmethod
    def insert_from_directory(self, directory_path: Path) -> None:
        """
        Insert all text files from a directory.
        
        Args:
            directory_path: Path to directory containing text files
            
        Raises:
            RAGError: If directory insertion fails
        """
        pass
    
    @abstractmethod
    def query_single_question(self, question: str, source_document: Optional[str] = None) -> str:
        """
        Query the knowledge base with a single question.
        
        Args:
            question: Question to query
            source_document: Optional document ID to filter results (for document isolation)
            
        Returns:
            Generated answer
            
        Raises:
            RAGError: If query fails
        """
        pass
    
    @abstractmethod
    def query_questions_batch(self, questions: List[str]) -> List[str]:
        """
        Query the knowledge base with multiple questions.
        
        Args:
            questions: List of questions to query
            
        Returns:
            List of generated answers
            
        Raises:
            RAGError: If batch query fails
        """
        pass
    
    @abstractmethod
    def generate_qa_pairs_from_questions(self, questions: List[Question]) -> QASet:
        """
        Generate QA pairs from questions using the knowledge base.
        
        Args:
            questions: List of Question objects
            
        Returns:
            QASet containing generated QA pairs
            
        Raises:
            RAGError: If QA generation fails
        """
        pass
    
    @abstractmethod
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current knowledge base.
        
        Returns:
            Dictionary containing knowledge base statistics
        """
        pass
    
    @abstractmethod
    def clear_knowledge_base(self) -> None:
        """
        Clear all documents from the knowledge base.
        
        Raises:
            RAGError: If clearing fails
        """
        pass


class RAGError(Exception):
    """Custom exception for RAG errors."""
    pass 