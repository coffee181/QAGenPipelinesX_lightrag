"""Question generator interface definition."""

from abc import ABC, abstractmethod
from typing import List
from ..models.document import DocumentChunk
from ..models.question import Question, QuestionSet


class QuestionGeneratorInterface(ABC):
    """Abstract interface for question generation implementations."""
    
    @abstractmethod
    def generate_questions_from_chunk(self, chunk: DocumentChunk) -> List[Question]:
        """
        Generate questions from a single text chunk.
        
        Args:
            chunk: DocumentChunk to generate questions from
            
        Returns:
            List of Question objects
            
        Raises:
            QuestionGenerationError: If generation fails
        """
        pass
    
    @abstractmethod
    def generate_questions_from_chunks(self, chunks: List[DocumentChunk]) -> QuestionSet:
        """
        Generate questions from multiple text chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            QuestionSet containing all generated questions
            
        Raises:
            QuestionGenerationError: If generation fails
        """
        pass
    
    @abstractmethod
    def parse_questions_from_response(self, response: str, source_chunk: DocumentChunk) -> List[Question]:
        """
        Parse questions from LLM response.
        
        Args:
            response: Raw response from LLM
            source_chunk: Source chunk for the questions
            
        Returns:
            List of parsed Question objects
            
        Raises:
            QuestionGenerationError: If parsing fails
        """
        pass
    
    @abstractmethod
    def validate_questions(self, questions: List[Question]) -> bool:
        """
        Validate generated questions.
        
        Args:
            questions: List of questions to validate
            
        Returns:
            True if all questions are valid, False otherwise
        """
        pass
    
    @abstractmethod
    def set_custom_prompts(self, system_prompt: str, human_prompt: str) -> None:
        """
        Set custom prompts for question generation.
        
        Args:
            system_prompt: System prompt template
            human_prompt: Human prompt template
        """
        pass


class QuestionGenerationError(Exception):
    """Custom exception for question generation errors."""
    pass 