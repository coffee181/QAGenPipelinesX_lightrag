"""Simple RAG implementation without external dependencies."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from ..interfaces.rag_interface import RAGInterface, RAGError
from ..models.document import Document
from ..models.question import Question
from ..models.qa_pair import QAPair, QASet
from ..utils.config import ConfigManager
from ..utils.file_utils import FileUtils


class SimpleRAGImplementation(RAGInterface):
    """Simple RAG implementation using basic text matching and embeddings."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize Simple RAG.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.working_dir = Path(config.get("rag.simple.working_dir", "./simple_rag_cache"))
        self.model = config.get("rag.simple.model", "deepseek-chat")
        self.max_context_length = config.get("rag.simple.max_context_length", 4000)
        self.max_results = config.get("rag.simple.max_results", 3)
        
        # Get API key
        self.api_key = config.get("question_generator.deepseek.api_key")
        
        if not self.api_key:
            raise RAGError("API key not found in configuration")
        
        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client for DeepSeek
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        
        # Initialize document storage
        self.documents_file = self.working_dir / "documents.json"
        self.documents = self._load_documents()
        
        logger.info(f"Simple RAG initialized with working directory: {self.working_dir}")
    
    def insert_document(self, document: Document) -> None:
        """
        Insert a single document into the knowledge base.
        
        Args:
            document: Document to insert
            
        Raises:
            RAGError: If insertion fails
        """
        try:
            logger.info(f"Inserting document: {document.name}")
            
            # Store document metadata and content
            doc_data = {
                "id": str(document.file_path),
                "name": document.name,
                "content": document.content,
                "file_type": document.file_type,
                "file_size": document.file_size,
                "created_at": document.created_at.isoformat() if document.created_at else None,
                "processed_at": datetime.now().isoformat()
            }
            
            # Add to documents list
            self.documents[doc_data["id"]] = doc_data
            
            # Save to file
            self._save_documents()
            
            logger.info(f"Successfully inserted document: {document.name}")
            
        except Exception as e:
            raise RAGError(f"Failed to insert document {document.name}: {e}")
    
    def insert_documents_batch(self, documents: List[Document]) -> None:
        """
        Insert multiple documents into the knowledge base.
        
        Args:
            documents: List of documents to insert
            
        Raises:
            RAGError: If batch insertion fails
        """
        try:
            logger.info(f"Inserting {len(documents)} documents in batch")
            
            for document in tqdm(documents, desc="Inserting documents"):
                try:
                    self.insert_document(document)
                except Exception as e:
                    logger.error(f"Failed to insert document {document.name}: {e}")
                    continue
            
            logger.info(f"Batch insertion completed")
            
        except Exception as e:
            raise RAGError(f"Batch insertion failed: {e}")
    
    def insert_from_directory(self, directory_path: Path) -> None:
        """
        Insert all text files from a directory.
        
        Args:
            directory_path: Path to directory containing text files
            
        Raises:
            RAGError: If directory insertion fails
        """
        try:
            if not directory_path.exists():
                raise RAGError(f"Directory does not exist: {directory_path}")
            
            # Find all text files
            text_files = list(directory_path.glob("*.txt"))
            
            if not text_files:
                logger.warning(f"No text files found in directory: {directory_path}")
                return
            
            logger.info(f"Found {len(text_files)} text files in directory: {directory_path}")
            
            for text_file in tqdm(text_files, desc="Processing text files"):
                try:
                    # Read file content
                    content = FileUtils.load_text_file(text_file)
                    
                    # Create document object
                    document = Document(
                        file_path=text_file,
                        content=content,
                        file_type=text_file.suffix,
                        file_size=text_file.stat().st_size,
                        created_at=datetime.fromtimestamp(text_file.stat().st_ctime),
                        processed_at=datetime.now()
                    )
                    
                    # Insert document
                    self.insert_document(document)
                    
                except Exception as e:
                    logger.error(f"Failed to process file {text_file}: {e}")
                    continue
            
            logger.info(f"Directory insertion completed")
            
        except Exception as e:
            raise RAGError(f"Failed to insert from directory {directory_path}: {e}")
    
    def query_single_question(self, question: str) -> str:
        """
        Query the knowledge base with a single question.
        
        Args:
            question: Question to query
            
        Returns:
            Generated answer
            
        Raises:
            RAGError: If query fails
        """
        try:
            logger.info(f"Querying question: {question[:100]}...")
            
            # Retrieve relevant documents
            relevant_docs = self._retrieve_relevant_documents(question)
            
            if not relevant_docs:
                return "抱歉，我无法在知识库中找到相关信息来回答这个问题。"
            
            # Build context from relevant documents
            context = self._build_context(relevant_docs)
            
            # Generate answer using LLM
            answer = self._generate_answer(question, context)
            
            logger.info(f"Generated answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            raise RAGError(f"Failed to query question: {e}")
    
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
        try:
            logger.info(f"Querying {len(questions)} questions in batch")
            
            answers = []
            
            for question in tqdm(questions, desc="Querying questions"):
                try:
                    answer = self.query_single_question(question)
                    answers.append(answer)
                except Exception as e:
                    logger.error(f"Failed to query question: {e}")
                    answers.append("")  # Add empty answer for failed queries
                    continue
            
            logger.info(f"Batch query completed. Generated {len(answers)} answers")
            return answers
            
        except Exception as e:
            raise RAGError(f"Batch query failed: {e}")
    
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
        try:
            if not questions:
                raise RAGError("No questions provided for QA generation")
            
            document_id = questions[0].source_document
            logger.info(f"Generating QA pairs for {len(questions)} questions from document: {document_id}")
            
            qa_pairs = []
            
            for question in tqdm(questions, desc="Generating QA pairs"):
                try:
                    # Query the knowledge base
                    answer = self.query_single_question(question.content)
                    
                    # Create QA pair
                    qa_pair = QAPair(
                        question_id=question.question_id,
                        question=question.content,
                        answer=answer,
                        source_document=question.source_document,
                        created_at=datetime.now()
                    )
                    
                    qa_pairs.append(qa_pair)
                    
                except Exception as e:
                    logger.error(f"Failed to generate QA pair for question {question.question_id}: {e}")
                    continue
            
            # Create QASet
            qa_set = QASet(
                document_id=document_id,
                qa_pairs=qa_pairs,
                created_at=datetime.now()
            )
            
            logger.info(f"Generated {len(qa_pairs)} QA pairs for document: {document_id}")
            return qa_set
            
        except Exception as e:
            raise RAGError(f"Failed to generate QA pairs: {e}")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current knowledge base.
        
        Returns:
            Dictionary containing knowledge base statistics
        """
        try:
            stats = {
                "working_directory": str(self.working_dir),
                "model": self.model,
                "max_context_length": self.max_context_length,
                "document_count": len(self.documents),
                "directory_exists": self.working_dir.exists(),
                "directory_size": self._get_directory_size()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {}
    
    def clear_knowledge_base(self) -> None:
        """
        Clear all documents from the knowledge base.
        
        Raises:
            RAGError: If clearing fails
        """
        try:
            logger.warning("Clearing knowledge base...")
            
            # Clear documents
            self.documents = {}
            self._save_documents()
            
            logger.info("Knowledge base cleared successfully")
            
        except Exception as e:
            raise RAGError(f"Failed to clear knowledge base: {e}")
    
    def _retrieve_relevant_documents(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the question using simple text matching.
        
        Args:
            question: Query question
            
        Returns:
            List of relevant document data
        """
        relevant_docs = []
        question_lower = question.lower()
        
        # Simple keyword-based retrieval
        for doc_id, doc_data in self.documents.items():
            content_lower = doc_data["content"].lower()
            
            # Calculate relevance score based on keyword overlap
            question_words = set(question_lower.split())
            content_words = set(content_lower.split())
            
            overlap = len(question_words.intersection(content_words))
            if overlap > 0:
                doc_data["relevance_score"] = overlap / len(question_words)
                relevant_docs.append(doc_data)
        
        # Sort by relevance score and return top results
        relevant_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_docs[:self.max_results]
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Build context string from relevant documents.
        
        Args:
            documents: List of relevant document data
            
        Returns:
            Context string
        """
        context_parts = []
        total_length = 0
        
        for doc in documents:
            content = doc["content"]
            
            # Truncate if context becomes too long
            if total_length + len(content) > self.max_context_length:
                remaining_length = self.max_context_length - total_length
                if remaining_length > 100:  # Only add if meaningful length remains
                    content = content[:remaining_length] + "..."
                    context_parts.append(f"文档: {doc['name']}\n{content}")
                break
            
            context_parts.append(f"文档: {doc['name']}\n{content}")
            total_length += len(content)
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with context.
        
        Args:
            question: Question to answer
            context: Context from relevant documents
            
        Returns:
            Generated answer
        """
        system_prompt = """你是一个专业的问答助手。请根据提供的文档内容回答用户的问题。
        
要求：
1. 仅基于提供的文档内容回答
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、详细且有用
4. 使用中文回答"""
        
        user_prompt = f"""文档内容：
{context}

问题：{question}

请根据上述文档内容回答问题："""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "抱歉，生成答案时出现错误。"
    
    def _load_documents(self) -> Dict[str, Dict[str, Any]]:
        """Load documents from storage file."""
        try:
            if self.documents_file.exists():
                return FileUtils.load_json_file(self.documents_file)
            return {}
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return {}
    
    def _save_documents(self) -> None:
        """Save documents to storage file."""
        try:
            FileUtils.save_json_file(self.documents, self.documents_file)
        except Exception as e:
            logger.error(f"Failed to save documents: {e}")
    
    def _get_directory_size(self) -> int:
        """Get total size of working directory in bytes."""
        try:
            total_size = 0
            for file_path in self.working_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0 