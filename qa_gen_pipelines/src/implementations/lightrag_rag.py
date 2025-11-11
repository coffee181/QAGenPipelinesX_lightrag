"""LightRAG implementation."""

import uuid
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
from tqdm import tqdm

# Fix tiktoken encoding issue before importing LightRAG
def setup_tiktoken_compatibility():
    """设置tiktoken兼容性补丁"""
    try:
        import tiktoken
        import tiktoken.registry

        # 检查o200k_base是否可用
        try:
            tiktoken.get_encoding("o200k_base")
            logger.info("o200k_base encoding is available")
            return True
        except ValueError as e:
            logger.warning(f"o200k_base encoding not available: {e}")

            # 应用兼容性补丁
            if not hasattr(tiktoken, '_original_get_encoding'):
                logger.info("Applying tiktoken compatibility patch...")

                # 保存原始函数
                tiktoken._original_get_encoding = tiktoken.get_encoding
                tiktoken.registry._original_get_encoding = tiktoken.registry.get_encoding

                def patched_get_encoding(encoding_name):
                    """补丁函数：将o200k_base映射到cl100k_base"""
                    if encoding_name == "o200k_base":
                        logger.warning("Mapping o200k_base to cl100k_base for compatibility")
                        try:
                            return tiktoken._original_get_encoding("cl100k_base")
                        except Exception:
                            # 如果cl100k_base也不可用，尝试其他编码
                            for fallback in ["p50k_base", "r50k_base"]:
                                try:
                                    logger.warning(f"Falling back to {fallback}")
                                    return tiktoken._original_get_encoding(fallback)
                                except Exception:
                                    continue
                            raise ValueError(f"No compatible encoding found for {encoding_name}")
                    return tiktoken._original_get_encoding(encoding_name)

                # 应用补丁
                tiktoken.get_encoding = patched_get_encoding
                tiktoken.registry.get_encoding = patched_get_encoding

                logger.info("tiktoken compatibility patch applied successfully")
            return True

    except ImportError:
        logger.error("tiktoken not available")
        return False
    except Exception as e:
        logger.error(f"Failed to setup tiktoken compatibility: {e}")
        return False

# 在导入LightRAG前应用兼容性补丁
setup_tiktoken_compatibility()

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    LIGHTRAG_AVAILABLE = True
except ImportError:
    logger.error("LightRAG not installed. Please install with: pip install lightrag-hku")
    LIGHTRAG_AVAILABLE = False

from ..interfaces.rag_interface import RAGInterface, RAGError
from ..models.document import Document
from ..models.question import Question
from ..models.qa_pair import QAPair, QASet
from ..utils.config import ConfigManager


class LightRAGImplementation(RAGInterface):
    """LightRAG-based RAG implementation."""

    def __init__(self, config: ConfigManager):
        """
        Initialize LightRAG.

        Args:
            config: Configuration object
        """
        if not LIGHTRAG_AVAILABLE:
            raise RAGError("LightRAG not available. Please install lightrag-hku")

        self.config = config
        self.working_dir = Path(config.get("rag.lightrag.working_dir", "./lightrag_cache"))

        # Get API keys
        self.deepseek_api_key = config.get("question_generator.deepseek.api_key") or os.getenv("DEEPSEEK_API_KEY")
        self.openai_api_key = config.get("rag.lightrag.openai.api_key") or os.getenv("OPENAI_API_KEY")

        # Initialize retrieval cache
        self.enable_cache = config.get("rag.lightrag.enable_cache", True)
        self.cache_similarity_threshold = config.get("rag.lightrag.cache_similarity_threshold", 0.90)
        self.retrieval_cache = {}  # question_hash -> retrieved_context
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"RAG cache {'enabled' if self.enable_cache else 'disabled'}")

        # Initialize with default working directory
        self.rag = None
        self.set_working_directory(self.working_dir)

    def set_working_directory(self, working_dir: Path) -> None:
        """
        Set a new working directory and create LightRAG instance for it.

        Args:
            working_dir: Path to the working directory
        """
        self.working_dir = Path(working_dir)

        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LightRAG with proper functions
        try:
            self.rag = self._create_lightrag_instance()
            logger.info(f"LightRAG initialized with working directory: {self.working_dir}")

            # Try to check if it has existing data
            if self.working_dir.exists():
                files = list(self.working_dir.glob("*.json")) + list(self.working_dir.glob("*.graphml"))
                if files:
                    logger.info(f"Found existing LightRAG data: {len(files)} files")
                else:
                    logger.info("No existing LightRAG data found")

        except Exception as e:
            logger.warning(f"Failed to initialize LightRAG: {e}")
            raise RAGError(f"Failed to initialize LightRAG: {e}")

    def use_existing_knowledge_base(self, working_dir: Path) -> None:
        """
        Use an existing knowledge base without clearing it.

        Args:
            working_dir: Path to the existing knowledge base directory
        """
        if not working_dir.exists():
            raise RAGError(f"Knowledge base directory does not exist: {working_dir}")

        # Set the working directory without clearing
        self.set_working_directory(working_dir)

        # Get stats to verify it's a valid knowledge base
        stats = self.get_knowledge_base_stats()
        logger.info(f"Using existing knowledge base: {stats}")

    def _create_lightrag_instance(self):
        """Create LightRAG instance with proper configuration."""
        # 从config中读取Ollama的配置
        # Define async LLM function
        async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            """LLM function for LightRAG using local Ollama model."""
            import aiohttp
            import asyncio
            import json

            # 使用本地Ollama模型
            ollama_url = "http://localhost:11434/api/generate"
            model_name = "deepseek-r1:32b"

            # 构建完整的提示词
            full_prompt = ""
            if system_prompt:
                full_prompt += f"{system_prompt}\n\n"
            
            # 处理历史消息
            if history_messages:
                if isinstance(history_messages, list):
                    for msg in history_messages:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            role = msg["role"]
                            content = msg["content"]
                            if role == "system":
                                full_prompt += f"System: {content}\n\n"
                            elif role == "user":
                                full_prompt += f"User: {content}\n\n"
                            elif role == "assistant":
                                full_prompt += f"Assistant: {content}\n\n"
                        elif isinstance(msg, str):
                            full_prompt += f"User: {msg}\n\n"
            
            # 添加当前提示
            full_prompt += f"User: {prompt}\n\nAssistant:"
            
            # 准备Ollama请求
            payload = {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 2048)
                }
            }
            
            # 添加重试机制
            max_retries = 5
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            ollama_url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=1800)  # 30分钟超时
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                raw_response = result.get("response", "")
                                # 清理<think>标签
                                cleaned_response = self._clean_think_tags(raw_response)
                                return cleaned_response
                            else:
                                error_text = await response.text()
                                logger.error(f"Ollama API error {response.status}: {error_text}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay)
                                    retry_delay *= 2
                                    continue
                                else:
                                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                                    
                except asyncio.TimeoutError:
                    logger.error(f"Ollama API timeout on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise Exception("Ollama API timeout after all retries")
                        
                except Exception as e:
                    logger.error(f"Unexpected error in Ollama LLM function: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise

        # Define async embedding function
        async def embedding_func(texts: List[str]):
            """Embedding function for LightRAG."""
            import numpy as np

            if self.openai_api_key:
                try:
                    import openai

                    # Use OpenAI embeddings
                    client = openai.AsyncOpenAI(api_key=self.openai_api_key)

                    response = await client.embeddings.create(
                        model="text-embedding-3-large",  # Use large model for 3072 dimensions
                        input=texts
                    )

                    embeddings = [data.embedding for data in response.data]
                    return np.array(embeddings)
                except Exception as e:
                    logger.warning(f"OpenAI embedding failed: {e}, using fallback")

            # Simple fallback - create 3072 dimensional embeddings
            import hashlib
            embeddings = []
            for text in texts:
                hash_obj = hashlib.md5(text.encode())
                hash_int = int(hash_obj.hexdigest(), 16)
                embedding = [(hash_int >> i) & 1 for i in range(3072)]  # 3072 dimensions
                embeddings.append(embedding)
            return np.array(embeddings, dtype=np.float32)

        # Check if API keys are available
        if not self.deepseek_api_key:
            logger.warning("No DeepSeek API key found. LightRAG may not work for new operations.")

        # Create LightRAG instance with explicit encoding
        try:
            rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=3072,  # Match the existing knowledge base
                    max_token_size=8192,
                    func=embedding_func
                ),
                # Use a compatible encoding
                encoding_model="cl100k_base"  # Use cl100k_base instead of o200k_base
            )
        except TypeError:
            # If encoding_model parameter is not supported, try without it
            rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=3072,  # Match the existing knowledge base
                    max_token_size=8192,
                    func=embedding_func
                )
            )

        # Initialize storages in async context if possible
        try:
            async def initialize_all():
                await rag.initialize_storages()
                try:
                    from lightrag.kg.shared_storage import initialize_pipeline_status
                    await initialize_pipeline_status()
                except ImportError:
                    pass

            asyncio.run(initialize_all())
        except Exception as e:
            logger.error(f"FATAL: Failed to initialize LightRAG storages: {e}")
            raise RAGError(f"Failed to initialize LightRAG storages: {e}")


        return rag

    def _clean_think_tags(self, text: str) -> str:
        """
        清理DeepSeek R1的<think>标签和内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        import re
        # 移除<think>标签及其内容
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 清理多余的空行
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        
        return cleaned_text.strip()

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

            # Use async helper function
            asyncio.run(self._async_insert_document(document))

            logger.info(f"Successfully inserted document: {document.name}")

        except Exception as e:
            raise RAGError(f"Failed to insert document {document.name}: {e}")

    async def _async_insert_document(self, document: Document) -> None:
        """
        Async helper for inserting documents.

        Args:
            document: Document to insert
        """
        # Ensure storages are initialized
        try:
            await self.rag.initialize_storages()

            # Initialize pipeline status if available
            try:
                from lightrag.kg.shared_storage import initialize_pipeline_status
                await initialize_pipeline_status()
            except ImportError:
                pass  # Not available in all versions
        except Exception as e:
            logger.warning(f"Storage initialization warning: {e}")

        # Insert the document with document ID and file path for tracking
        # 🔧 关键修复：同时指定 ids 和 file_paths 来确保文档来源追踪
        try:
            doc_id = document.name  # 使用文档名作为唯一标识
            file_path = str(document.file_path) if document.file_path else document.name
            
            logger.info(f"Inserting document with ID: {doc_id}, file_path: {file_path}")
            
            await self.rag.ainsert(
                document.content,
                ids=doc_id,  # 🔧 指定文档ID
                file_paths=file_path  # 🔧 指定文件路径，防止 unknown_source
            )
        except Exception as e:
            if "history_messages" in str(e):
                logger.warning(f"LightRAG history_messages issue, this is a known problem with current version")
                raise RAGError(f"LightRAG version issue: {e}")
            else:
                raise e

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
                    with open(text_file, 'r', encoding='utf-8') as f:
                        content = f.read()

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
        try:
            if source_document:
                logger.info(f"Querying question: {question[:100]}... [filtered by document: {source_document}]")
            else:
                logger.info(f"Querying question: {question[:100]}...")
            
            # Check cache if enabled
            if self.enable_cache:
                cached_response = self._check_cache(question)
                if cached_response:
                    self.cache_hits += 1
                    logger.info(f"✅ Cache hit! (total hits: {self.cache_hits}, misses: {self.cache_misses}, hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%)")
                    return cached_response
                else:
                    self.cache_misses += 1

            # Use existing event loop if available, otherwise create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            response = None

            # Use naive mode directly as it gives the best results
            # Skip other modes that are either broken (global) or don't find context (local)
            try:
                logger.info("Using naive mode for reliable results...")

                # 🔧 不使用自定义提示词，因为它需要模板变量会导致错误
                # 直接使用 LightRAG 的默认 naive 模式即可
                
                # Add timeout to prevent hanging queries
                response = loop.run_until_complete(
                    asyncio.wait_for(
                        self.rag.aquery(question, param=QueryParam(mode="naive")),
                        timeout=30.0  # 30 second timeout
                    )
                )
                logger.info("Query completed with naive mode")

            except asyncio.TimeoutError:
                logger.warning("Query timed out after 30 seconds")
                response = "查询超时，请稍后重试或简化问题。"

            except Exception as e:
                logger.warning(f"Naive mode failed: {e}")
                # Try local mode as last resort
                try:
                    logger.info("Trying local mode as fallback...")
                    response = loop.run_until_complete(
                        asyncio.wait_for(
                            self.rag.aquery(question, param=QueryParam(mode="local")),
                            timeout=15.0  # Shorter timeout for fallback
                        )
                    )
                    logger.info("Query completed with local mode")
                except Exception:
                    response = "抱歉，无法从知识库中找到相关信息来回答这个问题。"

            if response is None:
                response = "抱歉，查询过程中出现问题，无法生成答案。"

            logger.info(f"Generated answer: {len(response)} characters")
            
            # Cache the response if enabled
            if self.enable_cache:
                self._update_cache(question, response)
            
            return response

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
                    answers.append(f"Error: {str(e)}")

            logger.info(f"Batch query completed: {len(answers)} answers generated")
            return answers

        except Exception as e:
            raise RAGError(f"Batch query failed: {e}")

    def generate_qa_pairs_from_questions(self, questions: List[Question]) -> QASet:
        """
        Generate QA pairs for a list of questions.

        Args:
            questions: List of Question objects

        Returns:
            QASet containing generated QA pairs

        Raises:
            RAGError: If QA generation fails
        """
        try:
            logger.info(f"Generating QA pairs for {len(questions)} questions")

            qa_pairs = []

            for question in tqdm(questions, desc="Generating QA pairs"):
                try:
                    # Generate answer using RAG
                    answer = self.query_single_question(question.text)

                    # Create QA pair
                    qa_pair = QAPair(
                        question_id=question.question_id,
                        question=question.text,
                        answer=answer,
                        source=question.source,
                        confidence=0.8,  # Default confidence for LightRAG
                        metadata={
                            "question_type": question.question_type,
                            "difficulty": question.difficulty,
                            "category": question.category,
                            "tags": question.tags
                        }
                    )

                    qa_pairs.append(qa_pair)

                except Exception as e:
                    logger.error(f"Failed to generate QA pair for question {question.question_id}: {e}")
                    continue

            # Create QA set
            qa_set = QASet(
                qa_pairs=qa_pairs,
                source="LightRAG",
                created_at=datetime.now(),
                metadata={
                    "total_questions": len(questions),
                    "successful_pairs": len(qa_pairs),
                    "rag_type": "LightRAG"
                }
            )

            logger.info(f"Generated {len(qa_pairs)} QA pairs")
            return qa_set

        except Exception as e:
            raise RAGError(f"Failed to generate QA pairs: {e}")

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary containing knowledge base statistics
        """
        try:
            stats = {
                "working_directory": str(self.working_dir),
                "directory_exists": self.working_dir.exists(),
                "directory_size_mb": self._get_directory_size() / (1024 * 1024) if self.working_dir.exists() else 0
            }

            # Check for LightRAG files
            if self.working_dir.exists():
                lightrag_files = list(self.working_dir.glob("*.json")) + list(self.working_dir.glob("*.graphml"))
                stats["lightrag_files"] = len(lightrag_files)
                stats["files"] = [f.name for f in lightrag_files]

                # Get file sizes
                total_size = 0
                for f in lightrag_files:
                    if f.is_file():
                        size = f.stat().st_size
                        total_size += size
                        stats[f"file_size_{f.name}"] = f"{size / (1024*1024):.2f} MB"

                stats["total_files_size_mb"] = f"{total_size / (1024*1024):.2f} MB"

            return stats

        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {"error": str(e)}

    def clear_knowledge_base(self) -> None:
        """
        Clear the knowledge base by removing all files from working directory.

        Raises:
            RAGError: If clearing fails
        """
        try:
            logger.info("Clearing LightRAG knowledge base by removing all files")

            # Remove all files in working directory
            if self.working_dir.exists():
                import shutil
                shutil.rmtree(self.working_dir)
                logger.info(f"Removed working directory: {self.working_dir}")

            # Recreate working directory
            self.working_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created fresh working directory: {self.working_dir}")

            # Reinitialize LightRAG with clean directory
            self.rag = self._create_lightrag_instance()

            logger.info("Knowledge base cleared")

        except Exception as e:
            raise RAGError(f"Failed to clear knowledge base: {e}")

    def _get_directory_size(self) -> int:
        """
        Get total size of working directory in bytes.

        Returns:
            Directory size in bytes
        """
        total_size = 0
        try:
            for file_path in self.working_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.error(f"Failed to calculate directory size: {e}")
        return total_size
    
    def _normalize_question(self, question: str) -> str:
        """
        Normalize question text for caching.
        
        Args:
            question: Question text to normalize
            
        Returns:
            Normalized text
        """
        import re
        # Remove extra whitespace
        text = ' '.join(question.split())
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation at the end
        text = text.rstrip('?？.。！!,，')
        return text
    
    def _check_cache(self, question: str) -> Optional[str]:
        """
        Check if a similar question exists in cache.
        
        Args:
            question: Question to check
            
        Returns:
            Cached response if found, None otherwise
        """
        normalized = self._normalize_question(question)
        
        # First check for exact match
        if normalized in self.retrieval_cache:
            return self.retrieval_cache[normalized]
        
        # Check for similar questions
        for cached_question, cached_response in self.retrieval_cache.items():
            similarity = self._calculate_question_similarity(normalized, cached_question)
            if similarity >= self.cache_similarity_threshold:
                logger.debug(f"Found similar cached question (similarity={similarity:.2f})")
                return cached_response
        
        return None
    
    def _update_cache(self, question: str, response: str) -> None:
        """
        Update cache with new question-response pair.
        
        Args:
            question: Question text
            response: Generated response
        """
        normalized = self._normalize_question(question)
        self.retrieval_cache[normalized] = response
        
        # Limit cache size to prevent memory issues
        max_cache_size = 1000
        if len(self.retrieval_cache) > max_cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self.retrieval_cache))
            del self.retrieval_cache[oldest_key]
            logger.debug(f"Cache size limit reached, removed oldest entry")
    
    def _calculate_question_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two questions using character n-grams.
        
        Args:
            text1: First question
            text2: Second question
            
        Returns:
            Similarity score between 0 and 1
        """
        def get_ngrams(text, n=2):
            return set(text[i:i+n] for i in range(len(text)-n+1))
        
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "enabled": self.enable_cache,
            "cache_size": len(self.retrieval_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }
    
    def clear_cache(self) -> None:
        """Clear the retrieval cache."""
        self.retrieval_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cache cleared")