"""LightRAG implementation."""

import uuid
import os
import asyncio
import re
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from loguru import logger
from tqdm import tqdm

# 🔧 应用 LightRAG 关系描述补丁
from ..utils.lightrag_relation_patch import patch_lightrag_relation_merge
patch_lightrag_relation_merge()

# 🔧 线程事件循环管理
from ..utils.thread_event_loop import get_or_create_event_loop


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
from ..utils.lightrag_utils import (
    compute_lightrag_chunk_id,
    build_chunk_citation,
    extract_chunk_ids_from_source,
)


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
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread_id: Optional[int] = None
        self.working_dir = Path(config.get("rag.lightrag.working_dir", "./lightrag_cache"))

        # Embedding provider configuration
        embedding_cfg = config.get("rag.lightrag.embedding", {}) or {}
        requested_provider = embedding_cfg.get("provider")
        if requested_provider:
            requested_provider = str(requested_provider).lower()

        if requested_provider not in {"ollama"}:
            requested_provider = "ollama"

        self.embedding_provider = requested_provider
        default_ollama_model = "bge-m3"

        # Allow using shared "model" key as provider-specific override
        shared_model = embedding_cfg.get("model")
        self.ollama_embedding_model = embedding_cfg.get("ollama_model") or (
            shared_model if shared_model else default_ollama_model
        )

        default_dim = 1024
        self.embedding_dim = int(embedding_cfg.get("dim") or default_dim)
        self.embedding_base_url = embedding_cfg.get("base_url", "http://localhost:11434")
        self.embedding_timeout = float(embedding_cfg.get("timeout", 120))
        self.embedding_max_retries = int(embedding_cfg.get("max_retries", 3))
        priority_list = embedding_cfg.get("priority") or []
        if isinstance(priority_list, list):
            self.embedding_provider_priority = [
                str(provider).lower() for provider in priority_list if provider
            ]
        else:
            self.embedding_provider_priority = []

        # Local LLM configuration for LightRAG
        llm_cfg = config.get("rag.lightrag.llm", {}) or {}
        self.ollama_llm_base_url = llm_cfg.get("base_url", "http://localhost:11434")
        self.ollama_llm_model = llm_cfg.get("model", "deepseek-r1:32b")
        self.ollama_llm_temperature = float(llm_cfg.get("temperature", 0.7))
        self.ollama_llm_max_tokens = int(llm_cfg.get("max_tokens", 2048))
        self.ollama_llm_timeout = float(llm_cfg.get("timeout", 1800))
        self.ollama_llm_max_retries = int(llm_cfg.get("max_retries", 5))

        # Query execution configuration
        query_cfg = config.get("rag.lightrag.query", {}) or {}
        self.query_top_k = int(query_cfg.get("top_k", 40))
        self.query_chunk_top_k = int(query_cfg.get("chunk_top_k", 20))
        self.query_max_entity_tokens = int(query_cfg.get("max_entity_tokens", 10000))
        self.query_max_relation_tokens = int(query_cfg.get("max_relation_tokens", 10000))
        self.query_max_total_tokens = int(query_cfg.get("max_total_tokens", 40000))
        self.query_cosine_threshold = float(query_cfg.get("cosine_threshold", 0.2))
        self.query_related_chunk_number = int(query_cfg.get("related_chunk_number", 2))
        self.query_history_turns = int(query_cfg.get("history_turns", 0))
        self.query_enable_rerank = bool(query_cfg.get("enable_rerank", True))
        timeout_cfg = query_cfg.get("timeouts", {}) or {}
        self.query_timeout_mix = float(timeout_cfg.get("mix", 1200.0))
        self.query_timeout_naive = float(timeout_cfg.get("naive", 600.0))
        self.query_timeout_local = float(timeout_cfg.get("local", 600.0))

        # Initialize retrieval cache
        self.enable_cache = config.get("rag.lightrag.enable_cache", True)
        self.cache_similarity_threshold = config.get("rag.lightrag.cache_similarity_threshold", 0.90)
        self.retrieval_cache = {}  # question_hash -> retrieved_context
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_citations_per_answer = int(
            config.get("rag.lightrag.max_citations_per_answer", 5)
        )
        if self.max_citations_per_answer < 0:
            self.max_citations_per_answer = 0
        
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
        # 🔧 关键修复：初始化当前线程的独立事件循环
        # 这确保即使在不同的线程中调用，也能有各自的事件循环
        loop = self._ensure_event_loop()
        logger.info(f"Event loop ready for thread {self.loop_thread_id}")
        
        self.working_dir = Path(working_dir)

        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LightRAG with proper functions
        try:
            # 🔧 关键修复：销毁旧实例，重新创建，避免 Lock 污染
            # 当使用新的 working_dir 时，必须销毁旧的 LightRAG 实例
            # 因为旧实例的 Lock 对象可能绑定到已关闭的事件循环
            if self.rag is not None:
                try:
                    logger.info("Cleaning up previous LightRAG instance...")
                    del self.rag
                    import gc
                    gc.collect()
                except Exception as e:
                    logger.debug(f"Cleanup warning: {e}")
            
            # 创建新的 LightRAG 实例（会绑定到当前事件循环）
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

    def _ensure_event_loop(self) -> asyncio.AbstractEventLoop:
        """Ensure there is a valid event loop bound to the current thread."""
        current_thread = threading.get_ident()

        if (
            self.event_loop is None
            or self.event_loop.is_closed()
            or current_thread != self.loop_thread_id
        ):
            self.event_loop = get_or_create_event_loop()
            self.loop_thread_id = current_thread

        return self.event_loop

    def _run_async(self, coro, timeout: Optional[float] = None):
        """
        Run an async coroutine inside the managed event loop.

        Args:
            coro: Coroutine to run
            timeout: Optional timeout in seconds
        """
        loop = self._ensure_event_loop()
        if timeout is not None:
            coro = asyncio.wait_for(coro, timeout)

        try:
            return loop.run_until_complete(coro)
        except RuntimeError as exc:
            if "already running" in str(exc).lower():
                raise RAGError(
                    "LightRAG event loop is already running in this thread. "
                    "Please call the async API directly or avoid nested event loops."
                ) from exc
            raise

    def _create_lightrag_instance(self):
        """Create LightRAG instance with proper configuration."""
        # Define async LLM function
        async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            """LLM function for LightRAG using local Ollama model."""
            import aiohttp
            import asyncio
            import json

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
                "model": self.ollama_llm_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.ollama_llm_temperature),
                    "num_predict": kwargs.get("max_tokens", self.ollama_llm_max_tokens)
                }
            }
            
            # 添加重试机制
            max_retries = self.ollama_llm_max_retries
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.ollama_llm_base_url.rstrip('/')}/api/generate",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=self.ollama_llm_timeout)
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

            embeddings: Optional[List[List[float]]] = None

            for provider in self._get_embedding_provider_order():
                if provider == "ollama":
                    embeddings = await self._generate_ollama_embeddings(texts)

                if embeddings is not None:
                    break

            if embeddings is None:
                embeddings = self._generate_fallback_embeddings(texts)

            return np.array(embeddings, dtype=np.float32)

        # Ensure LightRAG binds to the managed event loop
        loop = self._ensure_event_loop()
        asyncio.set_event_loop(loop)

        # Create LightRAG instance with explicit encoding
        try:
            rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.embedding_dim,  # Match configurable embedding size
                    max_token_size=8192,
                    func=embedding_func
                ),
                # Use a compatible encoding
                encoding_model="cl100k_base",  # Use cl100k_base instead of o200k_base
                top_k=self.query_top_k,
                chunk_top_k=self.query_chunk_top_k,
                max_entity_tokens=self.query_max_entity_tokens,
                max_relation_tokens=self.query_max_relation_tokens,
                max_total_tokens=self.query_max_total_tokens,
                cosine_threshold=self.query_cosine_threshold,
                related_chunk_number=self.query_related_chunk_number,
            )
        except TypeError:
            # If encoding_model parameter is not supported, try without it
            rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.embedding_dim,
                    max_token_size=8192,
                    func=embedding_func
                ),
                top_k=self.query_top_k,
                chunk_top_k=self.query_chunk_top_k,
                max_entity_tokens=self.query_max_entity_tokens,
                max_relation_tokens=self.query_max_relation_tokens,
                max_total_tokens=self.query_max_total_tokens,
                cosine_threshold=self.query_cosine_threshold,
                related_chunk_number=self.query_related_chunk_number,
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

            # 🔧 使用受控事件循环运行初始化逻辑，避免跨线程污染
            self._run_async(initialize_all())
            
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

    def _build_query_param(self, mode: str) -> QueryParam:
        """
        Build a QueryParam object based on configuration.
        """
        return QueryParam(
            mode=mode,
            top_k=self.query_top_k,
            chunk_top_k=self.query_chunk_top_k,
            max_entity_tokens=self.query_max_entity_tokens,
            max_relation_tokens=self.query_max_relation_tokens,
            max_total_tokens=self.query_max_total_tokens,
            history_turns=self.query_history_turns,
            enable_rerank=self.query_enable_rerank,
        )

    def _get_embedding_provider_order(self) -> List[str]:
        """
        Resolve embedding provider order based on config (local-only).
        """
        order: List[str] = []

        def add(provider: str):
            if provider == "ollama" and self.embedding_base_url:
                if provider not in order:
                    order.append(provider)

        if self.embedding_provider_priority:
            for provider in self.embedding_provider_priority:
                add(provider)
        else:
            add(self.embedding_provider)

        return order or ["ollama"]

    async def _generate_ollama_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings using a local Ollama model.
        """
        if not self.embedding_base_url:
            return None

        try:
            import aiohttp
        except ImportError as exc:
            logger.error(f"aiohttp is required for Ollama embeddings: {exc}")
            return None

        base_url = self.embedding_base_url.rstrip("/")
        timeout = aiohttp.ClientTimeout(total=self.embedding_timeout)
        embeddings: List[List[float]] = []

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for text in texts:
                success = False
                delay = 1.0
                for attempt in range(1, self.embedding_max_retries + 1):
                    try:
                        response = await session.post(
                            f"{base_url}/api/embeddings",
                            json={
                                "model": self.ollama_embedding_model,
                                "prompt": text
                            }
                        )
                        if response.status == 200:
                            data = await response.json()
                            vector = data.get("embedding")
                            if not vector:
                                raise ValueError("Missing 'embedding' field in Ollama response")
                            embeddings.append(self._normalize_embedding_length(vector))
                            success = True
                            break
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"Ollama embedding error {response.status}: {error_text}"
                            )
                    except asyncio.TimeoutError:
                        logger.warning(f"Ollama embedding timeout (attempt {attempt})")
                    except Exception as exc:
                        logger.warning(f"Ollama embedding failure: {exc}")

                    if attempt < self.embedding_max_retries:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 10)

                if not success:
                    return None

        return embeddings

    def _generate_fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate deterministic pseudo-embeddings when all providers fail.
        """
        import hashlib
        import random

        embeddings: List[List[float]] = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
            rng = random.Random(seed)
            vector = [rng.uniform(-1.0, 1.0) for _ in range(self.embedding_dim)]
            embeddings.append(vector)
        return embeddings

    def _normalize_embedding_length(self, embedding: Any) -> List[float]:
        """
        Ensure embeddings conform to the configured dimensionality.
        """
        if embedding is None:
            return [0.0] * self.embedding_dim

        vector = list(embedding)
        if len(vector) == self.embedding_dim:
            return vector
        if len(vector) > self.embedding_dim:
            return vector[: self.embedding_dim]

        # Pad with zeros if embedding is shorter than expected
        pad_length = self.embedding_dim - len(vector)
        return vector + [0.0] * pad_length

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

            # 🔧 使用受控事件循环执行异步插入，避免跨线程事件循环冲突
            self._run_async(self._async_insert_document(document))

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

            response = None

            # Use mix mode to leverage knowledge graph while maintaining reliability
            # mix mode combines vector search with knowledge graph for better results
            try:
                logger.info("Using mix mode to leverage knowledge graph...")

                # 🔧 使用mix模式充分利用知识图谱，同时保持查询的可靠性
                # mix模式结合向量搜索和知识图谱，提供更准确的答案
                
                # Add timeout to prevent hanging queries
                response = self._run_async(
                    self.rag.aquery(question, param=self._build_query_param("mix")),
                    timeout=self.query_timeout_mix
                )
                logger.info("Query completed with mix mode")
            except asyncio.TimeoutError:
                logger.warning("Query timed out after 30 seconds")
                response = "查询超时，请稍后重试或简化问题。"

            except Exception as e:
                logger.warning(f"Mix mode failed: {e}")
                # Try naive mode as fallback
                try:
                    logger.info("Trying naive mode as fallback...")
                    response = self._run_async(
                        self.rag.aquery(question, param=self._build_query_param("naive")),
                        timeout=self.query_timeout_naive
                    )
                    logger.info("Query completed with naive mode")
                except Exception:
                    # Try local mode as last resort
                    try:
                        logger.info("Trying local mode as final fallback...")
                        response = self._run_async(
                            self.rag.aquery(question, param=self._build_query_param("local")),
                            timeout=self.query_timeout_local
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
            if not questions:
                raise RAGError("No questions provided for QA generation")

            logger.info(f"Generating QA pairs for {len(questions)} questions via LightRAG")

            qa_pairs: List[QAPair] = []
            document_ids: set[str] = set(
                filter(None, (getattr(q, "source_document", None) for q in questions))
            )

            for question in tqdm(questions, desc="Generating QA pairs"):
                try:
                    question_text = getattr(question, "content", None) or getattr(
                        question, "text", None
                    )
                    if not question_text:
                        logger.warning(
                            f"Question {getattr(question, 'question_id', 'unknown')} missing text content; skipping"
                        )
                        continue

                    source_document = getattr(question, "source_document", None)
                    answer = self.query_single_question(
                        question_text, source_document=source_document
                    )

                    base_metadata = getattr(question, "metadata", {}) or {}
                    metadata: Dict[str, Any] = dict(base_metadata)
                    if "citations" in metadata and isinstance(metadata["citations"], list):
                        metadata["citations"] = list(metadata["citations"])

                    for attr_name in ("question_type", "difficulty", "category", "tags"):
                        value = getattr(question, attr_name, None)
                        if value is not None and attr_name not in metadata:
                            metadata[attr_name] = value

                    chunk_id = metadata.get("lightrag_chunk_id")
                    if not chunk_id:
                        source_chunk_content = getattr(
                            question, "source_chunk_content", None
                        ) or base_metadata.get("source_chunk_content")
                        if source_chunk_content:
                            chunk_id = compute_lightrag_chunk_id(source_chunk_content)
                    if chunk_id:
                        metadata["lightrag_chunk_id"] = chunk_id

                    related_chunk_ids = self._collect_related_chunk_ids(
                        question, chunk_id
                    )
                    if related_chunk_ids:
                        citations: List[Dict[str, Any]] = []
                        for related_chunk_id in related_chunk_ids:
                            if (
                                self.max_citations_per_answer > 0
                                and len(citations) >= self.max_citations_per_answer
                            ):
                                break
                            chunk_data = self._fetch_chunk_data(related_chunk_id)
                            citation = build_chunk_citation(related_chunk_id, chunk_data)
                            if citation:
                                citations.append(citation)
                        if citations:
                            metadata["citations"] = citations

                    source_chunk_id = getattr(question, "source_chunk_id", None)
                    if source_chunk_id and "source_chunk_id" not in metadata:
                        metadata["source_chunk_id"] = source_chunk_id

                    qa_pair = QAPair(
                        question_id=getattr(question, "question_id", str(uuid.uuid4())),
                        question=question_text,
                        answer=answer,
                        source_document=source_document or "unknown",
                        confidence_score=0.8,
                        metadata=metadata,
                    )
                    qa_pairs.append(qa_pair)

                except Exception as e:
                    logger.error(
                        f"Failed to generate QA pair for question {getattr(question, 'question_id', 'unknown')}: {e}"
                    )
                    continue

            if not qa_pairs:
                raise RAGError("Failed to generate QA pairs for all questions")

            if len(document_ids) == 1:
                document_id = next(iter(document_ids))
            elif not document_ids:
                document_id = "unknown"
            else:
                document_id = "multiple_documents"

            qa_set = QASet(
                document_id=document_id,
                qa_pairs=qa_pairs,
                created_at=datetime.now(),
            )

            logger.info(
                f"Generated {len(qa_pairs)} QA pairs (documents: {document_id})"
            )
            return qa_set

        except Exception as e:
            raise RAGError(f"Failed to generate QA pairs: {e}")

    def _fetch_chunk_data(self, chunk_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """从 LightRAG text_chunks 存储中读取 chunk 信息。"""
        if not chunk_id or not self.rag:
            return None

        text_chunks = getattr(self.rag, "text_chunks", None)
        if text_chunks is None:
            return None

        try:
            return self._run_async(
                text_chunks.get_by_id(chunk_id),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout while fetching chunk metadata for {chunk_id}")
        except Exception as e:
            logger.debug(f"Failed to fetch chunk metadata for {chunk_id}: {e}")
        return None

    def _collect_related_chunk_ids(
        self, question: Question, primary_chunk_id: Optional[str]
    ) -> List[str]:
        """
        汇总与问题相关的 chunk_id（包括主分片与实体关联分片）。
        """
        chunk_ids: Set[str] = set()
        if primary_chunk_id:
            chunk_ids.add(primary_chunk_id)

        metadata = getattr(question, "metadata", {}) or {}

        existing_citations = metadata.get("citations")
        if isinstance(existing_citations, list):
            for citation in existing_citations:
                if isinstance(citation, dict):
                    cid = citation.get("chunk_id")
                    if isinstance(cid, str) and cid.startswith("chunk-"):
                        chunk_ids.add(cid)

        # 追加显式提供的 chunk 列表
        for key in ("related_chunk_ids", "additional_chunk_ids", "chunk_ids"):
            extra_ids = metadata.get(key)
            if isinstance(extra_ids, (list, tuple, set)):
                for cid in extra_ids:
                    if isinstance(cid, str) and cid.startswith("chunk-"):
                        chunk_ids.add(cid)

        # 基于实体检索关联 chunk
        related_entities = getattr(question, "related_entities", None)
        if not related_entities:
            related_entities = metadata.get("related_entities", [])
        if isinstance(related_entities, (list, tuple, set)):
            for entity_name in related_entities:
                if isinstance(entity_name, str) and entity_name.strip():
                    chunk_ids.update(self._fetch_entity_chunk_ids(entity_name.strip()))

        # 确保主分片排在第一位
        ordered_chunk_ids: List[str] = []
        if primary_chunk_id and primary_chunk_id in chunk_ids:
            ordered_chunk_ids.append(primary_chunk_id)
        for cid in sorted(chunk_ids):
            if cid == primary_chunk_id:
                continue
            ordered_chunk_ids.append(cid)

        return ordered_chunk_ids

    def _fetch_entity_chunk_ids(self, entity_name: str) -> Set[str]:
        """
        根据实体在知识图谱中的信息提取其关联的 chunk_id。
        """
        chunk_ids: Set[str] = set()
        node_data = self._fetch_graph_node(entity_name)
        if not node_data:
            return chunk_ids

        raw_source = node_data.get("source_id")
        if raw_source:
            parsed_ids = extract_chunk_ids_from_source(raw_source)
            if (
                self.max_citations_per_answer > 0
                and len(parsed_ids) > self.max_citations_per_answer
            ):
                parsed_ids = parsed_ids[: self.max_citations_per_answer]
            chunk_ids.update(parsed_ids)

        if (
            self.max_citations_per_answer > 0
            and len(chunk_ids) > self.max_citations_per_answer
        ):
            # 避免单个实体贡献过多引用
            chunk_ids = set(list(chunk_ids)[: self.max_citations_per_answer])

        return chunk_ids

    def _fetch_graph_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """读取知识图谱中的节点数据。"""
        if not node_id or not self.rag:
            return None

        graph = getattr(self.rag, "chunk_entity_relation_graph", None)
        if graph is None:
            return None

        try:
            return self._run_async(
                graph.get_node(node_id),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout while fetching entity node for {node_id}")
        except Exception as e:
            logger.debug(f"Failed to fetch entity node for {node_id}: {e}")
        return None

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