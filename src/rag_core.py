"""RAG核心模块 - 封装LightRAG"""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from loguru import logger
from tqdm import tqdm

# tiktoken兼容性处理
def _setup_tiktoken_compatibility():
    """设置tiktoken兼容性补丁"""
    try:
        import tiktoken
        import tiktoken.registry

        try:
            tiktoken.get_encoding("o200k_base")
            return True
        except ValueError:
            if not hasattr(tiktoken, '_original_get_encoding'):
                tiktoken._original_get_encoding = tiktoken.get_encoding
                tiktoken.registry._original_get_encoding = tiktoken.registry.get_encoding

                def patched_get_encoding(encoding_name):
                    if encoding_name == "o200k_base":
                        try:
                            return tiktoken._original_get_encoding("cl100k_base")
                        except Exception:
                            for fallback in ["p50k_base", "r50k_base"]:
                                try:
                                    return tiktoken._original_get_encoding(fallback)
                                except Exception:
                                    continue
                            raise ValueError(f"No compatible encoding found for {encoding_name}")
                    return tiktoken._original_get_encoding(encoding_name)

                tiktoken.get_encoding = patched_get_encoding
                tiktoken.registry.get_encoding = patched_get_encoding
            return True
    except ImportError:
        return False


_setup_tiktoken_compatibility()

# LightRAG导入
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LightRAG = None
    QueryParam = None
    EmbeddingFunc = None
    LIGHTRAG_AVAILABLE = False
    logger.warning("LightRAG 未安装，请运行: pip install lightrag-hku")


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """获取或创建事件循环"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def compute_chunk_id(content: Optional[str]) -> Optional[str]:
    """
    按照LightRAG规则计算chunk_id
    """
    if not content:
        return None

    cleaned = content.replace("\x00", "").strip()
    if not cleaned:
        return None

    try:
        from lightrag.utils import compute_mdhash_id
        return compute_mdhash_id(cleaned, prefix="chunk-")
    except ImportError:
        return "chunk-" + hashlib.md5(cleaned.encode("utf-8")).hexdigest()


class RAGCore:
    """
    RAG核心类 - 封装LightRAG的知识图谱构建和向量化功能
    """

    def __init__(
        self,
        working_dir: Path,
        embedding_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        query_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化RAG核心
        
        Args:
            working_dir: LightRAG工作目录
            embedding_config: Embedding配置
            llm_config: LLM配置
            query_config: 查询配置
        """
        if not LIGHTRAG_AVAILABLE:
            raise RuntimeError("LightRAG 不可用，请安装: pip install lightrag-hku")

        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Embedding配置
        emb_cfg = embedding_config or {}
        self.embedding_provider = emb_cfg.get("provider", "ollama")
        self.embedding_model = emb_cfg.get("model", "bge-m3")
        self.embedding_dim = int(emb_cfg.get("dim", 1024))
        self.embedding_base_url = emb_cfg.get("base_url", "http://localhost:11434")
        self.embedding_timeout = float(emb_cfg.get("timeout", 1200))
        self.embedding_max_retries = int(emb_cfg.get("max_retries", 3))

        # LLM配置
        llm_cfg = llm_config or {}
        self.llm_base_url = llm_cfg.get("base_url", "http://localhost:11434")
        self.llm_model = llm_cfg.get("model", "deepseek-r1:32b")
        self.llm_temperature = float(llm_cfg.get("temperature", 0.7))
        self.llm_max_tokens = int(llm_cfg.get("max_tokens", 2048))
        self.llm_timeout = float(llm_cfg.get("timeout", 1800))
        self.llm_max_retries = int(llm_cfg.get("max_retries", 5))

        # 查询配置
        query_cfg = query_config or {}
        self.query_top_k = int(query_cfg.get("top_k", 20))
        self.query_chunk_top_k = int(query_cfg.get("chunk_top_k", 10))
        self.query_max_entity_tokens = int(query_cfg.get("max_entity_tokens", 10000))
        self.query_max_relation_tokens = int(query_cfg.get("max_relation_tokens", 10000))
        self.query_max_total_tokens = int(query_cfg.get("max_total_tokens", 40000))
        self.query_cosine_threshold = float(query_cfg.get("cosine_threshold", 0.2))
        self.query_related_chunk_number = int(query_cfg.get("related_chunk_number", 2))

        # 事件循环管理
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread_id: Optional[int] = None

        # 初始化LightRAG
        self.rag = None
        self._initialize()

        logger.info(f"RAGCore 初始化完成: {self.working_dir}")

    def _ensure_event_loop(self) -> asyncio.AbstractEventLoop:
        """确保事件循环可用"""
        current_thread = threading.get_ident()

        if (
            self.event_loop is None
            or self.event_loop.is_closed()
            or current_thread != self.loop_thread_id
        ):
            self.event_loop = _get_or_create_event_loop()
            self.loop_thread_id = current_thread

        return self.event_loop

    def _run_async(self, coro, timeout: Optional[float] = None):
        """在事件循环中运行异步协程"""
        loop = self._ensure_event_loop()
        if timeout is not None:
            coro = asyncio.wait_for(coro, timeout)

        try:
            return loop.run_until_complete(coro)
        except RuntimeError as exc:
            if "already running" in str(exc).lower():
                raise RuntimeError(
                    "事件循环已在运行，请直接使用异步API"
                ) from exc
            raise

    def _initialize(self):
        """初始化LightRAG实例"""
        # 定义LLM函数
        async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            import aiohttp

            full_prompt = ""
            if system_prompt:
                full_prompt += f"{system_prompt}\n\n"

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

            full_prompt += f"User: {prompt}\n\nAssistant:"

            payload = {
                "model": self.llm_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.llm_temperature),
                    "num_predict": kwargs.get("max_tokens", self.llm_max_tokens)
                }
            }

            for attempt in range(self.llm_max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.llm_base_url.rstrip('/')}/api/generate",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=self.llm_timeout)
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                raw_response = result.get("response", "")
                                return self._clean_think_tags(raw_response)
                            else:
                                error_text = await response.text()
                                logger.error(f"Ollama API错误 {response.status}: {error_text}")
                                if attempt < self.llm_max_retries - 1:
                                    await asyncio.sleep(5 * (2 ** attempt))
                                    continue
                                raise RuntimeError(f"Ollama API错误 {response.status}")

                except asyncio.TimeoutError:
                    logger.error(f"Ollama API超时 (尝试 {attempt + 1})")
                    if attempt < self.llm_max_retries - 1:
                        await asyncio.sleep(5 * (2 ** attempt))
                        continue
                    raise RuntimeError("Ollama API超时")

                except Exception as e:
                    logger.error(f"Ollama API异常: {e}")
                    if attempt < self.llm_max_retries - 1:
                        await asyncio.sleep(5 * (2 ** attempt))
                        continue
                    raise

        # 定义Embedding函数
        async def embedding_func(texts: List[str]):
            import numpy as np
            embeddings = await self._generate_embeddings(texts)
            return np.array(embeddings, dtype=np.float32)

        # 确保事件循环
        loop = self._ensure_event_loop()
        asyncio.set_event_loop(loop)

        # 创建LightRAG实例
        try:
            self.rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.embedding_dim,
                    max_token_size=8192,
                    func=embedding_func
                ),
                encoding_model="cl100k_base",
                top_k=self.query_top_k,
                chunk_top_k=self.query_chunk_top_k,
                max_entity_tokens=self.query_max_entity_tokens,
                max_relation_tokens=self.query_max_relation_tokens,
                max_total_tokens=self.query_max_total_tokens,
                cosine_threshold=self.query_cosine_threshold,
                related_chunk_number=self.query_related_chunk_number,
            )
        except TypeError:
            # 如果encoding_model参数不支持
            self.rag = LightRAG(
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

        # 初始化存储
        async def init_storages():
            await self.rag.initialize_storages()
            try:
                from lightrag.kg.shared_storage import initialize_pipeline_status
                await initialize_pipeline_status()
            except ImportError:
                pass

        self._run_async(init_storages())

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成Embeddings"""
        import aiohttp

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
                            json={"model": self.embedding_model, "prompt": text}
                        )
                        if response.status == 200:
                            data = await response.json()
                            vector = data.get("embedding")
                            if not vector:
                                raise ValueError("响应中缺少 'embedding' 字段")
                            embeddings.append(self._normalize_embedding(vector))
                            success = True
                            break
                        else:
                            error_text = await response.text()
                            logger.warning(f"Embedding错误 {response.status}: {error_text}")
                    except asyncio.TimeoutError:
                        logger.warning(f"Embedding超时 (尝试 {attempt})")
                    except Exception as e:
                        logger.warning(f"Embedding失败: {e}")

                    if attempt < self.embedding_max_retries:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 10)

                if not success:
                    # 使用回退embeddings
                    embeddings.append(self._fallback_embedding(text))

        return embeddings

    def _fallback_embedding(self, text: str) -> List[float]:
        """生成回退embeddings"""
        import random
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.embedding_dim)]

    def _normalize_embedding(self, embedding: Any) -> List[float]:
        """规范化embedding维度"""
        if embedding is None:
            return [0.0] * self.embedding_dim

        vector = list(embedding)
        if len(vector) == self.embedding_dim:
            return vector
        if len(vector) > self.embedding_dim:
            return vector[:self.embedding_dim]
        return vector + [0.0] * (self.embedding_dim - len(vector))

    def _clean_think_tags(self, text: str) -> str:
        """清理DeepSeek R1的<think>标签"""
        if not text:
            return ""
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        return cleaned.strip()

    def insert_text(self, text: str, doc_id: Optional[str] = None, file_path: Optional[str] = None):
        """
        插入文本到知识库
        
        Args:
            text: 文本内容
            doc_id: 文档ID
            file_path: 文件路径
        """
        logger.info(f"插入文档: {doc_id or 'unknown'}")

        async def _insert():
            await self.rag.initialize_storages()
            try:
                from lightrag.kg.shared_storage import initialize_pipeline_status
                await initialize_pipeline_status()
            except ImportError:
                pass

            await self.rag.ainsert(
                text,
                ids=doc_id,
                file_paths=file_path or doc_id or "unknown"
            )

        self._run_async(_insert())
        logger.info(f"文档插入完成: {doc_id or 'unknown'}")

    def insert_directory(self, directory_path: Path, recursive: bool = True):
        """
        批量插入目录中的文本文件
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归搜索
        """
        pattern = "**/*.txt" if recursive else "*.txt"
        text_files = sorted(directory_path.glob(pattern))

        if not text_files:
            logger.warning(f"目录中未找到文本文件: {directory_path}")
            return

        logger.info(f"发现 {len(text_files)} 个文本文件")

        for i, text_file in enumerate(tqdm(text_files, desc="插入文档")):
            try:
                content = text_file.read_text(encoding="utf-8")
                doc_id = text_file.stem
                self.insert_text(content, doc_id=doc_id, file_path=str(text_file))
            except Exception as e:
                logger.error(f"插入文档失败 {text_file}: {e}")
                continue

        logger.info(f"批量插入完成")

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        stats = {
            "working_directory": str(self.working_dir),
            "directory_exists": self.working_dir.exists(),
            "directory_size_mb": 0,
        }

        if self.working_dir.exists():
            files = list(self.working_dir.glob("*.json")) + list(self.working_dir.glob("*.graphml"))
            stats["file_count"] = len(files)
            stats["files"] = [f.name for f in files]

            total_size = sum(f.stat().st_size for f in files if f.is_file())
            stats["directory_size_mb"] = total_size / (1024 * 1024)

        return stats

    def get_chunk_context(
        self,
        chunk_id: str,
        max_entities: int = 3,
        max_relations: int = 2,
        max_snippets: int = 2,
        snippet_chars: int = 200,
    ) -> Dict[str, Any]:
        """
        获取chunk相关的知识图谱上下文
        
        Args:
            chunk_id: Chunk ID
            max_entities: 最大实体数
            max_relations: 最大关系数
            max_snippets: 最大片段数
            snippet_chars: 片段字符数
            
        Returns:
            上下文信息字典
        """
        if not chunk_id or not self.rag:
            return {"prompt_context": "", "related_entities": [], "related_chunk_ids": []}

        graph = getattr(self.rag, "chunk_entity_relation_graph", None)
        text_chunks = getattr(self.rag, "text_chunks", None)
        
        if not graph:
            return {"prompt_context": "", "related_entities": [], "related_chunk_ids": []}

        # 获取节点和边
        try:
            nodes = self._run_async(graph.get_nodes_by_chunk_ids([chunk_id]), timeout=5.0) or []
        except Exception:
            nodes = []

        try:
            edges = self._run_async(graph.get_edges_by_chunk_ids([chunk_id]), timeout=5.0) or []
        except Exception:
            edges = []

        entity_lines: List[str] = []
        relation_lines: List[str] = []
        snippet_lines: List[str] = []
        related_entities: List[str] = []
        related_chunk_ids: List[str] = []

        # 处理实体节点
        for node in nodes:
            if len(entity_lines) >= max_entities:
                break

            name = node.get("entity_id") or node.get("entity_name") or node.get("id")
            if not name or name in related_entities:
                continue

            entity_type = node.get("entity_type")
            description = node.get("description")
            info = f"- 实体 {name}"
            if entity_type:
                info += f"（类型：{entity_type}）"
            if description:
                info += f"：{description}"
            else:
                info += "：暂无详细描述"

            entity_lines.append(info)
            related_entities.append(name)

            # 收集相关chunk_ids
            source_id = node.get("source_id")
            if source_id:
                for cid in self._parse_source_ids(source_id):
                    if cid and cid not in related_chunk_ids and cid != chunk_id:
                        related_chunk_ids.append(cid)

        # 处理关系边
        for edge in edges:
            if len(relation_lines) >= max_relations:
                break

            if "src_tgt" in edge:
                entity1, entity2 = edge.get("src_tgt", (None, None))
            else:
                entity1 = edge.get("src_id")
                entity2 = edge.get("tgt_id")

            if not entity1 or not entity2:
                continue

            description = edge.get("description")
            keywords = edge.get("keywords")

            info = f"- {entity1} ↔ {entity2}"
            if description:
                info += f"：{description}"
            elif keywords:
                info += f"：关键词 {keywords}"
            else:
                info += "：暂无详细描述"

            relation_lines.append(info)

            for candidate in (entity1, entity2):
                if candidate and candidate not in related_entities:
                    related_entities.append(candidate)

        # 获取相关chunk片段
        if text_chunks and related_chunk_ids:
            for cid in related_chunk_ids[:max_snippets]:
                try:
                    data = self._run_async(text_chunks.get_by_id(cid), timeout=5.0)
                except Exception:
                    data = None

                if not data:
                    continue

                content = data.get("content", "")
                if content:
                    snippet = content.strip().replace("\n", " ")[:snippet_chars]
                    snippet_lines.append(f"- 片段 {cid}: {snippet}")

        # 构建上下文
        context_sections: List[str] = []
        if entity_lines:
            context_sections.append("【相关实体信息】\n" + "\n".join(entity_lines))
        if relation_lines:
            context_sections.append("【相关关系信息】\n" + "\n".join(relation_lines))
        if snippet_lines:
            context_sections.append("【知识图谱片段】\n" + "\n".join(snippet_lines))

        prompt_context = "\n\n".join(context_sections).strip()

        # 确保主chunk_id在列表首位
        ordered_chunk_ids = [chunk_id]
        for cid in related_chunk_ids:
            if cid not in ordered_chunk_ids:
                ordered_chunk_ids.append(cid)

        return {
            "prompt_context": prompt_context,
            "related_entities": related_entities,
            "related_chunk_ids": ordered_chunk_ids,
        }

    def _parse_source_ids(self, raw_source: Any) -> List[str]:
        """解析source_id字段"""
        if not raw_source:
            return []

        chunk_ids: List[str] = []
        separators = [";", ",", "|"]

        if isinstance(raw_source, str):
            pattern = "|".join(re.escape(sep) for sep in separators)
            parts = re.split(pattern, raw_source) if pattern else [raw_source]
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    chunk_ids.append(cleaned)
        elif isinstance(raw_source, (list, tuple, set)):
            for item in raw_source:
                if isinstance(item, str) and item.strip():
                    chunk_ids.append(item.strip())

        return chunk_ids

