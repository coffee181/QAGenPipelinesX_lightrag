"""LLM客户端模块 - 封装Ollama/DeepSeek交互与问题生成"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import requests
from loguru import logger

if TYPE_CHECKING:
    from .rag_core import RAGCore


@dataclass
class DocumentChunk:
    """文档分块"""
    document_id: str
    chunk_id: str
    content: str
    start_position: int = 0
    end_position: int = 0
    chunk_index: int = 0
    total_chunks: int = 0

    @property
    def length(self) -> int:
        return len(self.content)


@dataclass
class Question:
    """问题数据结构"""
    question_id: str
    content: str
    source_document: str
    source_chunk_id: str
    question_index: int
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_chunk_content: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)


@dataclass
class QuestionSet:
    """问题集"""
    document_id: str
    questions: List[Question]
    created_at: datetime

    @property
    def total_questions(self) -> int:
        return len(self.questions)

    def to_jsonl(self) -> List[Dict[str, Any]]:
        """转换为JSONL格式"""
        return [
            {
                "question_id": q.question_id,
                "content": q.content,
                "source_document": q.source_document,
                "source_chunk_id": q.source_chunk_id,
                "question_index": q.question_index,
                "metadata": q.metadata,
                "created_at": q.created_at.isoformat() if q.created_at else None,
            }
            for q in self.questions
        ]


class LLMClient:
    """
    LLM客户端 - 封装Ollama交互和问题生成
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-r1:32b",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 30000,
        questions_per_chunk: int = 10,
        system_prompt: str = "",
        human_prompt: str = "",
        rag: Optional["RAGCore"] = None,
        kg_context_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化LLM客户端
        
        Args:
            base_url: Ollama服务地址
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            timeout: 超时时间(秒)
            questions_per_chunk: 每个chunk生成的问题数
            system_prompt: 系统提示词
            human_prompt: 用户提示词模板
            rag: RAG核心实例（用于获取知识图谱上下文）
            kg_context_config: 知识图谱上下文配置
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.questions_per_chunk = questions_per_chunk
        self.system_prompt = system_prompt
        self.human_prompt = human_prompt
        self.rag = rag

        # 知识图谱上下文配置
        kg_cfg = kg_context_config or {}
        self.kg_enabled = kg_cfg.get("enabled", True)
        self.kg_max_entities = kg_cfg.get("max_entities", 5)
        self.kg_max_relations = kg_cfg.get("max_relations", 5)
        self.kg_max_snippets = kg_cfg.get("max_snippets", 2)
        self.kg_snippet_chars = kg_cfg.get("snippet_chars", 200)

        # 测试连接
        if not self._test_connection():
            logger.warning(f"无法连接到Ollama服务: {self.base_url}")

        logger.info(f"LLM客户端初始化完成 - 模型: {self.model}")

    def _test_connection(self) -> bool:
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）
            
        Returns:
            生成的文本
        """
        effective_system = system_prompt or self.system_prompt

        payload = {
            "model": self.model,
            "prompt": f"{effective_system}\n\n{prompt}" if effective_system else prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                raw_text = result.get("response", "")
                return self._clean_think_tags(raw_text)
            else:
                raise RuntimeError(f"Ollama API错误: {response.status_code} - {response.text}")

        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama API超时 (超过 {self.timeout} 秒)")
        except Exception as e:
            raise RuntimeError(f"Ollama API异常: {e}")

    def _clean_think_tags(self, text: str) -> str:
        """清理DeepSeek R1的<think>标签"""
        if not text:
            return ""
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        return cleaned.strip()

    def generate_questions_from_chunk(self, chunk: DocumentChunk) -> List[Question]:
        """
        从单个chunk生成问题
        
        Args:
            chunk: 文档分块
            
        Returns:
            问题列表
        """
        logger.info(f"🔍 开始为块生成问题: {chunk.chunk_id}")
        logger.info(f"📄 文本块长度: {len(chunk.content)} 字符")

        # 构建知识图谱上下文
        context_package = self._build_context(chunk)
        prompt_context = context_package.get("prompt_context", "")

        # 准备提示词
        human_message = self.human_prompt.format(
            text=chunk.content.strip(),
            prompt_context=prompt_context,
            questions_per_chunk=self.questions_per_chunk,
            document_id=chunk.document_id,
        )

        logger.info(f"📝 提示词长度: {len(human_message)} 字符")
        logger.info(f"🤖 调用本地模型: {self.model}")

        # 调用LLM
        response = self.generate(human_message, self.system_prompt)

        logger.info(f"✅ 收到响应: {len(response)} 字符")

        # 解析问题
        questions = self._parse_questions(response, chunk, context_package)

        logger.info(f"🎉 成功生成 {len(questions)} 个问题")
        return questions

    def generate_questions_from_chunks(self, chunks: List[DocumentChunk]) -> QuestionSet:
        """
        从多个chunk生成问题
        
        Args:
            chunks: 文档分块列表
            
        Returns:
            问题集
        """
        if not chunks:
            raise ValueError("没有提供用于问题生成的块")

        document_id = chunks[0].document_id
        logger.info(f"为文档 {document_id} 的 {len(chunks)} 个块生成问题")

        all_questions = []
        for chunk in chunks:
            try:
                questions = self.generate_questions_from_chunk(chunk)
                all_questions.extend(questions)
            except Exception as e:
                logger.error(f"为块 {chunk.chunk_id} 生成问题失败: {e}")
                continue

        question_set = QuestionSet(
            document_id=document_id,
            questions=all_questions,
            created_at=datetime.now()
        )

        logger.info(f"为文档 {document_id} 总共生成了 {len(all_questions)} 个问题")
        return question_set

    def _build_context(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """构建知识图谱上下文"""
        if not self.kg_enabled or not self.rag:
            return {"prompt_context": "", "related_entities": [], "related_chunk_ids": []}

        # 计算chunk_id
        from .rag_core import compute_chunk_id
        chunk_id = compute_chunk_id(chunk.content)
        if not chunk_id:
            return {"prompt_context": "", "related_entities": [], "related_chunk_ids": []}

        try:
            context = self.rag.get_chunk_context(
                chunk_id,
                max_entities=self.kg_max_entities,
                max_relations=self.kg_max_relations,
                max_snippets=self.kg_max_snippets,
                snippet_chars=self.kg_snippet_chars,
            )
            return context
        except Exception as e:
            logger.debug(f"构建知识图谱上下文失败: {e}")
            return {"prompt_context": "", "related_entities": [], "related_chunk_ids": []}

    def _parse_questions(
        self,
        response: str,
        source_chunk: DocumentChunk,
        context_package: Dict[str, Any],
    ) -> List[Question]:
        """从LLM响应中解析问题"""
        cleaned_response = self._clean_think_tags(response)
        questions: List[Question] = []

        base_entities = list(dict.fromkeys(context_package.get("related_entities", [])))
        base_chunk_ids = list(dict.fromkeys(context_package.get("related_chunk_ids", [])))
        
        # 计算主chunk_id
        from .rag_core import compute_chunk_id
        primary_chunk_id = compute_chunk_id(source_chunk.content)
        
        if primary_chunk_id and primary_chunk_id not in base_chunk_ids:
            base_chunk_ids = [primary_chunk_id] + base_chunk_ids

        knowledge_used = bool(context_package.get("prompt_context"))

        # 解析"问题N:"格式
        question_pattern = r"问题(\d+)[:：]\s*(.+?)(?=\n\s*问题\d+[:：]|$)"
        matches = re.findall(question_pattern, cleaned_response, re.DOTALL)

        if matches:
            logger.info(f"✅ 找到 {len(matches)} 个问题")
            for match in matches:
                question_num = int(match[0])
                question_content = match[1].strip()
                question_content = re.sub(r"^问题[:：]\s*", "", question_content)
                question_content = re.sub(r"\n+", " ", question_content).strip()
                question_content = self._strip_answer(question_content)

                if self._is_valid_question(question_content):
                    question = self._build_question(
                        question_content, source_chunk, question_num,
                        base_entities, base_chunk_ids, primary_chunk_id, knowledge_used
                    )
                    questions.append(question)

        # 尝试备用解析
        if not questions:
            logger.info("⚠️ 尝试备用解析...")
            questions = self._fallback_parse(
                cleaned_response, source_chunk, context_package
            )

        return questions

    def _strip_answer(self, text: str) -> str:
        """移除答案部分"""
        if not text:
            return ""
        cleaned = re.sub(r"(答案[:：].*)", "", text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"(回答[:：].*)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"(Answer[:：].*)", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip(" \t\r\n。；;，,")

    def _is_valid_question(self, content: str) -> bool:
        """验证问题是否有效"""
        if not content or len(content) < 15:
            return False
        if not ("？" in content or "?" in content):
            return False
        if re.match(r"^#+\s", content):
            return False
        if re.match(r"^(复杂|中等|简单|关联|深度|事实).*问题", content):
            return False
        if content.startswith("【"):
            return False
        if any(bad in content for bad in ["答案", "解答", "Answer"]):
            return False
        return True

    def _build_question(
        self,
        content: str,
        source_chunk: DocumentChunk,
        question_index: int,
        base_entities: List[str],
        base_chunk_ids: List[str],
        primary_chunk_id: Optional[str],
        knowledge_used: bool,
    ) -> Question:
        """构建Question对象"""
        # 从问题中提取实体
        candidate_entities = self._extract_entities(content)
        combined_entities = list(dict.fromkeys(base_entities + candidate_entities))

        metadata: Dict[str, Any] = {"has_answer": False}
        if primary_chunk_id:
            metadata["lightrag_chunk_id"] = primary_chunk_id
        if combined_entities:
            metadata["related_entities"] = combined_entities
        if base_chunk_ids:
            metadata["related_chunk_ids"] = list(dict.fromkeys(base_chunk_ids))
        if knowledge_used:
            metadata["knowledge_context_used"] = True

        return Question(
            question_id=str(uuid.uuid4()),
            content=content,
            source_document=source_chunk.document_id,
            source_chunk_id=source_chunk.chunk_id,
            question_index=question_index,
            created_at=datetime.now(),
            metadata=metadata,
            source_chunk_content=source_chunk.content,
            related_entities=combined_entities,
        )

    def _extract_entities(self, text: str) -> List[str]:
        """从文本中提取实体"""
        if not text:
            return []

        candidates: List[str] = []
        patterns = [
            r"[A-Z]{2,}\d+[A-Z]*",
            r"[A-Z]+\d+[A-Z0-9]*",
            r"[A-Z][A-Za-z0-9\-]{2,}",
        ]

        matched_tokens = set()
        for pattern in patterns:
            matched_tokens.update(re.findall(pattern, text))

        for token in re.findall(r"\b[^\s]+\b", text):
            normalized = token.strip(".,;:!?，。；：（）()[]{}""\"'")
            if normalized in matched_tokens and normalized not in candidates:
                candidates.append(normalized)

        return candidates

    def _fallback_parse(
        self,
        response: str,
        source_chunk: DocumentChunk,
        context_package: Dict[str, Any],
    ) -> List[Question]:
        """备用问题解析"""
        questions = []
        
        from .rag_core import compute_chunk_id
        
        base_entities = list(dict.fromkeys(context_package.get("related_entities", [])))
        base_chunk_ids = list(dict.fromkeys(context_package.get("related_chunk_ids", [])))
        primary_chunk_id = compute_chunk_id(source_chunk.content)
        
        if primary_chunk_id and primary_chunk_id not in base_chunk_ids:
            base_chunk_ids = [primary_chunk_id] + base_chunk_ids
        knowledge_used = bool(context_package.get("prompt_context"))

        lines = response.split('\n')
        question_index = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#') or line.startswith('【'):
                continue
            if re.match(r'^(复杂|中等|简单|关联|深度|事实).*问题', line):
                continue

            if '?' in line or '？' in line or line.startswith(('如何', '什么', '为什么', '怎样', '哪些', '是否')):
                cleaned = re.sub(r'^[\d\.\-\*\s]+', '', line)
                cleaned = re.sub(r'^问题\d+[:\：]\s*', '', cleaned)
                cleaned = self._strip_answer(cleaned)

                if self._is_valid_question(cleaned):
                    question = self._build_question(
                        cleaned, source_chunk, question_index,
                        base_entities, base_chunk_ids, primary_chunk_id, knowledge_used
                    )
                    questions.append(question)
                    question_index += 1

                    if len(questions) >= self.questions_per_chunk:
                        break

        return questions


class TextChunker:
    """文本分块器"""

    def __init__(
        self,
        use_token_chunking: bool = True,
        tokenizer_model: str = "cl100k_base",
        chunk_token_size: int = 1200,
        chunk_overlap_token_size: int = 100,
        max_chunk_size: int = 60000,
        overlap_size: int = 3000,
    ):
        """
        初始化文本分块器
        
        Args:
            use_token_chunking: 是否使用token级分块
            tokenizer_model: tokenizer模型
            chunk_token_size: 每个chunk的token数
            chunk_overlap_token_size: 重叠token数
            max_chunk_size: 最大字符数（字符级分块）
            overlap_size: 重叠字符数（字符级分块）
        """
        self.use_token_chunking = use_token_chunking
        self.tokenizer_model = tokenizer_model
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap_token_size = chunk_overlap_token_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

        self.tokenizer = None
        if use_token_chunking:
            try:
                import tiktoken
                self.tokenizer = tiktoken.get_encoding(tokenizer_model)
                logger.info(f"Token分块器初始化完成: {tokenizer_model}")
            except Exception as e:
                logger.warning(f"Tokenizer初始化失败: {e}，将使用字符级分块")
                self.use_token_chunking = False

    def chunk_text(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        将文本分块
        
        Args:
            text: 文本内容
            document_id: 文档ID
            
        Returns:
            分块列表
        """
        if not text.strip():
            return []

        if self.use_token_chunking and self.tokenizer:
            return self._chunk_by_tokens(text, document_id)
        else:
            return self._chunk_by_chars(text, document_id)

    def _chunk_by_tokens(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Token级分块"""
        try:
            from lightrag.operate import chunking_by_token_size
            
            chunk_dicts = chunking_by_token_size(
                tokenizer=self.tokenizer,
                content=text,
                split_by_character=None,
                split_by_character_only=False,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size
            )
        except ImportError:
            logger.warning("LightRAG chunking不可用，使用字符级分块")
            return self._chunk_by_chars(text, document_id)

        chunks = []
        current_pos = 0

        for idx, chunk_dict in enumerate(chunk_dicts):
            content = chunk_dict.get("content", "").strip()
            if not content:
                continue

            from .rag_core import compute_chunk_id
            chunk_id = compute_chunk_id(content) or f"{document_id}_chunk_{idx}"

            start_pos = text.find(content, current_pos)
            end_pos = start_pos + len(content)
            current_pos = start_pos

            chunk = DocumentChunk(
                document_id=document_id,
                chunk_id=chunk_id,
                content=content,
                start_position=start_pos,
                end_position=end_pos,
                chunk_index=chunk_dict.get("chunk_order_index", idx),
                total_chunks=len(chunk_dicts)
            )
            chunks.append(chunk)

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        logger.info(f"Token分块完成: {len(chunks)} 个chunks")
        return chunks

    def _chunk_by_chars(self, text: str, document_id: str) -> List[DocumentChunk]:
        """字符级分块"""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            content = text[start:end]

            from .rag_core import compute_chunk_id
            chunk_id = compute_chunk_id(content) or f"{document_id}_chunk_{len(chunks)}"

            chunk = DocumentChunk(
                document_id=document_id,
                chunk_id=chunk_id,
                content=content,
                start_position=start,
                end_position=end,
                chunk_index=len(chunks),
                total_chunks=0
            )
            chunks.append(chunk)

            start = end - self.overlap_size if end < len(text) else end

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        logger.info(f"字符分块完成: {len(chunks)} 个chunks")
        return chunks

