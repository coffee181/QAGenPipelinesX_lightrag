"""本地模型问题生成器实现 - 支持Ollama"""

import re
import uuid
import requests
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
from datetime import datetime
from loguru import logger

from ..interfaces.question_generator_interface import QuestionGeneratorInterface, QuestionGenerationError
from ..models.document import DocumentChunk
from ..models.question import Question, QuestionSet
from ..utils.config import ConfigManager
from ..utils.lightrag_utils import compute_lightrag_chunk_id, LightRAGContextBuilder

if TYPE_CHECKING:
    from .lightrag_rag import LightRAGImplementation

# 导入超时配置
try:
    from ...timeout_config import configure_global_timeouts, configure_ollama_timeout
except ImportError:
    # 如果导入失败，使用默认配置
    def configure_global_timeouts():
        return requests.Session()
    
    def configure_ollama_timeout():
        return {'timeout': (60, 30000)}


class LocalQuestionGenerator(QuestionGeneratorInterface):
    """基于Ollama的本地问题生成器实现"""

    def __init__(
        self,
        config: ConfigManager,
        rag: Optional["LightRAGImplementation"] = None,
    ):
        """
        初始化本地模型问题生成器

        Args:
            config: 配置对象
        """
        self.config = config
        self.rag = rag
        
        # Ollama配置
        self.model_name = config.get("question_generator.local.model_name", "deepseek-r1:32b")
        self.base_url = config.get("question_generator.local.base_url", "http://localhost:11434")
        self.max_tokens = config.get("question_generator.local.max_tokens", 2048)
        self.temperature = config.get("question_generator.local.temperature", 0.7)
        self.timeout = config.get("question_generator.local.timeout", 120)
        self.questions_per_chunk = config.get("question_generator.local.questions_per_chunk", 10)

        # 知识图谱上下文配置
        self.enable_kg_context = config.get(
            "question_generator.local.enable_kg_context", True
        )
        self.max_context_entities = config.get(
            "question_generator.local.max_context_entities", 3
        )
        self.max_context_relations = config.get(
            "question_generator.local.max_context_relations", 2
        )
        self.max_context_snippets = config.get(
            "question_generator.local.max_context_snippets", 2
        )
        self.context_snippet_chars = config.get(
            "question_generator.local.context_snippet_chars", 200
        )
        self.max_related_chunk_ids = config.get(
            "question_generator.local.max_related_chunk_ids", 6
        )

        if not rag or not getattr(rag, "rag", None):
            self.enable_kg_context = False

        self.context_builder: Optional[LightRAGContextBuilder] = None
        if self.enable_kg_context:
            self.context_builder = LightRAGContextBuilder(
                rag,
                max_entities=self.max_context_entities,
                max_relations=self.max_context_relations,
                max_snippets=self.max_context_snippets,
                snippet_chars=self.context_snippet_chars,
                max_related_chunk_ids=self.max_related_chunk_ids,
            )

        # 加载提示词
        self.system_prompt = config.get("prompts.system_prompt", "")
        self.human_prompt = config.get("prompts.human_prompt", "")

        # 测试连接
        if not self._test_connection():
            raise QuestionGenerationError(f"无法连接到Ollama服务: {self.base_url}")

        # 配置全局超时设置
        self.session = configure_global_timeouts()
        self.ollama_config = configure_ollama_timeout()
        
        logger.info(f"本地模型问题生成器初始化完成 - 模型: {self.model_name}")

    def _test_connection(self) -> bool:
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False

    def generate_questions_from_chunk(self, chunk: DocumentChunk) -> List[Question]:
        """
        从单个文本块生成问题

        Args:
            chunk: 要生成问题的DocumentChunk

        Returns:
            Question对象列表
        """
        try:
            logger.info(f"🔍 开始为块生成问题: {chunk.chunk_id}")
            logger.info(f"📄 文本块长度: {len(chunk.content)} 字符")
            logger.info(f"🎯 目标问题数量: {self.questions_per_chunk}")

            context_package = self._build_context_for_chunk(chunk)
            prompt_text = self._compose_prompt_text(chunk.content)
            prompt_context = context_package.get("prompt_context", "")

            # 准备提示词
            human_message = self.human_prompt.format(
                text=prompt_text,
                prompt_context=prompt_context,
                questions_per_chunk=self.questions_per_chunk
            )
            
            logger.info(f"📝 提示词长度: {len(human_message)} 字符")
            logger.info(f"🤖 调用本地模型: {self.model_name}")

            # 调用Ollama API
            response_content = self._call_ollama_api(human_message)

            logger.info(f"✅ 收到本地模型响应: {len(response_content)} 字符")
            logger.info(f"📋 原始响应预览: {response_content[:200]}...")

            # 解析问题
            questions = self.parse_questions_from_response(
                response_content, chunk, context_package
            )

            logger.info(f"🎉 成功为块 {chunk.chunk_id} 生成了 {len(questions)} 个问题")
            
            # 显示生成的问题
            for i, question in enumerate(questions, 1):
                logger.info(f"  问题{i}: {question.content[:100]}{'...' if len(question.content) > 100 else ''}")
            
            return questions

        except Exception as e:
            raise QuestionGenerationError(f"为块 {chunk.chunk_id} 生成问题失败: {e}")

    def _extract_candidate_entities(self, text: str) -> List[str]:
        """从问题文本中提取可能的实体名称或型号。"""
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

        # 按出现顺序去重
        for token in re.findall(r"\b[^\s]+\b", text):
            normalized = token.strip(".,;:!?，。；：（）()[]{}“”\"'")
            if normalized in matched_tokens and normalized not in candidates:
                candidates.append(normalized)

        return candidates

    def _empty_context_package(self) -> Dict[str, Any]:
        return {
            "prompt_context": "",
            "related_entities": [],
            "related_chunk_ids": [],
        }

    def _build_context_for_chunk(self, chunk: DocumentChunk) -> Dict[str, Any]:
        if not self.context_builder:
            return self._empty_context_package()

        chunk_id = compute_lightrag_chunk_id(chunk.content)
        if not chunk_id:
            return self._empty_context_package()

        try:
            context = self.context_builder.build_context(chunk_id)
        except Exception as e:
            logger.debug(f"构建知识图谱上下文失败（chunk: {chunk.chunk_id}）: {e}")
            context = LightRAGContextBuilder._empty_context()

        if not context:
            return self._empty_context_package()

        return {
            "prompt_context": context.get("prompt_context", ""),
            "related_entities": context.get("related_entities", []) or [],
            "related_chunk_ids": context.get("related_chunk_ids", []) or [],
        }

    def _compose_prompt_text(self, chunk_text: str) -> str:
        return (chunk_text or "").strip()

    def _build_question_object(
        self,
        question_content: str,
        source_chunk: DocumentChunk,
        question_index: int,
        base_related_entities: Sequence[str],
        base_related_chunk_ids: Sequence[str],
        primary_chunk_id: Optional[str],
        knowledge_used: bool,
    ) -> Question:
        candidate_entities = self._extract_candidate_entities(question_content)
        combined_entities = list(
            dict.fromkeys(list(base_related_entities) + candidate_entities)
        )

        metadata: Dict[str, Any] = {"has_answer": False}
        if primary_chunk_id:
            metadata["lightrag_chunk_id"] = primary_chunk_id
        if combined_entities:
            metadata["related_entities"] = combined_entities
        if base_related_chunk_ids:
            metadata["related_chunk_ids"] = list(
                dict.fromkeys(base_related_chunk_ids)
            )
        if knowledge_used:
            metadata["knowledge_context_used"] = True

        question = Question(
            question_id=str(uuid.uuid4()),
            content=question_content,
            source_document=source_chunk.document_id,
            source_chunk_id=source_chunk.chunk_id,
            question_index=question_index,
            created_at=datetime.now(),
            metadata=metadata,
            source_chunk_content=source_chunk.content,
            related_entities=combined_entities,
        )
        return question

    def _call_ollama_api(self, prompt: str) -> str:
        """调用Ollama API"""
        try:
            # 设置全局requests超时
            import os
            import time
            os.environ['REQUESTS_TIMEOUT'] = str(self.timeout)
            
            payload = {
                "model": self.model_name,
                "prompt": f"{self.system_prompt}\n\n{prompt}",
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            logger.info(f"🚀 发送请求到: {self.base_url}/api/generate")
            logger.info(f"⏱️  超时设置: {self.timeout} 秒")
            logger.info(f"🌡️  温度参数: {self.temperature}")
            logger.info(f"📊 最大token数: {self.max_tokens}")
            
            start_time = time.time()
            
            # 使用配置好的session和超时设置
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                **self.ollama_config
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"⏰ API调用耗时: {duration:.2f} 秒")
            logger.info(f"📡 响应状态码: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                logger.info(f"✅ API调用成功，响应长度: {len(response_text)} 字符")
                return response_text
            else:
                logger.error(f"❌ API调用失败: {response.status_code}")
                logger.error(f"📄 错误响应: {response.text}")
                raise QuestionGenerationError(f"Ollama API调用失败: {response.status_code} - {response.text}")

        except requests.exceptions.Timeout:
            logger.error(f"⏰ API调用超时 (超过 {self.timeout} 秒)")
            raise QuestionGenerationError("Ollama API调用超时")
        except Exception as e:
            logger.error(f"💥 API调用异常: {e}")
            raise QuestionGenerationError(f"Ollama API调用异常: {e}")

    def generate_questions_from_chunks(self, chunks: List[DocumentChunk]) -> QuestionSet:
        """
        从多个文本块生成问题

        Args:
            chunks: DocumentChunk对象列表

        Returns:
            包含所有生成问题的QuestionSet
        """
        try:
            if not chunks:
                raise QuestionGenerationError("没有提供用于问题生成的块")

            document_id = chunks[0].document_id
            logger.info(f"为文档 {document_id} 的 {len(chunks)} 个块生成问题")

            all_questions = []

            for chunk in chunks:
                try:
                    chunk_questions = self.generate_questions_from_chunk(chunk)
                    all_questions.extend(chunk_questions)
                except Exception as e:
                    logger.error(f"为块 {chunk.chunk_id} 生成问题失败: {e}")
                    continue

            # 创建QuestionSet
            question_set = QuestionSet(
                document_id=document_id,
                questions=all_questions,
                created_at=datetime.now()
            )

            logger.info(f"为文档 {document_id} 总共生成了 {len(all_questions)} 个问题")
            return question_set

        except Exception as e:
            raise QuestionGenerationError(f"从块生成问题失败: {e}")

    def parse_questions_from_response(
        self,
        response: str,
        source_chunk: DocumentChunk,
        context_package: Dict[str, Any],
    ) -> List[Question]:
        """
        从LLM响应中解析问题（只解析问题，不解析答案）
        """
        try:
            cleaned_response = self._clean_think_tags(response)
            questions: List[Question] = []

            base_related_entities = list(
                dict.fromkeys(context_package.get("related_entities", []))
            )
            base_related_chunk_ids = list(
                dict.fromkeys(context_package.get("related_chunk_ids", []))
            )
            primary_chunk_id = compute_lightrag_chunk_id(source_chunk.content)
            if primary_chunk_id and primary_chunk_id not in base_related_chunk_ids:
                base_related_chunk_ids = [primary_chunk_id] + base_related_chunk_ids
            knowledge_used = bool(context_package.get("prompt_context"))

            question_pattern = r"问题(\d+)[:：]\s*(.+?)(?=\n\s*问题\d+[:：]|$)"
            question_matches = re.findall(question_pattern, cleaned_response, re.DOTALL)

            if question_matches:
                logger.info(f"✅ 找到新格式问题候选: {len(question_matches)} 个")
                for match in question_matches:
                    question_num = int(match[0])
                    question_content = match[1].strip()
                    question_content = re.sub(r"^问题[:：]\s*", "", question_content)
                    question_content = re.sub(r"\n+", " ", question_content).strip()

                    is_valid = (
                        question_content
                        and len(question_content) > 15
                        and ("？" in question_content or "?" in question_content)
                        and not re.match(r"^#+\s", question_content)
                        and not re.match(r"^(复杂|中等|简单|关联|深度|事实).*问题", question_content)
                        and not question_content.startswith("【")
                    )

                    if is_valid:
                        question = self._build_question_object(
                            question_content=question_content,
                            source_chunk=source_chunk,
                            question_index=question_num,
                            base_related_entities=base_related_entities,
                            base_related_chunk_ids=base_related_chunk_ids,
                            primary_chunk_id=primary_chunk_id,
                            knowledge_used=knowledge_used,
                        )
                        questions.append(question)
                        logger.debug(f"✅ 有效问题 {question_num}: {question_content[:50]}...")
                    else:
                        logger.warning(f"⚠️ 跳过无效内容 {question_num}: {question_content[:50]}...")

            if not questions:
                logger.info("⚠️ 未找到新格式问题，尝试兼容旧格式（问答对格式）...")
                qa_pair_pattern = r"问答对(\d+)[:：]\s*\n\s*问题[:：]\s*(.+?)(?:\s*\n\s*答案[:：]|(?=\n\s*问答对\d+|$))"
                qa_matches = re.findall(qa_pair_pattern, cleaned_response, re.DOTALL)

                if qa_matches:
                    logger.info(f"✅ 找到旧格式问答对（仅提取问题）: {len(qa_matches)} 个")
                    for match in qa_matches:
                        qa_num = int(match[0])
                        question_content = re.sub(r"^问题[:：]\s*", "", match[1]).strip()

                        if question_content:
                            question = self._build_question_object(
                                question_content=question_content,
                                source_chunk=source_chunk,
                                question_index=qa_num,
                                base_related_entities=base_related_entities,
                                base_related_chunk_ids=base_related_chunk_ids,
                                primary_chunk_id=primary_chunk_id,
                                knowledge_used=knowledge_used,
                            )
                            questions.append(question)
                            logger.debug(f"问题 {qa_num}: {question_content[:50]}...")

            if not questions:
                logger.error("❌ 所有格式都未匹配，尝试fallback提取...")
                questions = self._extract_fallback_questions(
                    cleaned_response,
                    source_chunk,
                    context_package,
                    start_index=1,
                )

            logger.info(f"从响应中解析出 {len(questions)} 个问题")
            return questions

        except Exception as e:
            raise QuestionGenerationError(f"从响应解析问题失败: {e}")

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
        
        # 移除<think>标签及其内容
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 清理多余的空行
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        
        return cleaned_text.strip()

    def validate_questions(self, questions: List[Question]) -> bool:
        """
        验证生成的问题

        Args:
            questions: 要验证的问题列表

        Returns:
            如果所有问题都有效则返回True，否则返回False
        """
        if not questions:
            return False

        for question in questions:
            # 检查问题是否有内容
            if not question.content or not question.content.strip():
                logger.error(f"问题 {question.question_id} 没有内容")
                return False

            # 检查问题是否太短
            if len(question.content.strip()) < 10:
                logger.error(f"问题 {question.question_id} 太短")
                return False

            # 检查问题格式
            if not question.content.startswith("问题"):
                logger.warning(f"问题 {question.question_id} 不以'问题'开头")

            # 检查必需字段
            if not question.source_document or not question.source_chunk_id:
                logger.error(f"问题 {question.question_id} 缺少源信息")
                return False

        return True

    def set_custom_prompts(self, system_prompt: str, human_prompt: str) -> None:
        """
        设置自定义提示词

        Args:
            system_prompt: 系统提示词模板
            human_prompt: 用户提示词模板
        """
        self.system_prompt = system_prompt
        self.human_prompt = human_prompt
        logger.info("自定义提示词已更新")

    def _extract_fallback_questions(
        self,
        response: str,
        source_chunk: DocumentChunk,
        context_package: Dict[str, Any],
        start_index: int = 1,
    ) -> List[Question]:
        """
        当结构化解析失败时使用备用方法提取问题

        Args:
            response: 原始响应文本
            source_chunk: 源块

        Returns:
            Question对象列表
        """
        questions = []

        # 按行分割并查找类似问题的内容
        lines = response.split('\n')
        question_index = start_index

        base_related_entities = list(
            dict.fromkeys(context_package.get("related_entities", []))
        )
        base_related_chunk_ids = list(
            dict.fromkeys(context_package.get("related_chunk_ids", []))
        )
        primary_chunk_id = compute_lightrag_chunk_id(source_chunk.content)
        if primary_chunk_id and primary_chunk_id not in base_related_chunk_ids:
            base_related_chunk_ids = [primary_chunk_id] + base_related_chunk_ids
        knowledge_used = bool(context_package.get("prompt_context"))

        for line in lines:
            line = line.strip()

            # 跳过空行
            if not line:
                continue
            
            # 跳过标题行（markdown标题、分类标记等）
            if line.startswith('#') or line.startswith('【') or line.startswith('##'):
                continue
            
            # 跳过问题分类标题
            if re.match(r'^(复杂|中等|简单|关联|深度|事实).*问题', line):
                continue
            
            # 跳过只包含"问题N:"但没有实际内容的行
            if re.match(r'^问题\d+[:\：]\s*$', line):
                continue

            # 查找可能是问题的行 - 必须包含问号或以疑问词开头
            if ('?' in line or '？' in line or
                    line.startswith(('如何', '什么', '为什么', '怎样', '哪些', '是否', '能否', '会不会'))):

                # 清理行内容
                cleaned_line = re.sub(r'^[\d\.\-\*\s]+', '', line)  # 移除编号
                cleaned_line = re.sub(r'^问题\d+[:\：]\s*', '', cleaned_line)  # 移除"问题N:"前缀

                # 必须有实质内容且包含问号
                if len(cleaned_line) > 15 and ('?' in cleaned_line or '？' in cleaned_line):
                    question = self._build_question_object(
                        question_content=cleaned_line,
                        source_chunk=source_chunk,
                        question_index=question_index,
                        base_related_entities=base_related_entities,
                        base_related_chunk_ids=base_related_chunk_ids,
                        primary_chunk_id=primary_chunk_id,
                        knowledge_used=knowledge_used,
                    )
                    questions.append(question)
                    question_index += 1

                    # 限制到预期的问题数量
                    if len(questions) >= self.questions_per_chunk:
                        break

        return questions
