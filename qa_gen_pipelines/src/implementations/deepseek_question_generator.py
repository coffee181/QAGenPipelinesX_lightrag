"""DeepSeek question generator implementation."""

import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
from datetime import datetime
from loguru import logger
from openai import OpenAI

from ..interfaces.question_generator_interface import QuestionGeneratorInterface, QuestionGenerationError
from ..models.document import DocumentChunk
from ..models.question import Question, QuestionSet
from ..utils.config import ConfigManager
from ..utils.lightrag_utils import compute_lightrag_chunk_id, LightRAGContextBuilder

if TYPE_CHECKING:
    from .lightrag_rag import LightRAGImplementation


class DeepSeekQuestionGenerator(QuestionGeneratorInterface):
    """DeepSeek-based question generation implementation."""

    def __init__(
        self,
        config: ConfigManager,
        rag: Optional["LightRAGImplementation"] = None,
    ):
        """
        Initialize DeepSeek question generator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.rag = rag
        self.api_key = config.get("question_generator.deepseek.api_key")
        self.model = config.get("question_generator.deepseek.model", "deepseek-chat")
        self.max_tokens = config.get("question_generator.deepseek.max_tokens", 2048)
        self.temperature = config.get("question_generator.deepseek.temperature", 0.7)
        self.timeout = config.get("question_generator.deepseek.timeout", 60)
        self.questions_per_chunk = config.get("question_generator.deepseek.questions_per_chunk", 10)

        self.enable_kg_context = config.get(
            "question_generator.deepseek.enable_kg_context", True
        )
        self.max_context_entities = config.get(
            "question_generator.deepseek.max_context_entities", 3
        )
        self.max_context_relations = config.get(
            "question_generator.deepseek.max_context_relations", 2
        )
        self.max_context_snippets = config.get(
            "question_generator.deepseek.max_context_snippets", 2
        )
        self.context_snippet_chars = config.get(
            "question_generator.deepseek.context_snippet_chars", 200
        )
        self.max_related_chunk_ids = config.get(
            "question_generator.deepseek.max_related_chunk_ids", 6
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

        # Load prompts from config
        self.system_prompt = config.get("prompts.system_prompt", "")
        self.human_prompt = config.get("prompts.human_prompt", "")

        if not self.api_key:
            raise QuestionGenerationError("DeepSeek API key not found in configuration")

        # Initialize OpenAI client for DeepSeek
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )

        logger.info(f"DeepSeek question generator initialized with model: {self.model}")

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
        try:
            logger.info(f"Generating questions for chunk: {chunk.chunk_id}")

            context_package = self._build_context_for_chunk(chunk)
            prompt_text = self._compose_prompt_text(
                chunk.content, context_package["prompt_context"]
            )

            # Prepare prompt with chunk content
            human_message = self.human_prompt.format(
                text=prompt_text,
                questions_per_chunk=self.questions_per_chunk,
            )

            # Call DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": human_message}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )

            # Extract response content
            response_content = response.choices[0].message.content
            logger.info(f"Received response from DeepSeek: {len(response_content)} characters")

            # Parse questions from response
            questions = self.parse_questions_from_response(
                response_content,
                chunk,
                context_package,
            )

            logger.info(f"Generated {len(questions)} questions for chunk: {chunk.chunk_id}")
            return questions

        except Exception as e:
            raise QuestionGenerationError(f"Failed to generate questions for chunk {chunk.chunk_id}: {e}")

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
        try:
            if not chunks:
                raise QuestionGenerationError("No chunks provided for question generation")

            document_id = chunks[0].document_id
            logger.info(f"Generating questions for {len(chunks)} chunks from document: {document_id}")

            all_questions = []

            for chunk in chunks:
                try:
                    chunk_questions = self.generate_questions_from_chunk(chunk)
                    all_questions.extend(chunk_questions)
                except Exception as e:
                    logger.error(f"Failed to generate questions for chunk {chunk.chunk_id}: {e}")
                    continue

            # Create QuestionSet
            question_set = QuestionSet(
                document_id=document_id,
                questions=all_questions,
                created_at=datetime.now()
            )

            logger.info(f"Generated total {len(all_questions)} questions for document: {document_id}")
            return question_set

        except Exception as e:
            raise QuestionGenerationError(f"Failed to generate questions from chunks: {e}")

    def _extract_candidate_entities(self, text: str) -> List[str]:
        """从问题文本中提取潜在实体或型号。"""
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
            logger.debug(f"Failed to build KG context for chunk {chunk.chunk_id}: {e}")
            context = LightRAGContextBuilder._empty_context()

        if not context:
            return self._empty_context_package()

        return {
            "prompt_context": context.get("prompt_context", ""),
            "related_entities": context.get("related_entities", []) or [],
            "related_chunk_ids": context.get("related_chunk_ids", []) or [],
        }

    def _compose_prompt_text(self, chunk_text: str, knowledge_context: str) -> str:
        if knowledge_context:
            return f"{chunk_text}\n\n<知识图谱参考>\n{knowledge_context}"
        return chunk_text

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

        return Question(
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

    def _clean_think_tags(self, text: str) -> str:
        if not text:
            return ""

        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned_text)
        return cleaned_text.strip()

    def parse_questions_from_response(
        self,
        response: str,
        source_chunk: DocumentChunk,
        context_package: Dict[str, Any],
    ) -> List[Question]:
        """
        Parse questions from LLM response.

        Args:
            response: Raw response from LLM
            source_chunk: Source chunk for the questions
            context_package: Knowledge graph context package

        Returns:
            List of parsed Question objects
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

            question_pattern = r"问题(\d+)：(.+?)(?=问题\d+：|$)"
            matches = re.findall(question_pattern, cleaned_response, re.DOTALL)

            for match in matches:
                question_num = int(match[0])
                question_content = match[1].strip()

                if question_content:
                    formatted_content = f"问题{question_num}：{question_content}"
                    question = self._build_question_object(
                        question_content=formatted_content,
                        source_chunk=source_chunk,
                        question_index=question_num,
                        base_related_entities=base_related_entities,
                        base_related_chunk_ids=base_related_chunk_ids,
                        primary_chunk_id=primary_chunk_id,
                        knowledge_used=knowledge_used,
                    )
                    questions.append(question)

            if not questions:
                questions = self._extract_fallback_questions(
                    cleaned_response,
                    source_chunk,
                    context_package,
                    start_index=1,
                )

            logger.info(f"Parsed {len(questions)} questions from response")
            return questions

        except Exception as e:
            raise QuestionGenerationError(f"Failed to parse questions from response: {e}")

    def validate_questions(self, questions: List[Question]) -> bool:
        """
        Validate generated questions.

        Args:
            questions: List of questions to validate

        Returns:
            True if all questions are valid, False otherwise
        """
        if not questions:
            return False

        for question in questions:
            # Check if question has content
            if not question.content or not question.content.strip():
                logger.error(f"Question {question.question_id} has no content")
                return False

            # Check if question is too short
            if len(question.content.strip()) < 10:
                logger.error(f"Question {question.question_id} is too short")
                return False

            # Check if question has proper format
            if not question.content.startswith("问题"):
                logger.warning(f"Question {question.question_id} doesn't start with '问题'")

            # Check required fields
            if not question.source_document or not question.source_chunk_id:
                logger.error(f"Question {question.question_id} missing source information")
                return False

        return True

    def set_custom_prompts(self, system_prompt: str, human_prompt: str) -> None:
        """
        Set custom prompts for question generation.

        Args:
            system_prompt: System prompt template
            human_prompt: Human prompt template
        """
        self.system_prompt = system_prompt
        self.human_prompt = human_prompt
        logger.info("Custom prompts updated")

    def _extract_fallback_questions(
        self,
        response: str,
        source_chunk: DocumentChunk,
        context_package: Dict[str, Any],
        start_index: int = 1,
    ) -> List[Question]:
        """
        Extract questions using fallback method when structured parsing fails.
        """
        questions: List[Question] = []
        lines = response.split("\n")
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
            if not line:
                continue

            if (
                "?" in line
                or "？" in line
                or line.startswith(("如何", "什么", "为什么", "怎样", "哪些", "是否"))
                or "问题" in line
            ):
                cleaned_line = re.sub(r"^[\d\.\-\*\s]+", "", line)

                if len(cleaned_line) > 10:
                    formatted_content = f"问题{question_index}：{cleaned_line}"
                    question = self._build_question_object(
                        question_content=formatted_content,
                        source_chunk=source_chunk,
                        question_index=question_index,
                        base_related_entities=base_related_entities,
                        base_related_chunk_ids=base_related_chunk_ids,
                        primary_chunk_id=primary_chunk_id,
                        knowledge_used=knowledge_used,
                    )
                    questions.append(question)
                    question_index += 1

                    if len(questions) >= self.questions_per_chunk:
                        break

        return questions