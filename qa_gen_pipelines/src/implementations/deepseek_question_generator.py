"""DeepSeek question generator implementation."""

import re
import uuid
from typing import List
from datetime import datetime
from loguru import logger
from openai import OpenAI

from ..interfaces.question_generator_interface import QuestionGeneratorInterface, QuestionGenerationError
from ..models.document import DocumentChunk
from ..models.question import Question, QuestionSet
from ..utils.config import ConfigManager


class DeepSeekQuestionGenerator(QuestionGeneratorInterface):
    """DeepSeek-based question generation implementation."""

    def __init__(self, config: ConfigManager):
        """
        Initialize DeepSeek question generator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.api_key = config.get("question_generator.deepseek.api_key")
        self.model = config.get("question_generator.deepseek.model", "deepseek-chat")
        self.max_tokens = config.get("question_generator.deepseek.max_tokens", 2048)
        self.temperature = config.get("question_generator.deepseek.temperature", 0.7)
        self.timeout = config.get("question_generator.deepseek.timeout", 60)
        self.questions_per_chunk = config.get("question_generator.deepseek.questions_per_chunk", 10)

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

            # Prepare prompt with chunk content
            human_message = self.human_prompt.format(text=chunk.content, questions_per_chunk=self.questions_per_chunk)

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
            questions = self.parse_questions_from_response(response_content, chunk)

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
        try:
            questions = []

            # Pattern to match questions starting with "问题N："
            question_pattern = r'问题(\d+)：(.+?)(?=问题\d+：|$)'

            matches = re.findall(question_pattern, response, re.DOTALL)

            for match in matches:
                question_num = int(match[0])
                question_content = match[1].strip()

                if question_content:
                    question = Question(
                        question_id=str(uuid.uuid4()),
                        content=f"问题{question_num}：{question_content}",
                        source_document=source_chunk.document_id,
                        source_chunk_id=source_chunk.chunk_id,
                        question_index=question_num,
                        created_at=datetime.now()
                    )
                    questions.append(question)

            # If no structured questions found, try to extract any questions
            if not questions:
                questions = self._extract_fallback_questions(response, source_chunk)

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

    def _extract_fallback_questions(self, response: str, source_chunk: DocumentChunk) -> List[Question]:
        """
        Extract questions using fallback method when structured parsing fails.

        Args:
            response: Raw response text
            source_chunk: Source chunk

        Returns:
            List of Question objects
        """
        questions = []

        # Split by lines and look for question-like content
        lines = response.split('\n')
        question_index = 1

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Look for lines that might be questions
            if ('?' in line or '？' in line or
                    line.startswith(('如何', '什么', '为什么', '怎样', '哪些', '是否')) or
                    '问题' in line):

                # Clean up the line
                cleaned_line = re.sub(r'^[\d\.\-\*\s]+', '', line)  # Remove numbering

                if len(cleaned_line) > 10:  # Minimum question length
                    question = Question(
                        question_id=str(uuid.uuid4()),
                        content=f"问题{question_index}：{cleaned_line}",
                        source_document=source_chunk.document_id,
                        source_chunk_id=source_chunk.chunk_id,
                        question_index=question_index,
                        created_at=datetime.now()
                    )
                    questions.append(question)
                    question_index += 1

                    # Limit to expected number of questions
                    if len(questions) >= self.questions_per_chunk:
                        break

        return questions 