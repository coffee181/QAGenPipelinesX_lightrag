"""Answer generation service."""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json
import asyncio

from ..interfaces.rag_interface import RAGInterface, RAGError
from ..interfaces.markdown_processor_interface import MarkdownProcessorInterface
from ..models.document import Document
from ..models.question import Question, QuestionSet
from ..models.qa_pair import QAPair, QASet
from ..utils.file_utils import FileUtils
from ..utils.path_utils import PathUtils
from ..utils.thread_event_loop import get_or_create_event_loop
from .progress_manager import ProgressManager


class AnswerType(Enum):
    """Answer type classification."""
    VALID_POSITIVE = "valid_positive"      # 有效的正面答案
    VALID_NEGATIVE = "valid_negative"      # 有效的否定答案 ("不支持XX功能")
    INVALID_NO_INFO = "invalid_no_info"    # 无效("文档中没有相关信息")
    INVALID_ERROR = "invalid_error"        # 错误信息


class AnswerService:
    """Service for generating answers using RAG."""

    def __init__(
            self,
            rag: RAGInterface,
            markdown_processor: MarkdownProcessorInterface,
            progress_manager: ProgressManager,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the answer service.
        
        Args:
            rag: RAG implementation for answer generation
            markdown_processor: Markdown processor for cleaning answers
            progress_manager: Progress manager for tracking operations
            logger: Optional logger instance
        """
        self.rag = rag
        self.markdown_processor = markdown_processor
        self.progress_manager = progress_manager
        self.logger = logger or logging.getLogger(__name__)

    def setup_knowledge_base(self, documents_path: Path, working_dir: Optional[Path] = None) -> None:
        """
        Setup the knowledge base with documents.
        
        Args:
            documents_path: Path to directory containing processed documents
            working_dir: Working directory for the knowledge base
            
        Raises:
            AnswerServiceError: If setup fails
        """
        try:
            # 使用新的路径工具标准化路径
            normalized_documents_path = PathUtils.normalize_path(documents_path)
            safe_path_str = PathUtils.safe_path_string(normalized_documents_path)

            self.logger.info(f"Setting up knowledge base from: {safe_path_str}")

            if working_dir:
                normalized_working_dir = PathUtils.normalize_path(working_dir)
                # Check if working directory exists and has data
                if normalized_working_dir.exists() and any(normalized_working_dir.glob("*.json")):
                    # Working directory exists with data - append to existing KB
                    self.logger.info(f"Appending to existing knowledge base: {PathUtils.safe_path_string(normalized_working_dir)}")
                    self.rag.use_existing_knowledge_base(normalized_working_dir)
                else:
                    # Working directory doesn't exist or is empty - create new KB
                    self.logger.info(f"Creating new knowledge base: {PathUtils.safe_path_string(normalized_working_dir)}")
                    self.rag.set_working_directory(normalized_working_dir)
            else:
                # No working directory specified - clear default KB
                self.logger.info("Using default working directory with clean KB")
                self.rag.clear_knowledge_base()

            # 验证文档路径
            is_valid, error_msg = PathUtils.validate_path(
                normalized_documents_path,
                require_exists=True
            )

            if not is_valid:
                raise AnswerServiceError(f"Invalid documents path: {error_msg}")

            # Load documents from directory
            if normalized_documents_path.is_file() and normalized_documents_path.suffix == '.txt':
                # Single document
                document = self._load_document_from_file(normalized_documents_path)
                self.rag.insert_document(document)
            elif normalized_documents_path.is_dir():
                # Directory of documents
                documents = self._load_documents_from_directory(normalized_documents_path)
                if documents:
                    self.rag.insert_documents_batch(documents)
                else:
                    raise AnswerServiceError(f"No valid documents found in {safe_path_str}")
            else:
                raise AnswerServiceError(f"Invalid documents path: {safe_path_str} (not a .txt file or directory)")

            # Get knowledge base stats
            stats = self.rag.get_knowledge_base_stats()
            self.logger.info(f"Knowledge base setup complete: {stats}")

        except AnswerServiceError:
            # 重新抛出已经格式化的错误
            raise
        except Exception as e:
            error_msg = f"Failed to setup knowledge base: {str(e)}"
            self.logger.error(error_msg)
            raise AnswerServiceError(error_msg) from e

    def generate_answers_for_questions(
            self,
            questions_file: Path,
            output_file: Path,
            session_id: Optional[str] = None,
            resume: bool = True
    ) -> QASet:
        """
        Generate answers for questions from a file.
        
        Args:
            questions_file: Path to questions JSONL file
            output_file: Path to output QA pairs JSONL file
            session_id: Optional session ID for progress tracking
            resume: Whether to resume from previous session
            
        Returns:
            QASet: Generated QA pairs
            
        Raises:
            AnswerServiceError: If generation fails
        """
        try:
            self.logger.info(f"Generating answers for questions file: {questions_file}")
            
            # Load questions
            all_questions = self._load_questions_from_file(questions_file)
            if not all_questions:
                raise AnswerServiceError(f"No questions found in file: {questions_file}")
            
            self.logger.info(f"Loaded {len(all_questions)} questions from {questions_file}")
            
            # Create or resume session
            if not session_id:
                session_id = f"answer_gen_{questions_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Check for existing progress
            existing_qa_pairs = []
            if resume and output_file.exists():
                existing_qa_pairs = self._load_existing_qa_pairs(output_file)
                self.logger.info(f"Loaded {len(existing_qa_pairs)} existing QA pairs")
            
            # Filter out already answered questions
            questions_to_process = self._filter_unanswered_questions(all_questions, existing_qa_pairs)
            self.logger.info(f"Processing {len(questions_to_process)} remaining questions")
            
            # Start or update session
            if not resume or not existing_qa_pairs:
                # New session
                self.progress_manager.start_session(
                    session_id=session_id,
                    total_items=len(all_questions),
                    operation_type="answer_generation"
                )

            # Update progress for existing pairs
            if existing_qa_pairs:
                self.progress_manager.update_progress(session_id, len(existing_qa_pairs))

            # Generate answers for remaining questions
            if questions_to_process:
                self.logger.info(f"开始生成答案: {len(questions_to_process)}个问题")
                new_qa_set = self._generate_answers_batch(questions_to_process, session_id, output_file)
                new_qa_pairs = new_qa_set.qa_pairs
                self.logger.info(f"答案生成完成: {len(new_qa_pairs)}个答案")
            else:
                new_qa_pairs = []

            # Combine existing and new QA pairs
            all_qa_pairs = existing_qa_pairs + new_qa_pairs

            # Create final QA set
            qa_set = QASet(
                document_id=session_id,
                qa_pairs=all_qa_pairs,
                created_at=datetime.now()
            )

            # Save final results
            if not resume or new_qa_pairs:  # Only save if not resuming or if we have new pairs
                self._save_qa_set(qa_set, output_file)

            # Complete session
            self.progress_manager.complete_session(session_id)

            self.logger.info(f"Answer generation completed: {len(qa_set.qa_pairs)} total QA pairs "
                             f"({len(existing_qa_pairs)} existing + {len(new_qa_pairs)} new)")
            return qa_set

        except Exception as e:
            if session_id:
                self.progress_manager.fail_session(session_id, str(e))
            error_msg = f"Failed to generate answers: {str(e)}"
            self.logger.error(error_msg)
            raise AnswerServiceError(error_msg) from e

    def generate_answers_for_directory(
            self,
            questions_dir: Path,
            output_dir: Path,
            session_id: Optional[str] = None
    ) -> Dict[str, QASet]:
        """
        Generate answers for all question files in a directory.
        
        Args:
            questions_dir: Directory containing question JSONL files
            output_dir: Directory for output QA JSONL files
            session_id: Optional session ID for progress tracking
            
        Returns:
            Dictionary mapping file names to QASet objects
            
        Raises:
            AnswerServiceError: If batch processing fails
        """
        try:
            # Create session if not provided
            if session_id is None:
                session_id = f"answer_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.logger.info(f"Starting batch answer generation: {session_id}")

            # Find question files
            question_files = list(questions_dir.glob("*questions.jsonl"))
            if not question_files:
                raise AnswerServiceError(f"No question files found in {questions_dir}")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize progress
            self.progress_manager.start_session(
                session_id=session_id,
                total_items=len(question_files),
                operation_type="batch_answer_generation"
            )

            results = {}

            for question_file in question_files:
                try:
                    # Generate output filename
                    base_name = question_file.stem.replace("_questions", "")
                    output_file = output_dir / f"{base_name}_qa.jsonl"

                    # Generate answers
                    qa_set = self.generate_answers_for_questions(
                        question_file, output_file, f"{session_id}_{base_name}"
                    )

                    results[question_file.name] = qa_set

                    # Update progress
                    self.progress_manager.update_progress(session_id, 1)

                except Exception as e:
                    self.logger.error(f"Failed to process {question_file}: {str(e)}")
                    self.progress_manager.add_error(session_id, f"{question_file.name}: {str(e)}")
                    continue

            # Complete session
            self.progress_manager.complete_session(session_id)

            self.logger.info(f"Batch answer generation completed: {len(results)} files processed")
            return results

        except Exception as e:
            if session_id:
                self.progress_manager.fail_session(session_id, str(e))
            error_msg = f"Failed to process directory: {str(e)}"
            self.logger.error(error_msg)
            raise AnswerServiceError(error_msg) from e

    def resume_answer_generation(self, session_id: str) -> Optional[QASet]:
        """
        Resume a failed answer generation session.
        
        Args:
            session_id: Session ID to resume
            
        Returns:
            QASet if resumption successful, None otherwise
            
        Raises:
            AnswerServiceError: If resumption fails
        """
        try:
            session = self.progress_manager.get_session(session_id)
            if not session:
                raise AnswerServiceError(f"Session not found: {session_id}")

            if session.status == "completed":
                self.logger.info(f"Session {session_id} already completed")
                return None

            self.logger.info(f"Resuming answer generation session: {session_id}")

            # Resume logic would depend on how we store intermediate results
            # For now, we'll just log that resumption is not implemented
            self.logger.warning("Answer generation resumption not yet implemented")
            return None

        except Exception as e:
            error_msg = f"Failed to resume session {session_id}: {str(e)}"
            self.logger.error(error_msg)
            raise AnswerServiceError(error_msg) from e

    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """
        Get information about the current knowledge base.
        
        Returns:
            Dictionary containing knowledge base information
        """
        try:
            return self.rag.get_knowledge_base_stats()
        except Exception as e:
            self.logger.error(f"Failed to get knowledge base info: {str(e)}")
            return {"error": str(e)}

    def _load_document_from_file(self, file_path: Path) -> Document:
        """Load a single document from file."""
        content = file_path.read_text(encoding='utf-8')
        return Document(
            file_path=file_path,
            content=content,
            file_type=file_path.suffix,
            file_size=len(content),
            created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
            processed_at=datetime.now()
        )

    def _load_documents_from_directory(self, directory_path: Path) -> List[Document]:
        """Load all documents from a directory."""
        documents = []

        for file_path in directory_path.glob("*.txt"):
            try:
                document = self._load_document_from_file(file_path)
                documents.append(document)
            except Exception as e:
                self.logger.warning(f"Failed to load document {file_path}: {str(e)}")
                continue

        return documents

    def _load_questions_from_file(self, questions_file: Path) -> List[Question]:
        """Load questions from JSONL file (supports both single-line and multi-line formats)."""
        try:
            try:
                data = FileUtils.load_jsonl(questions_file)
            except Exception:
                # 兼容旧的“多行 JSON（每个对象被 json.dump(indent=2) 写出）”格式：
                # 这种文件无法用标准 jsonlines 逐行解析，这里按空行分块回退解析。
                raw = questions_file.read_text(encoding="utf-8")
                blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
                data = [json.loads(b) for b in blocks]
            questions = []

            for item in data:
                # Support for standard question format
                if "question_id" in item and ("text" in item or "content" in item):
                    # Standard question format
                    question_text = item.get("text") or item.get("content")
                    question = Question(
                        question_id=item["question_id"],
                        content=question_text,
                        # 优先使用 question 文件中显式写入的 source_document（通常是原始 .txt 路径或文档名）
                        # 其次兼容旧字段 source
                        source_document=item.get("source_document")
                        or item.get("source")
                        or questions_file.stem,
                        source_chunk_id=item.get("source_chunk_id", "unknown"),
                        question_index=item.get("question_index", 1),
                        created_at=datetime.now(),
                        metadata={
                            "file": str(questions_file),
                            "question_type": item.get("question_type"),
                            "difficulty": item.get("difficulty"),
                            "category": item.get("category"),
                            "tags": item.get("tags", [])
                        }
                    )
                    questions.append(question)

                # Support for messages format (backward compatibility)
                elif "messages" in item and isinstance(item["messages"], list):
                    for i, msg in enumerate(item["messages"]):
                        # New format: messages is a list of dicts with full Question data
                        if isinstance(msg, dict):
                            question = Question(
                                question_id=msg.get("question_id", f"{questions_file.stem}_{i}"),
                                content=msg.get("content", ""),
                                source_document=msg.get("source_document", questions_file.stem),
                                source_chunk_id=msg.get("source_chunk_id", "unknown"),
                                question_index=msg.get("question_index", i+1),
                                created_at=datetime.fromisoformat(msg["created_at"]) if msg.get("created_at") else datetime.now(),
                                metadata=msg.get("metadata", {})  # 加载metadata（包含预生成的答案）
                            )
                        # Old format: messages is a list of strings
                        else:
                            question = Question(
                                question_id=f"{questions_file.stem}_{i}",
                                content=str(msg),
                                source_document=questions_file.stem,
                                source_chunk_id="unknown",
                                question_index=i+1,
                                created_at=datetime.now(),
                                metadata={"file": str(questions_file)}
                            )
                        questions.append(question)

            return questions

        except Exception as e:
            raise AnswerServiceError(f"Failed to load questions from {questions_file}: {str(e)}")

    def _load_existing_qa_pairs(self, output_file: Path) -> List[QAPair]:
        """Load existing QA pairs from output file for resuming."""
        try:
            qa_pairs = []

            if not output_file.exists():
                return qa_pairs

            # Load existing QA pairs from file
            jsonl_data = FileUtils.load_jsonl(output_file)

            for item in jsonl_data:
                if isinstance(item, dict) and "messages" in item:
                    # Messages format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                    messages = item["messages"]

                    # Process messages in pairs (user question, assistant answer)
                    for i in range(0, len(messages), 2):
                        if i + 1 < len(messages):
                            user_msg = messages[i]
                            assistant_msg = messages[i + 1]

                            if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                                # Create QA pair with synthetic question_id
                                question_id = f"qa_{len(qa_pairs) + 1}"

                                qa_pair = QAPair(
                                    question_id=question_id,
                                    question=user_msg.get("content", ""),
                                    answer=assistant_msg.get("content", ""),
                                    source_document="unknown",
                                    confidence_score=1.0,
                                    metadata={"resumed": True}
                                )
                                qa_pairs.append(qa_pair)

            self.logger.info(f"Loaded {len(qa_pairs)} existing QA pairs from {output_file}")
            return qa_pairs

        except Exception as e:
            error_msg = f"Failed to load existing QA pairs from {output_file}: {str(e)}"
            self.logger.error(error_msg)
            raise AnswerServiceError(error_msg) from e

    def _generate_answers_batch(self, questions: List[Question], session_id: str, output_file: Path = None) -> QASet:
        """Generate answers for a batch of questions（单次写入，无增量保存）."""
        try:
            total_questions = len(questions)
            self.logger.info(f"🚀 开始为 {total_questions} 个问题生成答案")
            
            qa_pairs: List[QAPair] = []
            for i, question in enumerate(questions, 1):
                try:
                    self.logger.info(f"📝 处理问题 {i}/{total_questions}: {question.content[:100]}{'...' if len(question.content) > 100 else ''}")
                    self.logger.info(f"🆔 问题ID: {question.question_id}")
                    
                    # 始终使用 RAG 生成答案（充分利用向量化和知识图谱）
                    raw_answer = None
                    max_retries = 0
                    
                    # 使用 RAG 查询并进行答案质量验证
                    for attempt in range(max_retries + 1):
                        try:
                            raw_answer = self.rag.query_single_question(
                                question.content,
                                source_document=question.source_document,
                            )
                            
                            # Clean <think> tags before validation
                            cleaned_for_validation = self.markdown_processor.clean_llm_response(raw_answer)
                            
                            # 轻量化幻觉检测：仅记录关键提醒
                            if not self._verify_answer_authenticity(question.content, cleaned_for_validation):
                                self.logger.debug("可能存在幻觉：仅记录，不阻断")
                            
                            # Classify answer type
                            answer_type = self._classify_answer_type(cleaned_for_validation)

                            # 不再因“无依据类答案”导致整题失败：允许负向/无依据答案正常落盘
                            if answer_type in [AnswerType.VALID_POSITIVE, AnswerType.VALID_NEGATIVE]:
                                break

                            # 仍然保留最基础的兜底：记录后继续（不抛异常阻断整题）
                            self.logger.warning(
                                "[answers] 答案质量不佳，按原样保留并继续: type=%s, preview=%s",
                                getattr(answer_type, "value", str(answer_type)),
                                cleaned_for_validation[:100],
                            )
                            break
                                    
                        except Exception as e:
                            self.logger.warning(f"生成答案失败: {e}")
                            raise e

                    # Process answer with markdown processor
                    processed_answer = self.markdown_processor.clean_llm_response(raw_answer)

                    qa_pair = QAPair(
                        question_id=question.question_id,
                        question=question.content,
                        answer=processed_answer,
                        source_document=question.source_document,
                        confidence_score=1.0,  # RAG doesn't provide confidence scores
                        metadata={
                            "raw_answer": raw_answer,
                            "processing_session": session_id,
                            "answer_length": len(processed_answer)
                        }
                    )

                    qa_pairs.append(qa_pair)

                    if i % 10 == 0 or i == total_questions:
                        self.logger.info(f"[answers] 进度 {i}/{total_questions}")

                    # Update progress only after successful generation
                    self.progress_manager.update_progress(session_id, 1)

                except Exception as e:
                    self.logger.warning(f"[answers] 问题 {i}/{total_questions} 失败: {e}")
                    self.progress_manager.add_error(session_id, f"Question {question.question_id}: {str(e)}")
                    # Don't update progress for failed questions
                    continue

            # Final save（批量一次写入）
            qa_set = QASet(
                document_id=session_id,
                qa_pairs=qa_pairs.copy(),
                created_at=datetime.now()
            )
            if output_file:
                self.logger.info(f"[answers] 保存 {len(qa_pairs)} QA 对 -> {output_file}")
                self._save_qa_set(qa_set, output_file)

            # Log generation summary
            total_questions = len(questions)
            successful_qa_pairs = len(qa_pairs)
            failed_questions = total_questions - successful_qa_pairs

            self.logger.info(f"[answers] 完成 {successful_qa_pairs}/{total_questions}, 失败 {failed_questions}")

            return qa_set

        except Exception as e:
            self.logger.error(f"[answers] 批量答案生成失败: {e}")
            raise

    def _verify_answer_authenticity(self, question: str, answer: str) -> bool:
        """
        验证答案是否可能包含幻觉（编造内容）
        
        Args:
            question: 原始问题
            answer: 生成的答案
            
        Returns:
            bool: True表示答案可信，False表示可能有幻觉
        """
        import re
        
        # 如果答案说找不到信息，认为是诚实的
        no_info_keywords = ["未找到依据", "无法找到依据", "未检索到", "无法找到", "找不到", "没有相关信息", "未提供", "文档中没有"]
        if any(kw in answer for kw in no_info_keywords):
            return True
        
        # 提取问题中的型号名称
        question_models = set(re.findall(r'[A-Z]{2,}\d+[A-Z]*|[A-Z]+\d+[a-z]*', question))
        
        # 提取答案中的型号名称
        answer_models = set(re.findall(r'[A-Z]{2,}\d+[A-Z]*|[A-Z]+\d+[a-z]*', answer))
        
        # 🚨 检测幻觉：答案中出现了问题中没有的新型号
        new_models = answer_models - question_models
        
        if new_models:
            # 答案引入了新的型号，可能是幻觉
            self.logger.warning(f"⚠️ 检测到可能的幻觉：答案中出现新型号 {new_models}，问题只提到 {question_models}")
            self.logger.warning(f"   问题: {question[:60]}")
            self.logger.warning(f"   答案: {answer[:100]}")
            return False
        
        # 🚨 检测数值不一致
        # 提取问题中的数字
        question_numbers = set(re.findall(r'\d+(?:\.\d+)?', question))
        answer_numbers = set(re.findall(r'\d+(?:\.\d+)?', answer))
        
        # 如果答案中的数字与问题完全不重叠，且答案包含技术参数，可能有问题
        if question_numbers and answer_numbers and not (question_numbers & answer_numbers):
            # 检查答案是否像技术规格（包含单位）
            has_units = any(unit in answer for unit in ['mm', 'rpm', 'MPa', 'kW', 'kg', '°C', 'Hz'])
            if has_units and len(answer) > 50:
                self.logger.warning(f"⚠️ 检测到可能的幻觉：答案中的数值与问题不一致")
                self.logger.warning(f"   问题数值: {question_numbers}")
                self.logger.warning(f"   答案数值: {answer_numbers}")
                # 不立即拒绝，只是警告，因为可能是合理的补充信息
        
        return True
    
    def _classify_answer_type(self, answer: str) -> AnswerType:
        """
        Classify the answer type to determine if it's valid and what kind.
        
        Args:
            answer: Generated answer to classify
            
        Returns:
            AnswerType enum value
        """
        if not answer or not answer.strip():
            return AnswerType.INVALID_ERROR

        answer_stripped = answer.strip()

        # 先处理“诚实的无依据/无信息”回答：这在本项目中是允许的有效负向答案
        # （与 answer_system_prompt 的“未找到依据”策略保持一致）
        honest_no_info_patterns = [
            "未找到依据",
            "无法找到依据",
            "未检索到依据",
            "未检索到",
            "无法从提供的内容中找到",
            "无法从知识库中找到",
            "文档中没有",
            "文档中未包含",
            "文档中未提及",
            "未提供",
            "没有提供",
            "没有相关信息",
            "找不到",
            "未找到",
            "未能找到",
        ]
        if any(p in answer_stripped for p in honest_no_info_patterns):
            return AnswerType.VALID_NEGATIVE
        
        # First, check for technical negatives (valuable answers)
        technical_negative_patterns = [
            "不支持", "不具备", "不包括", "不包含", "不适用", "不兼容",
            "无需", "无此", "没有此", "不需要", "不必",
            "禁止", "不允许", "不能", "不可",
            "非", "否", "未设置", "未配置",
        ]
        
        for pattern in technical_negative_patterns:
            if pattern in answer:
                # 技术性否定不要求很长（常见的是一句话“XX不支持YY”）
                return AnswerType.VALID_NEGATIVE

        # Check for common error messages and "no information" responses
        no_info_patterns = [
            # 查询错误
            "查询超时",
            "请稍后重试",
            "查询过程中出现问题",
            "无法生成答案",
            
            # "无法找到"相关
            "无法从知识库中找到",
            "无法找到",
            "找不到",
            "没有找到",
            "未找到",
            "无法找到关于",
            "目前无法找到",
            "未能找到",
            "无法获取",
            
            # "没有信息"相关
            "没有相关信息",
            "没有相关的",
            "缺乏相关",
            "缺乏详细",
            "没有详细",
            "没有明确",
            "没有具体",
            "未提供",
            "没有提供",
            "没有给出",
            "未给出",
            
            # "文档中没有"相关
            "文档中未包含",
            "文档中没有",
            "文档中未提及",
            "文档未涉及",
            "文档中缺乏",
            "知识库中缺乏",
            "提供的信息中没有",
            "所提供的内容中没有",
            
            # "根据提供的信息"后跟否定
            "根据提供的信息，目前没有",
            "根据提供的信息，没有",
            "根据提供的信息，无法",
            "根据提供的资料，没有",
            "根据提供的文档，没有",
            "根据现有信息，没有",
            "根据现有资料，没有",
            
            # 建议性回复（表示没有答案）
            "建议查阅",
            "建议联系",
            "建议参考",
            "建议咨询",
            "请查阅",
            "请联系",
            "请参考",
            "请咨询",
            
            # 特定型号相关
            "未包含该特定型号",
            "未涉及该特定型号",
            "没有涉及该型号",
            "未涉及该型号",
            "未包含该型号",
            
            # 英文错误模式
            "Error:",
            "抱歉",
            "Sorry",
            "I don't have",
            "I cannot",
            "I can't",
            "No information",
            "not found",
            "unable to find",
            "does not contain",
            "does not provide",
            "not provided",
            "not mentioned",
            "not specified",
            
            # 其他否定表述
            "缺少",
            "缺失",
            "不足以回答",
            "无法回答",
            "难以回答",
        ]

        answer_lower = answer.lower().strip()
        for pattern in no_info_patterns:
            if pattern.lower() in answer_lower:
                self.logger.debug(f"Answer classified as NO_INFO: contains pattern '{pattern}'")
                return AnswerType.INVALID_NO_INFO
        
        # 检查答案开头是否明确表示没有信息
        negative_starts = [
            "根据提供的信息，目前没有",
            "根据提供的信息，没有",
            "根据提供的信息，无法",
            "根据所提供的信息，没有",
            "根据文档，没有",
            "根据文档，无法",
            "很抱歉",
            "抱歉",
        ]
        
        for negative_start in negative_starts:
            if answer.strip().startswith(negative_start):
                self.logger.debug(f"Answer classified as NO_INFO: starts with negative phrase '{negative_start}'")
                return AnswerType.INVALID_NO_INFO

        # Check if answer is too short (likely an error message or incomplete answer)
        # But allow short answers if they contain technical content (numbers, units, specific terms)
        answer_stripped = answer.strip()
        if len(answer_stripped) < 50:  # Very short answers need extra validation
            # Check if it contains technical content markers
            import re
            has_numbers = bool(re.search(r'\d', answer_stripped))
            has_units = bool(re.search(r'(mm|cm|m|kg|r/min|rpm|Hz|kW|V|A|°C|MPa|%)', answer_stripped, re.IGNORECASE))
            has_colon = ':' in answer_stripped or '：' in answer_stripped
            
            # If it has technical content markers, it's likely valid despite being short
            if not (has_numbers or has_units or has_colon):
                self.logger.debug(f"Answer classified as ERROR: too short and lacks technical content ({len(answer_stripped)} chars)")
                return AnswerType.INVALID_ERROR
        
        # 检查答案是否主要由"无法回答"类型的否定词组成
        # 但要排除那些给出具体否定答案的情况（如"该设备不支持XX功能"）
        sentences = [s.strip() for s in answer.replace('。', '.').replace('！', '!').replace('？', '?').split('.') if s.strip()]
        if len(sentences) > 1:  # 只对多句子答案检查（单句答案可能就是简洁的否定回答）
            # 只检查那些表示"无法回答"的否定词，而不是正常的技术否定描述
            meta_negative_phrases = [
                "没有相关", "没有提供", "没有给出", "没有说明", "没有提及", "没有涉及",
                "无法找到", "无法确定", "无法获取", "无法回答",
                "未提供", "未给出", "未说明", "未提及",
                "缺乏相关", "缺少", "缺失",
                "不足以回答", "难以回答"
            ]
            negative_sentence_count = sum(1 for s in sentences if any(phrase in s for phrase in meta_negative_phrases))
            if negative_sentence_count / len(sentences) > 0.5:  # 超过50%的句子含"无法回答"类型的否定
                self.logger.debug(f"Answer classified as NO_INFO: too many meta-negative sentences ({negative_sentence_count}/{len(sentences)})")
                return AnswerType.INVALID_NO_INFO

        # If we get here, the answer is valid and positive
        return AnswerType.VALID_POSITIVE

    def _save_qa_set(self, qa_set: QASet, output_file: Path) -> None:
        """Save QA set to JSONL file (single-line JSON per QA)."""
        try:
            # Ensure output directory exists
            FileUtils.ensure_directory(output_file.parent)

            with open(output_file, 'w', encoding='utf-8') as f:
                for qa_pair in qa_set.qa_pairs:
                    qa_data = {
                        "question": qa_pair.question,
                        "answer": qa_pair.answer,
                        "source_document": qa_pair.source_document,
                        "question_id": qa_pair.question_id,
                        "confidence_score": qa_pair.confidence_score,
                        "metadata": qa_pair.metadata,
                        "created_at": qa_pair.created_at.isoformat() if qa_pair.created_at else None
                    }
                    f.write(json.dumps(qa_data, ensure_ascii=False))
                    f.write("\n")

            self.logger.info(f"QA set saved to: {output_file} ({len(qa_set.qa_pairs)} QA pairs)")

        except Exception as e:
            self.logger.error(f"Failed to save QA set: {e}")
            raise

    def generate_answers_from_existing_kb(
            self,
            questions_file: Path,
            working_dir: Path,
            output_file: Path,
            session_id: Optional[str] = None,
            resume: bool = True
    ) -> QASet:
        """
        Generate answers using an existing knowledge base.
        
        Args:
            questions_file: Path to questions JSONL file
            working_dir: Working directory containing existing knowledge base
            output_file: Path to output QA JSONL file
            session_id: Optional session ID for progress tracking
            resume: Whether to resume from existing progress (default: True)
            
        Returns:
            QASet containing generated QA pairs
            
        Raises:
            AnswerServiceError: If answer generation fails
        """
        try:
            # Create session if not provided
            if session_id is None:
                session_id = f"answer_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.logger.info(f"Generating answers from existing knowledge base: {working_dir}")

            # Use existing knowledge base
            self.rag.use_existing_knowledge_base(working_dir)

            # Load questions
            all_questions = self._load_questions_from_file(questions_file)
            if not all_questions:
                raise AnswerServiceError(f"No questions found in {questions_file}")

            # Check for resume
            questions_to_process = all_questions
            existing_qa_pairs = []

            if resume and output_file.exists():
                try:
                    existing_qa_pairs = self._load_existing_qa_pairs(output_file)
                    processed_question_ids = {qa.question_id for qa in existing_qa_pairs}
                    questions_to_process = [q for q in all_questions if q.question_id not in processed_question_ids]

                    self.logger.info(f"Resuming from {len(existing_qa_pairs)} existing QA pairs, "
                                     f"{len(questions_to_process)} questions remaining")
                except Exception as e:
                    self.logger.warning(f"Failed to load existing progress: {e}, starting fresh")
                    questions_to_process = all_questions
                    existing_qa_pairs = []

            # Initialize or update progress
            if resume and self.progress_manager.get_session_progress(session_id):
                # Session exists, continue
                self.logger.info(f"Continuing existing session: {session_id}")
            else:
                # New session
                self.progress_manager.start_session(
                    session_id=session_id,
                    total_items=len(all_questions),
                    operation_type="answer_generation"
                )

            # Update progress for existing pairs
            if existing_qa_pairs:
                self.progress_manager.update_progress(session_id, len(existing_qa_pairs))

            # Generate answers for remaining questions
            if questions_to_process:
                new_qa_set = self._generate_answers_batch(questions_to_process, session_id, output_file)
                new_qa_pairs = new_qa_set.qa_pairs
            else:
                new_qa_pairs = []

            # Combine existing and new QA pairs
            all_qa_pairs = existing_qa_pairs + new_qa_pairs

            # Create final QA set
            qa_set = QASet(
                document_id=session_id,
                qa_pairs=all_qa_pairs,
                created_at=datetime.now()
            )

            # Save final results
            if not resume or new_qa_pairs:  # Only save if not resuming or if we have new pairs
                self._save_qa_set(qa_set, output_file)

            # Complete session
            self.progress_manager.complete_session(session_id)

            self.logger.info(f"Answer generation completed: {len(qa_set.qa_pairs)} total QA pairs "
                             f"({len(existing_qa_pairs)} existing + {len(new_qa_pairs)} new)")
            return qa_set

        except Exception as e:
            if session_id:
                self.progress_manager.fail_session(session_id, str(e))
            error_msg = f"Failed to generate answers: {str(e)}"
            self.logger.error(error_msg)
            raise AnswerServiceError(error_msg) from e


class AnswerServiceError(Exception):
    """Custom exception for answer service errors."""
    pass 