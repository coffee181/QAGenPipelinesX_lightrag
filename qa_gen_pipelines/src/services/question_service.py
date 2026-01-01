"""Question generation service."""

import uuid
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Set
from loguru import logger
from tqdm import tqdm

from ..interfaces.question_generator_interface import QuestionGeneratorInterface
from ..interfaces.text_chunker_interface import TextChunkerInterface
from ..interfaces.markdown_processor_interface import MarkdownProcessorInterface
from ..models.document import Document
from ..models.question import Question, QuestionSet
from ..utils.config import ConfigManager
from ..utils.file_utils import FileUtils
from ..utils.chunk_repository import ChunkRepository
from .progress_manager import ProgressManager


class QuestionService:
    """Service for generating questions from documents."""
    
    def __init__(self, config: ConfigManager, 
                 question_generator: QuestionGeneratorInterface,
                 text_chunker: TextChunkerInterface,
                 markdown_processor: MarkdownProcessorInterface,
                 progress_manager: ProgressManager):
        """
        Initialize question service.
        
        Args:
            config: Configuration object
            question_generator: Question generator implementation
            text_chunker: Text chunker implementation
            markdown_processor: Markdown processor implementation
            progress_manager: Progress manager for tracking progress
        """
        self.config = config
        self.question_generator = question_generator
        self.text_chunker = text_chunker
        self.markdown_processor = markdown_processor
        self.progress_manager = progress_manager
        
        # Configuration
        self.output_dir = Path(config.get("file_processing.output_dir", "./output"))
        
        # Question quality and deduplication settings
        self.enable_deduplication = config.get("question_generator.enable_deduplication", True)
        self.dedup_similarity_threshold = config.get("question_generator.dedup_similarity_threshold", 0.85)
        self.enable_quality_filter = config.get("question_generator.enable_quality_filter", True)
        # local_scope（*_scope.json）导出已移除：不再生成/维护局部检索范围文件
        
        # 🚀 优化：初始化 ChunkRepository（如果配置了持久化）
        self.chunk_repository = None
        if config.get("text_chunker.persist_chunks", False):
            try:
                self.chunk_repository = ChunkRepository(config)
                logger.info("ChunkRepository initialized for question generation")
            except Exception as e:
                logger.warning(f"Failed to initialize ChunkRepository: {e}")
        
        # Ensure output directory exists
        FileUtils.ensure_directory(self.output_dir)
        
        logger.info("Question service initialized")
    
    def generate_questions_from_document(self, document: Document, 
                                       session_id: Optional[str] = None) -> Optional[QuestionSet]:
        """
        Generate questions from a single document.
        
        Args:
            document: Document to generate questions from
            session_id: Optional session ID for progress tracking
            
        Returns:
            QuestionSet if successful, None otherwise
        """
        try:
            logger.info(f"Generating questions for document: {document.name}")
            
            # Clean markdown formatting from content if needed
            cleaned_content = self.markdown_processor.clean_llm_response(document.content)
            
            # Create a cleaned document
            cleaned_document = Document(
                file_path=document.file_path,
                content=cleaned_content,
                file_type=document.file_type,
                file_size=len(cleaned_content),
                created_at=document.created_at,
                processed_at=document.processed_at
            )
            
            # Chunk the document
            chunks = self.text_chunker.chunk_document(cleaned_document)
            
            if not chunks:
                logger.warning(f"No chunks generated for document: {document.name}")
                if session_id:
                    self.progress_manager.update_session_progress(
                        session_id, str(document.file_path), False, "No chunks generated"
                    )
                return None
            
            logger.info(f"Generated {len(chunks)} chunks for document: {document.name}")
            
            # Generate questions from chunks
            question_set = self.question_generator.generate_questions_from_chunks(chunks)
            
            # Apply quality filtering
            if self.enable_quality_filter:
                original_count = len(question_set.questions)
                question_set.questions = self._filter_quality_questions(question_set.questions, chunks)
                filtered_count = original_count - len(question_set.questions)
                if filtered_count > 0:
                    logger.info(f"Filtered out {filtered_count} low-quality questions")
            
            # Apply deduplication
            if self.enable_deduplication:
                original_count = len(question_set.questions)
                question_set.questions = self._deduplicate_questions(question_set.questions)
                dedup_count = original_count - len(question_set.questions)
                if dedup_count > 0:
                    logger.info(f"Removed {dedup_count} duplicate questions")
            
            # Validate questions
            if not self.question_generator.validate_questions(question_set.questions):
                logger.warning(f"Generated questions failed validation for document: {document.name}")
            
            # Save questions to JSONL file in multi-line format
            questions_file = self.output_dir / f"{document.name}_questions.jsonl"
            self._save_questions_multiline(question_set, questions_file)

            # local_scope（*_scope.json）导出已移除
            
            # Update progress if session provided
            if session_id:
                self.progress_manager.update_session_progress(
                    session_id, str(document.file_path), True
                )
            
            logger.info(f"✅ 问题生成完成: {document.name} → {len(question_set.questions)}个问题")
            logger.info(f"💾 问题已保存到: {questions_file}")
            return question_set
            
        except Exception as e:
            logger.error(f"Failed to generate questions for document {document.name}: {e}")
            if session_id:
                self.progress_manager.update_session_progress(
                    session_id, str(document.file_path), False, str(e)
                )
            return None
    
    def generate_questions_from_text_file(self, text_file_path: Path, 
                                        session_id: Optional[str] = None) -> Optional[QuestionSet]:
        """
        Generate questions from a text file.
        
        Args:
            text_file_path: Path to text file
            session_id: Optional session ID for progress tracking
            
        Returns:
            QuestionSet if successful, None otherwise
        """
        try:
            # Load text content
            content = FileUtils.load_text_file(text_file_path)
            
            # Create document object
            document = Document(
                file_path=text_file_path,
                content=content,
                file_type=text_file_path.suffix,
                file_size=len(content),
                created_at=text_file_path.stat().st_ctime,
                processed_at=None
            )
            
            # Generate questions
            return self.generate_questions_from_document(document, session_id)
            
        except Exception as e:
            logger.error(f"Failed to generate questions from text file {text_file_path}: {e}")
            if session_id:
                self.progress_manager.update_session_progress(
                    session_id, str(text_file_path), False, str(e)
                )
            return None
    
    def generate_questions_from_directory(
        self,
        input_dir: Path,
        resume_session: bool = True,
        skip_if_output_exists: bool = True,
    ) -> List[QuestionSet]:
        """
        Generate questions from all text files in a directory.
        
        Args:
            input_dir: Directory containing text files
            resume_session: Whether to resume from previous session
            
        Returns:
            List of QuestionSet objects
        """
        try:
            if not input_dir.exists():
                raise ValueError(f"Input directory does not exist: {input_dir}")
            
            # Find all text files
            text_files = FileUtils.get_files_by_extension(input_dir, [".txt"])
            
            if not text_files:
                logger.warning(f"No text files found in directory: {input_dir}")
                return []
            
            logger.info(f"[questions] 待处理文件数: {len(text_files)} 目录: {input_dir}")
            
            # Create session ID
            session_id = f"question_generation_{uuid.uuid4().hex[:8]}"
            
            # Check for existing session to resume
            if resume_session:
                existing_sessions = self.progress_manager.list_sessions()
                for session in existing_sessions:
                    if (session["operation_type"] == "question_generation" and 
                        session["status"] == "running" and
                        session["metadata"].get("input_dir") == str(input_dir)):
                        session_id = session["session_id"]
                        logger.info(f"Resuming existing session: {session_id}")
                        break
            
            # Get session progress or create new session
            session_progress = self.progress_manager.get_session_progress(session_id)
            if not session_progress:
                self.progress_manager.create_session(
                    session_id=session_id,
                    operation_type="question_generation",
                    total_items=len(text_files),
                    metadata={
                        "input_dir": str(input_dir),
                        "output_dir": str(self.output_dir)
                    }
                )
            
            # Get remaining files to process
            remaining_files = self.progress_manager.get_remaining_files(
                session_id, [str(f) for f in text_files]
            )
            remaining_paths = [Path(f) for f in remaining_files]
            
            logger.info(f"[questions] 开始处理剩余文件: {len(remaining_paths)}")
            
            # Process files
            question_sets = []
            
            for text_file in tqdm(remaining_paths, desc="Generating questions", leave=False):
                # 增量保护：已有输出且未强制重新生成则跳过
                if skip_if_output_exists:
                    questions_file = self.output_dir / f"{text_file.stem}_questions.jsonl"
                    if questions_file.exists() and questions_file.stat().st_mtime >= text_file.stat().st_mtime:
                        logger.info(f"[skip] {text_file.name} 已有问题文件且未过期")
                        # 标记进度为完成，避免后续重复
                        self.progress_manager.update_status(text_file, "qa_gen", "done")
                        continue

                question_set = self.generate_questions_from_text_file(text_file, session_id)
                if question_set:
                    question_sets.append(question_set)
                    logger.info(f"[ok] {text_file.name} -> {len(question_set.questions)} questions")
            
            # Complete session
            self.progress_manager.complete_session(session_id, "completed")
            
            # Get final stats
            stats = self.progress_manager.get_session_stats(session_id)
            logger.info(f"[questions] 完成: {stats['completed_items']}/{stats['total_items']} 文件")
            
            return question_sets
            
        except Exception as e:
            logger.error(f"Failed to generate questions from directory {input_dir}: {e}")
            if 'session_id' in locals():
                self.progress_manager.complete_session(session_id, "failed")
            return []
    
    def get_generation_stats(self, session_id: str) -> dict:
        """
        Get question generation statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with generation statistics
        """
        return self.progress_manager.get_session_stats(session_id)
    
    def list_generation_sessions(self) -> List[dict]:
        """
        List all question generation sessions.
        
        Returns:
            List of session information
        """
        all_sessions = self.progress_manager.list_sessions()
        question_sessions = [
            session for session in all_sessions 
            if session["operation_type"] in ["question_generation", "question_doc_generation"]
        ]
        return question_sessions
    
    def _deduplicate_questions(self, questions: List[Question]) -> List[Question]:
        """
        Remove duplicate questions using semantic similarity.
        
        Args:
            questions: List of questions to deduplicate
            
        Returns:
            List of unique questions
        """
        if not questions:
            return questions
        
        unique_questions = []
        seen_hashes: Set[str] = set()
        
        for question in questions:
            # Create a normalized version for comparison
            normalized = self._normalize_question_text(question.content)
            
            # Use hash for exact duplicates
            question_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
            
            if question_hash in seen_hashes:
                logger.debug(f"Filtered exact duplicate: {question.content[:50]}...")
                continue
            
            # Check for semantic similarity with existing questions
            is_duplicate = False
            for unique_q in unique_questions:
                unique_normalized = self._normalize_question_text(unique_q.content)
                similarity = self._calculate_text_similarity(normalized, unique_normalized)
                
                if similarity >= self.dedup_similarity_threshold:
                    logger.debug(f"Filtered similar question (similarity={similarity:.2f}): {question.content[:50]}...")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_questions.append(question)
                seen_hashes.add(question_hash)
        
        return unique_questions
    
    def _normalize_question_text(self, text: str) -> str:
        """
        Normalize question text for comparison.
        
        Args:
            text: Question text to normalize
            
        Returns:
            Normalized text
        """
        # Remove "问题N:" prefix
        text = re.sub(r'^问题\d+[:：]\s*', '', text)
        text = re.sub(r'^Question\s*\d+[:：]\s*', '', text, flags=re.IGNORECASE)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove punctuation at the end
        text = text.rstrip('?？.。！!,，')
        
        return text
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using character-level n-grams.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Use character bigrams for Chinese text
        def get_ngrams(text, n=2):
            return set(text[i:i+n] for i in range(len(text)-n+1))
        
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _filter_quality_questions(self, questions: List[Question], chunks: List) -> List[Question]:
        """
        Filter out low-quality questions.
        
        Args:
            questions: List of questions to filter
            chunks: Document chunks (for context validation)
            
        Returns:
            List of high-quality questions
        """
        quality_questions = []
        
        for question in questions:
            if self._is_quality_question(question, chunks):
                quality_questions.append(question)
            else:
                logger.debug(f"Filtered low-quality question: {question.content[:50]}...")
        
        return quality_questions
    
    def _is_quality_question(self, question: Question, chunks: List) -> bool:
        """
        Check if a question meets quality standards.
        
        Args:
            question: Question to validate
            chunks: Document chunks (for context validation)
            
        Returns:
            True if question is high quality, False otherwise
        """
        content = question.content
        
        # 1. Check if question is complete (has question mark or is a complete sentence)
        if not (content.endswith('?') or content.endswith('？') or len(content) > 15):
            logger.debug(f"Question too short or incomplete: {content}")
            return False
        
        # 2. Check minimum length
        if len(content) < 10:
            logger.debug(f"Question too short: {content}")
            return False
        
        # 3. Check for overly vague questions
        vague_patterns = [
            r'^什么$',
            r'^怎么样$', 
            r'^如何$',
            r'^是什么\??$',
            r'^是什么意思\??$',
        ]
        
        for pattern in vague_patterns:
            if re.match(pattern, content.strip()):
                logger.debug(f"Question too vague: {content}")
                return False
        
        # 4. Check for questions that are too short but don't contain specific terms
        # Relaxed check: only filter very short questions (< 15 chars) without any specific content
        if len(content) < 15:
            # Very short questions should at least contain specific entities or numbers
            has_specific_content = (
                bool(re.search(r'\d', content)) or  # Contains numbers
                bool(re.search(r'[A-Z]{2,}', content)) or  # Contains acronyms
                bool(re.search(r'[a-zA-Z]+\d+[a-zA-Z]*', content)) or  # Contains model numbers
                bool(re.search(r'[\u4e00-\u9fff]{3,}', content))  # Contains Chinese words >= 3 chars
            )
            if not has_specific_content:
                logger.debug(f"Short question lacks specific content: {content}")
                return False
        
        # 5. Check if question mentions entities that exist in source chunk
        # Find the source chunk for this question
        source_chunk = None
        for chunk in chunks:
            if chunk.chunk_id == question.source_chunk_id:
                source_chunk = chunk
                break
        
        if source_chunk:
            # 5.1 运维“可答性”约束：
            # 流程/配置/排查/恢复类问题，必须在该 chunk 中存在明确的“步骤/参数/报警/地址/菜单路径”等证据，
            # 否则这类问题大概率无法在上下文内给出可验证短答案，最终会在答案阶段变成“未找到依据”。
            procedural_triggers = [
                "如何", "怎么", "怎样",
                "配置", "设置", "设定", "启用", "开启", "关闭",
                "排查", "诊断", "处理", "恢复", "解决",
                "校准", "标定", "调试", "切换", "映射",
                "注册", "管理", "导入", "导出",
                "步骤", "流程",
            ]
            is_procedural_question = any(t in content for t in procedural_triggers)

            if is_procedural_question:
                chunk_text = source_chunk.content or ""

                # 证据特征：参数号/报警号/位号地址/步骤编号/按键菜单等（满足其一即可）
                evidence_patterns = [
                    r"参数\s*\d{2,5}",
                    r"\bP\d{2,5}\b",
                    r"\b\d{4,5}\b",  # 常见报警号/参数号
                    r"(报警|告警|ALM|Alarm)",
                    r"(BIT\d+|Y地址|X地址|PLC|I/O|IO|从站|逻辑ID)",
                    r"(步骤|按下|按键|软件键|菜单|页面|界面|选择|输入|确认|保存|加载|导入|导出)",
                    r"(\(\d+\)|（\d+）|[a-zA-Z]\)|[一二三四五六七八九十]+、)",
                ]

                has_procedural_evidence = any(re.search(p, chunk_text, re.IGNORECASE) for p in evidence_patterns)
                if not has_procedural_evidence:
                    logger.debug(
                        "Procedural question rejected (no actionable evidence in chunk): %s",
                        content[:80],
                    )
                    return False

            # Extract potential entities from question (words longer than 2 characters)
            question_words = set(re.findall(r'[\u4e00-\u9fff]{3,}|[a-zA-Z0-9]{3,}', content))
            
            # Check if question contains numbers that exist in source chunk
            question_numbers = set(re.findall(r'\d+', content))
            has_number_match = False
            if question_numbers:
                has_number_match = any(num in source_chunk.content for num in question_numbers)
            
            # Check if question is using pronouns or generic terms
            pronouns = {'这台', '这个', '这些', '该', '此'}
            has_pronoun = any(pronoun in content for pronoun in pronouns)
            
            # Check if at least one entity exists in source chunk
            has_entity_match = False
            if question_words:
                # 🔧 增强：严格验证型号名称是否存在于文档中
                # Match patterns like "VMC850L", "GSK27i", "980TDi", "RMD08" etc.
                model_patterns = re.findall(r'[A-Z]{2,}\d+[A-Z]*|[A-Z]+\d+[a-z]*|[A-Z]+\d+', content)
                
                if model_patterns:
                    # 检查每个型号是否都存在于源文档中
                    all_models_exist = True
                    for pattern in model_patterns:
                        if pattern not in source_chunk.content:
                            # 型号不存在，标记为虚构内容
                            logger.warning(f"⚠️ 检测到虚构型号: '{pattern}' 不在文档中，问题: {content[:60]}")
                            all_models_exist = False
                            break
                    
                    if not all_models_exist:
                        # 如果有任何型号是虚构的，直接拒绝这个问题
                        logger.debug(f"Question contains fabricated model names: {content[:50]}")
                        return False
                    else:
                        has_entity_match = True
                
                # Check for longer keywords (>=4 chars) which are more likely to be technical terms/model names
                if not has_entity_match:
                    long_keywords = {w for w in question_words if len(w) >= 4}
                    if long_keywords:
                        has_entity_match = any(word in source_chunk.content for word in long_keywords)
            
            # Filter logic:
            # 1. If question has pronouns but no entity match - REJECT (missing model/specific name)
            # 2. If question has no pronouns but no entity match and no number match - REJECT (no context)
            # 3. If question has number match - ACCEPT (has specific data)
            # 4. If question has entity match - ACCEPT (has specific model/name)
            if has_pronoun and not has_entity_match:
                # Question uses pronouns like "这台设备" but doesn't mention specific model/name
                logger.debug(f"Question uses pronouns but lacks specific model/name: {content[:50]}")
                return False
            elif not has_pronoun and not has_entity_match and not has_number_match:
                # Question has no pronouns but also no matching entities or numbers
                logger.debug(f"Question entities not found in source chunk: {content[:50]}")
                return False
        
        return True
    
    def _save_questions_multiline(self, question_set: QuestionSet, output_file: Path) -> None:
        """Save questions to JSONL file (single-line JSON per line)."""
        try:
            # Ensure output directory exists
            FileUtils.ensure_directory(output_file.parent)
            
            import json
            with open(output_file, "w", encoding="utf-8") as f:
                for question_data in question_set.to_jsonl_format():
                    f.write(json.dumps(question_data, ensure_ascii=False))
                    f.write("\n")

            logger.info(f"Questions saved to: {output_file} ({len(question_set.questions)} questions)")
            
        except Exception as e:
            logger.error(f"Failed to save questions: {e}")
            raise
