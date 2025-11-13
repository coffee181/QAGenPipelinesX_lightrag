"""Main application for QA Generation Pipelines."""

import logging
import argparse
import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import threading
import concurrent.futures
from datetime import datetime

# 首先修复控制台编码问题
from src.utils.console_utils import ConsoleOutputFixer, safe_print, console_log
ConsoleOutputFixer.fix_console_encoding()

from src.utils.path_utils import PathUtils
from src.utils.logging_utils import setup_project_logging, UTF8Logger

def get_executable_dir():
    """Get the directory containing the executable or script."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        return Path(sys.executable).parent
    else:
        # Running as script
        return Path(__file__).parent

def setup_runtime_paths():
    """Setup runtime paths for both executable and script modes."""
    exe_dir = get_executable_dir()
    
    # Change working directory to executable location
    os.chdir(exe_dir)
    
    # Load environment variables from .env file in executable directory
    env_file = exe_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Fallback to default .env loading
        load_dotenv()

# Setup runtime paths first
setup_runtime_paths()

from src.utils.config import ConfigManager
from src.services.progress_manager import ProgressManager
from src.services.pdf_processor import PDFProcessor
from src.services.question_service import QuestionService
from src.services.answer_service import AnswerService

# Core implementations
from src.implementations.simple_text_chunker import SimpleTextChunker
from src.implementations.deepseek_question_generator import DeepSeekQuestionGenerator
from src.implementations.local_question_generator import LocalQuestionGenerator
from src.implementations.lightrag_rag import LightRAGImplementation
from src.implementations.simple_markdown_processor import SimpleMarkdownProcessor

# Optional implementations
try:
    from src.implementations.tesseract_ocr import TesseractOCR
    TESSERACT_AVAILABLE = True
except ImportError:
    TesseractOCR = None
    TESSERACT_AVAILABLE = False

try:
    from src.implementations.paddle_ocr import PaddleOCREngine
    PADDLE_AVAILABLE = True
except ImportError:
    PaddleOCREngine = None
    PADDLE_AVAILABLE = False


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration with UTF-8 encoding."""
    return setup_project_logging(log_level)


def create_services(config: ConfigManager, logger: logging.Logger) -> tuple:
    """Create and configure all services."""
    
    # Create progress manager
    progress_manager = ProgressManager(config)
    
    # Create implementations
    ocr_provider = config.get("ocr.provider", "tesseract").lower()
    ocr = None

    if ocr_provider == "paddle":
        if PADDLE_AVAILABLE:
            ocr = PaddleOCREngine(config)
            logger.info("PaddleOCR initialized successfully")
        elif TESSERACT_AVAILABLE:
            ocr = TesseractOCR(config)
            logger.warning("PaddleOCR unavailable, falling back to Tesseract")
        else:
            logger.warning("No OCR engine available - PDF processing will be disabled")
    else:
        if TESSERACT_AVAILABLE:
            ocr = TesseractOCR(config)
            logger.info("TesseractOCR initialized successfully")
        elif PADDLE_AVAILABLE:
            ocr = PaddleOCREngine(config)
            logger.warning("TesseractOCR unavailable, using PaddleOCR instead")
        else:
            logger.warning("No OCR engine available - PDF processing will be disabled")
    
    text_chunker = SimpleTextChunker(config)
    
    rag = LightRAGImplementation(config)
    
    # 根据配置选择问题生成器
    provider = config.get("question_generator.provider", "deepseek")
    if provider == "local":
        question_generator = LocalQuestionGenerator(config, rag=rag)
        safe_print(f"使用本地模型: {config.get('question_generator.local.model_name', 'unknown')}")
    else:
        question_generator = DeepSeekQuestionGenerator(config, rag=rag)
        safe_print("使用DeepSeek API")
    
    markdown_processor = SimpleMarkdownProcessor()
    
    # Create services
    pdf_processor = PDFProcessor(
        config=config,
        ocr_implementation=ocr,
        progress_manager=progress_manager
    )
    
    question_service = QuestionService(
        config=config,
        question_generator=question_generator,
        text_chunker=text_chunker,
        markdown_processor=markdown_processor,
        progress_manager=progress_manager
    )
    
    answer_service = AnswerService(
        rag=rag,
        markdown_processor=markdown_processor,
        progress_manager=progress_manager,
        logger=logger
    )
    
    pdf_processor = (
        PDFProcessor(
            config=config,
            ocr_implementation=ocr,
            progress_manager=progress_manager
        )
        if ocr is not None
        else None
    )
    
    return pdf_processor, question_service, answer_service, progress_manager


def process_pdfs_command(args, services, logger):
    """Handle PDF processing command."""
    pdf_processor, _, _, _ = services

    if pdf_processor is None:
        logger.error("PDF processing is not available because no OCR engine is configured")
        return
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    logger.info(f"Processing PDFs from {input_path} to {output_path}")
    
    if input_path.is_file():
        # Single PDF
        result = pdf_processor.process_pdf(input_path, output_path, args.session_id)
        logger.info(f"Successfully processed PDF: {input_path.name}")
    else:
        # Directory of PDFs
        results = pdf_processor.process_directory(input_path, output_path, args.session_id)
        logger.info(f"Processed {len(results)} PDFs")


def generate_questions_command(args, services, logger):
    """Handle question generation command."""
    _, question_service, _, _ = services
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    logger.info(f"Generating questions from {input_path} to {output_path}")
    
    if input_path.is_file():
        # Single document
        result = question_service.generate_questions_for_document(
            input_path, output_path, args.session_id
        )
        logger.info(f"Generated {len(result.questions)} questions")
    else:
        # Directory of documents
        results = question_service.generate_questions_for_directory(
            input_path, output_path, args.session_id
        )
        logger.info(f"Generated questions for {len(results)} documents")


def generate_answers_command(args, services, logger):
    """Handle answer generation command."""
    _, _, answer_service, _ = services
    
    questions_path = Path(args.questions)
    working_dir = Path(args.working_dir)
    output_path = Path(args.output)
    
    if hasattr(args, 'insert_documents') and args.insert_documents:
        # Mode 1: Insert documents and generate answers
        documents_path = Path(args.insert_documents)
        
        logger.info(f"Generating answers with document insertion")
        logger.info(f"Documents: {documents_path}, Working dir: {working_dir}")
        
        # Setup knowledge base with new working directory
        answer_service.setup_knowledge_base(documents_path, working_dir)
        
        if questions_path.is_file():
            # Single questions file
            result = answer_service.generate_answers_for_questions(
                questions_path, output_path, args.session_id, resume=not args.restart
            )
            logger.info(f"Generated {len(result.qa_pairs)} QA pairs")
        else:
            # Directory of question files
            results = answer_service.generate_answers_for_directory(
                questions_path, output_path, args.session_id
            )
            logger.info(f"Generated answers for {len(results)} question files")
    else:
        # Mode 2: Use existing knowledge base
        logger.info(f"Generating answers from existing knowledge base: {working_dir}")
        
        if questions_path.is_file():
            # Single questions file
            result = answer_service.generate_answers_from_existing_kb(
                questions_path, working_dir, output_path, args.session_id, resume=not args.restart
            )
            logger.info(f"Generated {len(result.qa_pairs)} QA pairs")
        else:
            # Directory of question files - need to implement this for existing KB
            raise NotImplementedError("Directory processing with existing KB not yet implemented")


def insert_documents_command(args, services, logger):
    """Handle document insertion command."""
    _, _, answer_service, _ = services
    
    # 使用PathUtils处理中文路径
    try:
        working_dir = PathUtils.normalize_path(args.working_dir)
        documents_path = PathUtils.normalize_path(args.documents)
        
        safe_working_dir_str = PathUtils.safe_path_string(working_dir)
        safe_documents_str = PathUtils.safe_path_string(documents_path)
        
        logger.info(f"Inserting documents from {safe_documents_str} to working directory {safe_working_dir_str}")
        
        # Insert documents to working directory
        stats = answer_service.insert_documents_to_working_dir(
            documents_path, working_dir, args.session_id
        )
        
        logger.info(f"Document insertion completed: {stats}")
        
    except Exception as e:
        error_msg = f"Failed to insert documents: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Application failed: {error_msg}")
        raise


def full_pipeline_command(args, services, logger):
    """Handle full pipeline command."""
    pdf_processor, question_service, answer_service, progress_manager = services
    
    if not TESSERACT_AVAILABLE:
        logger.error("Full pipeline is not available because TesseractOCR dependencies are missing")
        logger.error("Please install pytesseract, PIL, and pdf2image packages")
        return
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Create subdirectories
    texts_dir = output_path / "texts"
    questions_dir = output_path / "questions"
    qa_dir = output_path / "qa"
    
    for dir_path in [texts_dir, questions_dir, qa_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    session_id = args.session_id or f"full_pipeline_{progress_manager._generate_session_id()}"
    
    logger.info(f"Starting full pipeline: {session_id}")
    
    try:
        # Step 1: Process PDFs
        logger.info("Step 1: Processing PDFs...")
        if input_path.is_file():
            pdf_results = [pdf_processor.process_pdf(input_path, texts_dir, f"{session_id}_pdf")]
        else:
            pdf_results = pdf_processor.process_directory(input_path, texts_dir, f"{session_id}_pdf")
        
        logger.info(f"Processed {len(pdf_results)} PDFs")
        
        # Step 2: Generate questions
        logger.info("Step 2: Generating questions...")
        question_results = question_service.generate_questions_for_directory(
            texts_dir, questions_dir, f"{session_id}_questions"
        )
        
        logger.info(f"Generated questions for {len(question_results)} documents")
        
        # Step 3: Setup knowledge base
        logger.info("Step 3: Setting up knowledge base...")
        answer_service.setup_knowledge_base(texts_dir)
        
        # Step 4: Generate answers
        logger.info("Step 4: Generating answers...")
        answer_results = answer_service.generate_answers_for_directory(
            questions_dir, qa_dir, f"{session_id}_answers"
        )
        
        logger.info(f"Generated answers for {len(answer_results)} question files")
        
        logger.info(f"Full pipeline completed successfully: {session_id}")
        
    except Exception as e:
        logger.error(f"Full pipeline failed: {str(e)}")
        raise


def show_progress_command(args, services, logger):
    """Handle show progress command."""
    _, _, _, progress_manager = services
    
    # Import progress display utilities
    from src.utils.progress_display import (
        RealTimeProgressMonitor, 
        ProgressDisplayFormatter,
        display_session_summary,
        display_sessions_overview,
        monitor_sessions_realtime
    )
    
    if hasattr(args, 'monitor') and args.monitor:
        # Real-time monitoring mode
        try:
            safe_print("🚀 启动实时进度监控...")
            
            # Get session IDs to monitor if specified
            session_ids = None
            if args.session_id:
                session_ids = [args.session_id]
            
            # Start real-time monitoring
            monitor_sessions_realtime(progress_manager, session_ids)
            
        except KeyboardInterrupt:
            safe_print("\n📊 实时监控已停止")
        except Exception as e:
            logger.error(f"Real-time monitoring failed: {e}")
    
    elif hasattr(args, 'detailed') and args.detailed:
        # Detailed view mode
        if args.session_id:
            # Show detailed summary for specific session
            display_session_summary(progress_manager, args.session_id)
        else:
            # Show detailed overview for all sessions
            display_sessions_overview(progress_manager)
    
    elif args.session_id:
        # Show specific session (standard mode)
        session = progress_manager.get_session_progress(args.session_id)
        if session:
            stats = progress_manager.get_session_stats(args.session_id)
            
            # Use enhanced formatting
            summary = progress_manager.get_progress_summary(args.session_id)
            safe_print(summary)
            
            # Display additional details
            if stats.get('percentage_milestones'):
                safe_print("\n📈 进度里程碑:")
                for milestone in stats['percentage_milestones'][-5:]:  # Show last 5 milestones
                    timestamp = datetime.fromisoformat(milestone['timestamp']).strftime('%H:%M:%S')
                    safe_print(f"   {milestone['percentage']:3.0f}% - {timestamp}")
            
            if stats.get('failure_percentage', 0) > 0:
                failed_files = progress_manager.get_failed_files(args.session_id)
                if failed_files:
                    safe_print(f"\n❌ 最近失败项目 (显示最后 3 个):")
                    for failed in failed_files[-3:]:
                        error_msg = failed.get('error', 'Unknown error')[:50]
                        safe_print(f"   • {failed.get('file', 'Unknown')}: {error_msg}")
        else:
            logger.warning(f"Session not found: {args.session_id}")
    
    else:
        # Show all sessions overview
        sessions = progress_manager.list_sessions()
        
        if not sessions:
            safe_print("📋 没有找到任何会话")
            return
        
        # Enhanced overview display
        safe_print("📊 会话概览")
        safe_print("=" * 80)
        
        # Group sessions by status
        running_sessions = [s for s in sessions if s['status'] == 'running']
        completed_sessions = [s for s in sessions if s['status'] == 'completed']
        failed_sessions = [s for s in sessions if s['status'] == 'failed']
        
        # Display running sessions with progress bars
        if running_sessions:
            safe_print("🔄 正在运行的会话:")
            for session in running_sessions:
                session_id = session['session_id']
                stats = progress_manager.get_session_stats(session_id)
                
                # Create mini progress bar
                percentage = stats['completion_percentage']
                bar_width = 20
                filled_width = int(bar_width * percentage / 100)
                bar = "█" * filled_width + "░" * (bar_width - filled_width)
                
                # Format session info
                operation_type = session['operation_type'][:15]
                safe_print(f"   {session_id[:15]:15s} |{bar}| {percentage:5.1f}% {operation_type:15s}")
            safe_print("")
        
        # Display completed sessions
        if completed_sessions:
            safe_print("✅ 已完成的会话:")
            for session in completed_sessions:
                session_id = session['session_id']
                stats = progress_manager.get_session_stats(session_id)
                
                operation_type = session['operation_type'][:15]
                completed = stats['completed_items']
                total = stats['total_items']
                duration = ""
                
                if stats.get('duration_seconds'):
                    duration_seconds = stats['duration_seconds']
                    if duration_seconds < 60:
                        duration = f"{duration_seconds:.0f}s"
                    elif duration_seconds < 3600:
                        duration = f"{duration_seconds/60:.1f}m"
                    else:
                        duration = f"{duration_seconds/3600:.1f}h"
                
                safe_print(f"   {session_id[:15]:15s} {completed:4d}/{total:4d} {operation_type:15s} {duration:>8s}")
            safe_print("")
        
        # Display failed sessions
        if failed_sessions:
            safe_print("❌ 失败的会话:")
            for session in failed_sessions:
                session_id = session['session_id']
                stats = progress_manager.get_session_stats(session_id)
                
                operation_type = session['operation_type'][:15]
                completed = stats['completed_items']
                total = stats['total_items']
                failed = stats['failed_items']
                
                safe_print(f"   {session_id[:15]:15s} {completed:4d}/{total:4d} (失败:{failed:2d}) {operation_type:15s}")
            safe_print("")
        
        # Display summary statistics
        total_sessions = len(sessions)
        safe_print("📈 统计汇总:")
        safe_print(f"   总会话数: {total_sessions}")
        safe_print(f"   运行中: {len(running_sessions)}")
        safe_print(f"   已完成: {len(completed_sessions)}")
        safe_print(f"   失败: {len(failed_sessions)}")
        
        # Show tips for additional commands
        safe_print("\n💡 提示:")
        safe_print("   使用 --detailed 查看详细信息")
        safe_print("   使用 --monitor 启动实时监控")
        safe_print("   使用 --session-id <ID> 查看特定会话")


def generate_qapairs_command(args, services, logger):
    """Handle QA pairs generation command."""
    pdf_processor, question_service, answer_service, progress_manager = services
    
    # 使用PathUtils处理中文路径
    try:
        input_path = PathUtils.normalize_path(args.input)
        working_dir = PathUtils.normalize_path(args.working_dir)
        
        safe_input_str = PathUtils.safe_path_string(input_path)
        safe_working_dir_str = PathUtils.safe_path_string(working_dir)
        
        # 处理输出参数
        if hasattr(args, 'directory_mode') and args.directory_mode:
            # 目录模式：自动生成文件名
            output_questions_dir = PathUtils.normalize_path(args.output_questions_file)
            output_dir = PathUtils.normalize_path(args.output_file)
            
            # 确保输出目录存在
            output_questions_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            safe_questions_dir_str = PathUtils.safe_path_string(output_questions_dir)
            safe_output_dir_str = PathUtils.safe_path_string(output_dir)
            
            logger.info(f"Starting QA pairs generation in directory mode")
            logger.info(f"Input: {safe_input_str}")
            logger.info(f"Questions dir: {safe_questions_dir_str}")
            logger.info(f"Output dir: {safe_output_dir_str}")
            logger.info(f"Working dir: {safe_working_dir_str}")
        else:
            # 文件模式：直接使用指定的文件路径
            output_questions_file = PathUtils.normalize_path(args.output_questions_file)
            output_file = PathUtils.normalize_path(args.output_file)
            
            safe_questions_file_str = PathUtils.safe_path_string(output_questions_file)
            safe_output_file_str = PathUtils.safe_path_string(output_file)
            
            logger.info(f"Starting QA pairs generation in file mode")
            logger.info(f"Input: {safe_input_str}")
            logger.info(f"Questions file: {safe_questions_file_str}")
            logger.info(f"Output file: {safe_output_file_str}")
            logger.info(f"Working dir: {safe_working_dir_str}")
        
        # 生成会话ID
        session_id = args.session_id or f"qapairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 检查是否需要文档插入
        if hasattr(args, 'insert_documents') and args.insert_documents:
            # 模式1：并行执行问题生成和文档插入
            logger.info("Mode: Parallel question generation and document insertion")
            _parallel_question_and_insertion_mode(
                args, services, logger, session_id, 
                input_path, working_dir, safe_input_str, safe_working_dir_str
            )
        else:
            # 模式2：仅问题生成和答案生成
            logger.info("Mode: Question generation and answer generation only")
            _sequential_question_and_answer_mode(
                args, services, logger, session_id,
                input_path, working_dir, safe_input_str, safe_working_dir_str
            )
        
        logger.info(f"QA pairs generation completed: {session_id}")
        
    except Exception as e:
        error_msg = f"Failed to generate QA pairs: {str(e)}"
        logger.error(error_msg)
        raise


def _parallel_question_and_insertion_mode(args, services, logger, session_id, 
                                         input_path, working_dir, safe_input_str, safe_working_dir_str):
    """并行执行问题生成和文档插入模式"""
    _, question_service, answer_service, _ = services
    
    question_results = []
    insertion_stats = None
    
    def generate_questions_task():
        """问题生成任务"""
        nonlocal question_results
        try:
            logger.info("Starting question generation task...")
            
            if hasattr(args, 'directory_mode') and args.directory_mode:
                # 目录模式
                output_questions_dir = PathUtils.normalize_path(args.output_questions_file)
                question_results = question_service.generate_questions_for_directory(
                    input_path, output_questions_dir, f"{session_id}_questions"
                )
            else:
                # 文件模式
                if input_path.is_file():
                    output_questions_file = PathUtils.normalize_path(args.output_questions_file)
                    output_questions_file.parent.mkdir(parents=True, exist_ok=True)
                    result = question_service.generate_questions_for_document(
                        input_path, output_questions_file.parent, f"{session_id}_questions"
                    )
                    if result:
                        # 重命名生成的问题文件到指定位置
                        default_name = output_questions_file.parent / f"{input_path.stem}_questions.jsonl"
                        if default_name.exists() and default_name != output_questions_file:
                            default_name.rename(output_questions_file)
                        question_results = [result]
                    else:
                        question_results = []
                else:
                    raise ValueError(f"Input path is not a file in file mode: {input_path}")
            
            logger.info(f"Question generation completed: {len(question_results)} question sets")
            
        except Exception as e:
            logger.error(f"Question generation task failed: {e}")
            raise
    
    def insert_documents_task():
        """文档插入任务"""
        nonlocal insertion_stats
        try:
            logger.info("Starting document insertion task...")
            insertion_stats = answer_service.insert_documents_to_working_dir(
                input_path, working_dir, f"{session_id}_insertion"
            )
            logger.info(f"Document insertion completed: {insertion_stats}")
            
        except Exception as e:
            logger.error(f"Document insertion task failed: {e}")
            raise
    
    # 并行执行问题生成和文档插入
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        question_future = executor.submit(generate_questions_task)
        insertion_future = executor.submit(insert_documents_task)
        
        # 等待两个任务完成
        try:
            question_future.result()
            insertion_future.result()
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise
    
    # 验证两个任务都成功完成
    if not question_results:
        raise ValueError("No questions were generated")
    
    if insertion_stats is None:
        raise ValueError("Document insertion failed")
    
    # 现在生成答案
    logger.info("Starting answer generation...")
    _generate_answers_from_questions(
        args, services, logger, session_id, question_results
    )


def _sequential_question_and_answer_mode(args, services, logger, session_id,
                                       input_path, working_dir, safe_input_str, safe_working_dir_str):
    """顺序执行问题生成和答案生成模式"""
    _, question_service, answer_service, _ = services
    
    # 步骤1：生成问题
    logger.info("Step 1: Generating questions...")
    
    question_results = []
    if hasattr(args, 'directory_mode') and args.directory_mode:
        # 目录模式
        output_questions_dir = PathUtils.normalize_path(args.output_questions_file)
        question_results = question_service.generate_questions_for_directory(
            input_path, output_questions_dir, f"{session_id}_questions"
        )
    else:
        # 文件模式
        if input_path.is_file():
            output_questions_file = PathUtils.normalize_path(args.output_questions_file)
            output_questions_file.parent.mkdir(parents=True, exist_ok=True)
            result = question_service.generate_questions_for_document(
                input_path, output_questions_file.parent, f"{session_id}_questions"
            )
            if result:
                # 重命名生成的问题文件到指定位置
                default_name = output_questions_file.parent / f"{input_path.stem}_questions.jsonl"
                if default_name.exists() and default_name != output_questions_file:
                    default_name.rename(output_questions_file)
                question_results = [result]
            else:
                question_results = []
        else:
            raise ValueError(f"Input path is not a file in file mode: {input_path}")
    
    if not question_results:
        raise ValueError("No questions were generated")
    
    logger.info(f"Generated questions for {len(question_results)} documents")
    
    # 步骤2：设置知识库
    logger.info("Step 2: Setting up knowledge base...")
    answer_service.setup_knowledge_base(input_path, working_dir)
    
    # 步骤3：生成答案
    logger.info("Step 3: Generating answers...")
    _generate_answers_from_questions(
        args, services, logger, session_id, question_results
    )


def _generate_answers_from_questions(args, services, logger, session_id, question_results):
    """从问题结果生成答案"""
    _, _, answer_service, _ = services
    
    if hasattr(args, 'directory_mode') and args.directory_mode:
        # 目录模式：批量处理多个问题文件
        output_questions_dir = PathUtils.normalize_path(args.output_questions_file)
        output_dir = PathUtils.normalize_path(args.output_file)
        
        answer_results = {}
        for question_result in question_results:
            # 构建对应的输出文件路径
            questions_file = output_questions_dir / f"{question_result.document_id}_questions.jsonl"
            output_qa_file = output_dir / f"{question_result.document_id}_qapairs.jsonl"
            
            if questions_file.exists():
                logger.info(f"Generating answers for: {PathUtils.safe_path_string(questions_file)}")
                
                qa_result = answer_service.generate_answers_from_existing_kb(
                    questions_file, 
                    PathUtils.normalize_path(args.working_dir),
                    output_qa_file,
                    f"{session_id}_answers",
                    resume=True
                )
                
                answer_results[question_result.document_id] = qa_result
                logger.info(f"Generated {len(qa_result.qa_pairs)} QA pairs for {question_result.document_id}")
        
        logger.info(f"Generated answers for {len(answer_results)} documents")
        
    else:
        # 文件模式：处理单个问题文件
        output_questions_file = PathUtils.normalize_path(args.output_questions_file)
        output_file = PathUtils.normalize_path(args.output_file)
        
        qa_result = answer_service.generate_answers_from_existing_kb(
            output_questions_file,
            PathUtils.normalize_path(args.working_dir),
            output_file,
            f"{session_id}_answers",
            resume=True
        )
        
        logger.info(f"Generated {len(qa_result.qa_pairs)} QA pairs")


def _get_file_pairs_for_directory_mode(input_path, output_questions_dir, output_dir):
    """为目录模式获取文件对应关系"""
    pairs = []
    
    if input_path.is_file():
        # 单个文件
        stem = input_path.stem
        questions_file = output_questions_dir / f"{stem}_questions.jsonl"
        output_file = output_dir / f"{stem}_qapairs.jsonl"
        pairs.append((input_path, questions_file, output_file))
    else:
        # 目录中的多个文件
        for text_file in input_path.glob("*.txt"):
            stem = text_file.stem
            questions_file = output_questions_dir / f"{stem}_questions.jsonl"
            output_file = output_dir / f"{stem}_qapairs.jsonl"
            pairs.append((text_file, questions_file, output_file))
    
    return pairs


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="QA Generation Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--session-id", help="Session ID for progress tracking")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # PDF processing command
    pdf_parser = subparsers.add_parser("process-pdfs", help="Process PDF files")
    pdf_parser.add_argument("input", help="Input PDF file or directory")
    pdf_parser.add_argument("output", help="Output directory for text files")
    
    # Question generation command
    questions_parser = subparsers.add_parser("generate-questions", help="Generate questions")
    questions_parser.add_argument("input", help="Input text file or directory")
    questions_parser.add_argument("output", help="Output directory for question files")
    
    # Answer generation command
    answers_parser = subparsers.add_parser("generate-answers", help="Generate answers from questions")
    answers_parser.add_argument("questions", help="Questions file or directory (JSONL format)")
    answers_parser.add_argument("working_dir", help="Working directory containing knowledge base (for existing KB) or target directory (for new KB)")
    answers_parser.add_argument("output", help="Output file or directory for QA results")
    answers_parser.add_argument("-i", "--insert-documents", help="Documents to insert into knowledge base (enables insertion mode - creates new KB in working_dir)")
    answers_parser.add_argument("--restart", action="store_true", help="Restart from beginning (ignore existing progress)")
    
    # QA pairs generation command (NEW)
    qapairs_parser = subparsers.add_parser("generate-qapairs", help="Generate questions and answers from text documents")
    qapairs_parser.add_argument("input", help="Input text file or directory")
    qapairs_parser.add_argument("output_questions_file", help="Output questions JSONL file or directory (use with -d)")
    qapairs_parser.add_argument("working_dir", help="Working directory for vector database")
    qapairs_parser.add_argument("output_file", help="Output QA pairs JSONL file or directory (use with -d)")
    qapairs_parser.add_argument("-d", "--directory-mode", action="store_true", 
                                help="Directory mode: auto-generate filenames (input_name_questions.jsonl, input_name_qapairs.jsonl)")
    qapairs_parser.add_argument("-i", "--insert-documents", action="store_true", 
                                help="Parallel mode: simultaneously insert documents and generate questions, then generate answers")
    
    # Insert documents command
    insert_parser = subparsers.add_parser("insert-documents", help="Insert documents into a knowledge base working directory")
    insert_parser.add_argument("working_dir", help="Target working directory for the knowledge base")
    insert_parser.add_argument("documents", help="Documents file or directory to insert")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("full-pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("input", help="Input PDF file or directory")
    pipeline_parser.add_argument("output", help="Output directory")
    
    # Progress command
    progress_parser = subparsers.add_parser("show-progress", help="Show progress")
    progress_parser.add_argument("--session-id", help="Show specific session progress")
    progress_parser.add_argument("--detailed", action="store_true", help="Show detailed progress information")
    progress_parser.add_argument("--monitor", action="store_true", help="Start real-time progress monitoring")
    progress_parser.add_argument("--refresh-interval", type=float, default=1.0, help="Refresh interval for monitoring (seconds)")
    progress_parser.add_argument("--show-all", action="store_true", help="Show all sessions including completed ones")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Resolve configuration file path relative to executable directory
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = get_executable_dir() / args.config
        
        # Load configuration
        config = ConfigManager(config_path)
        
        # Create services
        services = create_services(config, logger)
        
        # Execute command
        if args.command == "process-pdfs":
            process_pdfs_command(args, services, logger)
        elif args.command == "generate-questions":
            generate_questions_command(args, services, logger)
        elif args.command == "generate-answers":
            generate_answers_command(args, services, logger)
        elif args.command == "generate-qapairs":
            generate_qapairs_command(args, services, logger)
        elif args.command == "insert-documents":
            insert_documents_command(args, services, logger)
        elif args.command == "full-pipeline":
            full_pipeline_command(args, services, logger)
        elif args.command == "show-progress":
            show_progress_command(args, services, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            return
            
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 