"""PDF processing service."""

import uuid
from pathlib import Path
from typing import List, Optional
from loguru import logger
from tqdm import tqdm

from ..interfaces.ocr_interface import OCRInterface
from ..models.document import Document
from ..utils.config import ConfigManager
from ..utils.file_utils import FileUtils
from .progress_manager import ProgressManager


class PDFProcessor:
    """Service for processing PDF files with OCR."""
    
    def __init__(self, config: ConfigManager, ocr_implementation: OCRInterface, 
                 progress_manager: ProgressManager):
        """
        Initialize PDF processor.
        
        Args:
            config: Configuration object
            ocr_implementation: OCR implementation to use
            progress_manager: Progress manager for tracking progress
        """
        self.config = config
        self.ocr = ocr_implementation
        self.progress_manager = progress_manager
        
        # Configuration
        self.input_formats = config.get("file_processing.input_formats", [".pdf"])
        self.output_dir = Path(config.get("file_processing.output_dir", "./output"))
        self.temp_dir = Path(config.get("file_processing.temp_dir", "./temp"))
        self.batch_size = config.get("file_processing.batch_size", 10)
        
        # Ensure directories exist
        FileUtils.ensure_directory(self.output_dir)
        FileUtils.ensure_directory(self.temp_dir)
        
        logger.info(f"PDF processor initialized with output dir: {self.output_dir}")
    
    def process_single_pdf(self, pdf_path: Path, session_id: Optional[str] = None) -> Optional[Document]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            session_id: Optional session ID for progress tracking
            
        Returns:
            Document object if successful, None otherwise
        """
        try:
            import time
            start_time = time.time()
            
            logger.info(f"📄 开始处理PDF文件: {pdf_path.name}")
            logger.info(f"📁 文件路径: {pdf_path}")
            logger.info(f"📊 文件大小: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # Check if file is supported
            if not self.ocr.is_supported_format(pdf_path):
                logger.error(f"❌ 不支持的文件格式: {pdf_path}")
                if session_id:
                    self.progress_manager.update_session_progress(
                        session_id, str(pdf_path), False, "Unsupported file format"
                    )
                return None
            
            logger.info(f"🔍 开始OCR文字提取...")
            # Process PDF to document
            document = self.ocr.process_pdf_to_document(pdf_path, output_dir=self.output_dir)
            
            ocr_time = time.time() - start_time
            logger.info(f"✅ OCR提取完成，耗时: {ocr_time:.2f} 秒")
            logger.info(f"📝 提取文本长度: {len(document.content)} 字符")
            
            # Save extracted text
            text_filename = pdf_path.stem + ".txt"
            text_path = self.output_dir / text_filename
            logger.info(f"💾 保存提取的文本到: {text_path}")
            FileUtils.save_text_file(document.content, text_path)
            
            # Update progress if session provided
            if session_id:
                self.progress_manager.update_session_progress(
                    session_id, str(pdf_path), True
                )
            
            total_time = time.time() - start_time
            logger.info(f"🎉 PDF处理完成: {pdf_path.name}")
            logger.info(f"   ⏰ 总耗时: {total_time:.2f} 秒")
            logger.info(f"   📄 文本长度: {len(document.content)} 字符")
            logger.info(f"   💾 保存位置: {text_path}")
            
            # 强制刷新日志输出
            from ..utils.logging_utils import UTF8Logger
            UTF8Logger.force_flush_logs()
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            if session_id:
                self.progress_manager.update_session_progress(
                    session_id, str(pdf_path), False, str(e)
                )
            return None
    
    def process_pdf_directory(self, input_dir: Path, resume_session: bool = True) -> List[Document]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            resume_session: Whether to resume from previous session
            
        Returns:
            List of processed Document objects
        """
        try:
            import time
            start_time = time.time()
            
            logger.info(f"📁 开始处理PDF目录: {input_dir}")
            
            if not input_dir.exists():
                raise ValueError(f"Input directory does not exist: {input_dir}")
            
            # Find all PDF files
            pdf_files = FileUtils.get_files_by_extension(input_dir, self.input_formats)
            
            if not pdf_files:
                logger.warning(f"⚠️  目录中未找到PDF文件: {input_dir}")
                return []
            
            # 计算总文件大小
            total_size = sum(f.stat().st_size for f in pdf_files)
            total_size_mb = total_size / 1024 / 1024
            
            logger.info(f"📊 发现 {len(pdf_files)} 个PDF文件")
            logger.info(f"📁 目录路径: {input_dir}")
            logger.info(f"💾 总文件大小: {total_size_mb:.2f} MB")
            logger.info(f"📄 文件列表:")
            for i, pdf_file in enumerate(pdf_files, 1):
                file_size_mb = pdf_file.stat().st_size / 1024 / 1024
                logger.info(f"   {i}. {pdf_file.name} ({file_size_mb:.2f} MB)")
            
            # Create session ID
            session_id = f"pdf_processing_{uuid.uuid4().hex[:8]}"
            
            # Check for existing session to resume
            if resume_session:
                existing_sessions = self.progress_manager.list_sessions()
                best_session = None
                max_completed = 0
                
                for session in existing_sessions:
                    if (session["operation_type"] == "pdf_processing" and 
                        session["status"] in ["running", "completed"] and
                        session["metadata"].get("input_dir") == str(input_dir)):
                        
                        # Find session with most completed files
                        if session["completed_items"] > max_completed:
                            max_completed = session["completed_items"]
                            best_session = session
                
                if best_session:
                    session_id = best_session["session_id"]
                    logger.info(f"🔄 恢复现有会话: {session_id} (状态: {best_session['status']}, 已完成: {best_session['completed_items']}个文件)")
                    
                    # 如果会话已完成，重新激活它
                    if best_session["status"] == "completed":
                        self.progress_manager.reactivate_session(session_id, len(pdf_files))
                        logger.info(f"🔄 重新激活已完成会话: {session_id}")
            
            # Get session progress or create new session
            session_progress = self.progress_manager.get_session_progress(session_id)
            if not session_progress:
                self.progress_manager.create_session(
                    session_id=session_id,
                    operation_type="pdf_processing",
                    total_items=len(pdf_files),
                    metadata={
                        "input_dir": str(input_dir),
                        "output_dir": str(self.output_dir)
                    }
                )
            
            # Get remaining files to process
            remaining_files = self.progress_manager.get_remaining_files(
                session_id, [str(f) for f in pdf_files]
            )
            remaining_paths = [Path(f) for f in remaining_files]
            
            logger.info(f"🚀 开始处理 {len(remaining_paths)} 个PDF文件")
            
            # Process files
            documents = []
            success_count = 0
            failed_count = 0
            
            for i, pdf_path in enumerate(remaining_paths, 1):
                logger.info(f"📄 处理文件 {i}/{len(remaining_paths)}: {pdf_path.name}")
                
                document = self.process_single_pdf(pdf_path, session_id)
                if document:
                    documents.append(document)
                    success_count += 1
                    logger.info(f"✅ 文件 {i}/{len(remaining_paths)} 处理成功")
                else:
                    failed_count += 1
                    logger.error(f"❌ 文件 {i}/{len(remaining_paths)} 处理失败")
                
                # 计算进度和ETA
                if i > 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_file = elapsed_time / i
                    remaining_files = len(remaining_paths) - i
                    eta_seconds = remaining_files * avg_time_per_file
                    eta_minutes = eta_seconds / 60
                    
                    if eta_minutes < 1:
                        eta_str = f"{eta_seconds:.0f}秒"
                    else:
                        eta_str = f"{eta_minutes:.1f}分钟"
                    
                    logger.info(f"📊 进度: {i}/{len(remaining_paths)} ({i/len(remaining_paths)*100:.1f}%) | 预计剩余: {eta_str}")
                    
                    # 强制刷新日志输出
                    from ..utils.logging_utils import UTF8Logger
                    UTF8Logger.force_flush_logs()
            
            # Complete session
            self.progress_manager.complete_session(session_id, "completed")
            
            # Get final stats
            stats = self.progress_manager.get_session_stats(session_id)
            total_time = time.time() - start_time
            
            logger.info(f"🎉 PDF目录处理完成:")
            logger.info(f"   ✅ 成功: {success_count}/{len(remaining_paths)} 个文件")
            logger.info(f"   ❌ 失败: {failed_count} 个文件")
            logger.info(f"   ⏰ 总耗时: {total_time/60:.1f} 分钟")
            logger.info(f"   📊 平均每文件: {total_time/len(remaining_paths):.1f} 秒")
            logger.info(f"   📄 生成文档: {len(documents)} 个")
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process PDF directory {input_dir}: {e}")
            if 'session_id' in locals():
                self.progress_manager.complete_session(session_id, "failed")
            return []
    
    def process_pdf_list(self, pdf_paths: List[Path], resume_session: bool = True) -> List[Document]:
        """
        Process a specific list of PDF files.
        
        Args:
            pdf_paths: List of PDF file paths
            resume_session: Whether to resume from previous session
            
        Returns:
            List of processed Document objects
        """
        try:
            if not pdf_paths:
                logger.warning("No PDF files provided")
                return []
            
            logger.info(f"Processing {len(pdf_paths)} PDF files")
            
            # Create session ID
            session_id = f"pdf_list_processing_{uuid.uuid4().hex[:8]}"
            
            # Create session
            self.progress_manager.create_session(
                session_id=session_id,
                operation_type="pdf_list_processing",
                total_items=len(pdf_paths),
                metadata={
                    "file_count": len(pdf_paths),
                    "output_dir": str(self.output_dir)
                }
            )
            
            # Process files
            documents = []
            
            for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
                document = self.process_single_pdf(pdf_path, session_id)
                if document:
                    documents.append(document)
            
            # Complete session
            self.progress_manager.complete_session(session_id, "completed")
            
            # Get final stats
            stats = self.progress_manager.get_session_stats(session_id)
            logger.info(f"PDF list processing completed: {stats['completed_items']}/{stats['total_items']} successful")
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process PDF list: {e}")
            if 'session_id' in locals():
                self.progress_manager.complete_session(session_id, "failed")
            return []
    
    def get_processing_stats(self, session_id: str) -> dict:
        """
        Get processing statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with processing statistics
        """
        return self.progress_manager.get_session_stats(session_id)
    
    def list_processing_sessions(self) -> List[dict]:
        """
        Get list of all PDF processing sessions.
        
        Returns:
            List of session information dictionaries
        """
        sessions = self.progress_manager.list_sessions()
        pdf_sessions = [
            session for session in sessions 
            if session["operation_type"] in ["pdf_processing", "pdf_list_processing"]
        ]
        return pdf_sessions
    
    def process_directory(self, input_dir: Path, output_dir: Path, session_id: Optional[str] = None) -> List[Document]:
        """
        Process all PDF files in a directory (alias for backward compatibility).
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Output directory for processed files
            session_id: Optional session ID for progress tracking
            
        Returns:
            List of processed Document objects
        """
        # Set output directory temporarily
        original_output_dir = self.output_dir
        self.output_dir = Path(output_dir)
        FileUtils.ensure_directory(self.output_dir)
        
        try:
            # Process directory
            documents = self.process_pdf_directory(input_dir, resume_session=True)
            return documents
        finally:
            # Restore original output directory
            self.output_dir = original_output_dir
    
    def process_pdf(self, pdf_path: Path, output_dir: Path, session_id: Optional[str] = None) -> Optional[Document]:
        """
        Process a single PDF file (alias for backward compatibility).
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for processed files
            session_id: Optional session ID for progress tracking
            
        Returns:
            Document object if successful, None otherwise
        """
        # Set output directory temporarily
        original_output_dir = self.output_dir
        self.output_dir = Path(output_dir)
        FileUtils.ensure_directory(self.output_dir)
        
        try:
            # Process single PDF
            return self.process_single_pdf(pdf_path, session_id)
        finally:
            # Restore original output directory
            self.output_dir = original_output_dir
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                FileUtils.delete_directory_contents(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up temp files: {e}")
    
    def validate_pdf_file(self, pdf_path: Path) -> bool:
        """
        Validate if a PDF file can be processed.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if file is valid and can be processed
        """
        try:
            # Check if file exists
            if not pdf_path.exists():
                return False
            
            # Check if file is supported format
            if not self.ocr.is_supported_format(pdf_path):
                return False
            
            # Check file size (basic validation)
            file_size = pdf_path.stat().st_size
            if file_size == 0:
                return False
            
            # Additional validation could be added here
            # (e.g., checking if PDF is readable, not corrupted, etc.)
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating PDF file {pdf_path}: {e}")
            return False 