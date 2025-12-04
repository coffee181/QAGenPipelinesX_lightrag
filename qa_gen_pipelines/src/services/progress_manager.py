"""Progress management service."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from loguru import logger

from ..utils.config import ConfigManager
from ..utils.file_utils import FileUtils
from ..utils.console_utils import safe_print, console_log


class ProgressManager:
    """Manages processing progress for resumable operations."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize progress manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.progress_file = Path(config.get("progress.progress_file", "./progress.json"))
        self.save_interval = config.get("progress.save_interval", 1)  # Save more frequently for better recovery
        self.enable_percentage_display = config.get("progress.enable_percentage_display", True)
        self.percentage_display_interval = config.get("progress.percentage_display_interval", 10)  # Display every 10%
        
        # Progress data structure
        self.progress_data = {
            "sessions": {},
            "last_updated": None
        }
        
        # Progress display tracking
        self.last_percentage_displayed = {}
        self.progress_callbacks = {}
        
        # Load existing progress
        self.load_progress()
        
        logger.info(f"Progress manager initialized with file: {self.progress_file}")
    
    def create_session(self, session_id: str, operation_type: str, total_items: int, 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a new processing session.
        
        Args:
            session_id: Unique identifier for the session
            operation_type: Type of operation (e.g., 'pdf_processing', 'question_generation')
            total_items: Total number of items to process
            metadata: Additional metadata for the session
        """
        session_data = {
            "operation_type": operation_type,
            "total_items": total_items,
            "completed_items": 0,
            "failed_items": 0,
            "processed_files": [],
            "failed_files": [],
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "status": "running",
            "metadata": metadata or {},
            "percentage_milestones": []  # Track when certain percentages were reached
        }
        
        self.progress_data["sessions"][session_id] = session_data
        self.progress_data["last_updated"] = datetime.now().isoformat()
        
        # Initialize percentage tracking
        self.last_percentage_displayed[session_id] = 0
        
        self.save_progress()
        
        # Display initial progress
        if self.enable_percentage_display:
            console_log("INFO",f"🚀 开始会话: {session_id} ({operation_type})")
            console_log("INFO",f"📊 总项目数: {total_items}")
            console_log("INFO",f"⏱️ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Created session: {session_id} ({operation_type}) with {total_items} items")
    
    def start_session(self, session_id: str, total_items: int, operation_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start a new session (alias for create_session for backward compatibility).
        
        Args:
            session_id: Unique session identifier
            total_items: Total number of items to process
            operation_type: Type of operation (e.g., 'pdf_processing', 'question_generation')
            metadata: Optional metadata for the session
        """
        self.create_session(session_id, operation_type, total_items, metadata)
    
    def update_session_progress(self, session_id: str, completed_file: str, 
                               success: bool = True, error_message: str = None) -> None:
        """
        Update progress for a session.
        
        Args:
            session_id: Session identifier
            completed_file: File that was processed
            success: Whether processing was successful
            error_message: Error message if processing failed
        """
        if session_id not in self.progress_data["sessions"]:
            logger.warning(f"Session {session_id} not found")
            return
        
        session = self.progress_data["sessions"][session_id]
        
        if success:
            session["completed_items"] += 1
            session["processed_files"].append({
                "file": completed_file,
                "timestamp": datetime.now().isoformat()
            })
        else:
            session["failed_items"] += 1
            session["failed_files"].append({
                "file": completed_file,
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            })
        
        session["last_update"] = datetime.now().isoformat()
        self.progress_data["last_updated"] = datetime.now().isoformat()
        
        # Update percentage display
        self._update_percentage_display(session_id)
        
        # Call progress callback if registered
        if session_id in self.progress_callbacks:
            try:
                stats = self.get_session_stats(session_id)
                self.progress_callbacks[session_id](stats)
            except Exception as e:
                logger.warning(f"Progress callback error for {session_id}: {e}")
        
        # Auto-save based on interval
        if session["completed_items"] % self.save_interval == 0:
            self.save_progress()
        
        logger.info(f"Session {session_id}: {session['completed_items']}/{session['total_items']} completed")
    
    def update_progress(self, session_id: str, increment: int = 1) -> None:
        """
        Update progress for a session by incrementing completed items.
        
        Args:
            session_id: Session identifier
            increment: Number of items to increment
        """
        if session_id not in self.progress_data["sessions"]:
            logger.warning(f"Session {session_id} not found")
            return
        
        session = self.progress_data["sessions"][session_id]
        session["completed_items"] += increment
        session["last_update"] = datetime.now().isoformat()
        self.progress_data["last_updated"] = datetime.now().isoformat()
        
        # Update percentage display
        self._update_percentage_display(session_id)
        
        # Call progress callback if registered
        if session_id in self.progress_callbacks:
            try:
                stats = self.get_session_stats(session_id)
                self.progress_callbacks[session_id](stats)
            except Exception as e:
                logger.warning(f"Progress callback error for {session_id}: {e}")
        
        # Auto-save based on interval
        if session["completed_items"] % self.save_interval == 0:
            self.save_progress()
        
        logger.info(f"Session {session_id}: {session['completed_items']}/{session['total_items']} completed")
    
    def add_error(self, session_id: str, error_message: str) -> None:
        """
        Add an error to a session.
        
        Args:
            session_id: Session identifier
            error_message: Error message to add
        """
        if session_id not in self.progress_data["sessions"]:
            logger.warning(f"Session {session_id} not found")
            return
        
        session = self.progress_data["sessions"][session_id]
        session["failed_items"] += 1
        
        if "errors" not in session:
            session["errors"] = []
        
        session["errors"].append({
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        })
        
        session["last_update"] = datetime.now().isoformat()
        self.progress_data["last_updated"] = datetime.now().isoformat()
        
        logger.error(f"Added error to session {session_id}: {error_message}")
    
    def reactivate_session(self, session_id: str, new_total_items: int = None) -> None:
        """
        Reactivate a completed session to continue processing.
        
        Args:
            session_id: Session identifier to reactivate
            new_total_items: New total items count (optional)
        """
        if session_id not in self.progress_data["sessions"]:
            logger.warning(f"Session {session_id} not found for reactivation")
            return
        
        session = self.progress_data["sessions"][session_id]
        
        # Update status back to running
        session["status"] = "running"
        
        # Update total items if provided
        if new_total_items and new_total_items > session["total_items"]:
            session["total_items"] = new_total_items
        
        # Remove end_time if exists
        if "end_time" in session:
            del session["end_time"]
        
        session["last_update"] = datetime.now().isoformat()
        self.progress_data["last_updated"] = datetime.now().isoformat()
        
        # Save immediately
        self.save_progress()
        
        logger.info(f"Reactivated session {session_id} with {session['completed_items']} completed items")

    def complete_session(self, session_id: str, status: str = "completed") -> None:
        """
        Mark a session as completed.
        
        Args:
            session_id: Session identifier
            status: Final status (completed, failed, cancelled)
        """
        if session_id not in self.progress_data["sessions"]:
            logger.warning(f"Session {session_id} not found")
            return
        
        session = self.progress_data["sessions"][session_id]
        session["status"] = status
        session["end_time"] = datetime.now().isoformat()
        session["last_update"] = datetime.now().isoformat()
        
        self.progress_data["last_updated"] = datetime.now().isoformat()
        self.save_progress()
        
        # Display completion summary
        if self.enable_percentage_display:
            self._display_completion_summary(session_id, status)
        
        # Clean up tracking data
        if session_id in self.last_percentage_displayed:
            del self.last_percentage_displayed[session_id]
        if session_id in self.progress_callbacks:
            del self.progress_callbacks[session_id]
        
        logger.info(f"Session {session_id} completed with status: {status}")
    
    def fail_session(self, session_id: str, error_message: str) -> None:
        """
        Mark a session as failed.
        
        Args:
            session_id: Session identifier
            error_message: Error message describing the failure
        """
        if session_id not in self.progress_data["sessions"]:
            logger.warning(f"Session {session_id} not found")
            return
        
        session = self.progress_data["sessions"][session_id]
        session["status"] = "failed"
        session["error_message"] = error_message
        session["end_time"] = datetime.now().isoformat()
        session["last_update"] = datetime.now().isoformat()
        
        self.progress_data["last_updated"] = datetime.now().isoformat()
        self.save_progress()
        
        # Display failure summary
        if self.enable_percentage_display:
            console_log('ERROR',f"❌ 会话失败: {session_id}")
            console_log('ERROR',f"📝 错误信息: {error_message}")
        
        logger.error(f"Session {session_id} failed: {error_message}")
    
    def register_progress_callback(self, session_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to be called on progress updates.
        
        Args:
            session_id: Session identifier
            callback: Function to call with session stats on updates
        """
        self.progress_callbacks[session_id] = callback
    
    def unregister_progress_callback(self, session_id: str) -> None:
        """
        Unregister progress callback for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.progress_callbacks:
            del self.progress_callbacks[session_id]
    
    def _update_percentage_display(self, session_id: str) -> None:
        """
        Update percentage display for a session.
        
        Args:
            session_id: Session identifier
        """
        if not self.enable_percentage_display or session_id not in self.progress_data["sessions"]:
            return
        
        session = self.progress_data["sessions"][session_id]
        total = session["total_items"]
        completed = session["completed_items"]
        
        if total == 0:
            return
        
        current_percentage = (completed / total) * 100
        last_displayed = self.last_percentage_displayed.get(session_id, 0)
        
        # Check if we should display a percentage milestone
        percentage_thresholds = list(range(self.percentage_display_interval, 101, self.percentage_display_interval))
        
        for threshold in percentage_thresholds:
            if current_percentage >= threshold and last_displayed < threshold:
                # Record milestone
                if "percentage_milestones" not in session:
                    session["percentage_milestones"] = []
                
                milestone = {
                    "percentage": threshold,
                    "timestamp": datetime.now().isoformat(),
                    "completed_items": completed,
                    "total_items": total
                }
                session["percentage_milestones"].append(milestone)
                
                # Display progress
                self._display_progress_milestone(session_id, threshold, completed, total)
                
                self.last_percentage_displayed[session_id] = threshold
                break
    
    def _display_progress_milestone(self, session_id: str, percentage: float, completed: int, total: int) -> None:
        """
        Display a progress milestone.
        
        Args:
            session_id: Session identifier
            percentage: Percentage completed
            completed: Number of items completed
            total: Total number of items
        """
        # Calculate ETA
        session = self.progress_data["sessions"][session_id]
        start_time = datetime.fromisoformat(session["start_time"])
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        if completed > 0:
            estimated_total_time = elapsed_time * total / completed
            remaining_time = estimated_total_time - elapsed_time
            eta_str = self._format_duration(remaining_time)
            speed = completed / elapsed_time if elapsed_time > 0 else 0
            speed_str = f"{speed:.2f} 项/秒"
        else:
            eta_str = "--:--:--"
            speed_str = "-- 项/秒"
        
        # Create progress bar
        bar_width = 30
        filled_width = int(bar_width * percentage / 100)
        bar = "█" * filled_width + "░" * (bar_width - filled_width)
        
        # Display progress
        console_log("INFO",f"📊 {session_id}: |{bar}| {percentage:5.1f}% ({completed}/{total}) "
                   f"[速度: {speed_str}, ETA: {eta_str}]")
    
    def _display_completion_summary(self, session_id: str, status: str) -> None:
        """
        Display completion summary for a session.
        
        Args:
            session_id: Session identifier
            status: Final status
        """
        stats = self.get_session_stats(session_id)
        
        if status == "completed":
            console_log("INFO", f"✅ 会话完成: {session_id}")
        elif status == "failed":
            console_log("ERROR",f"❌ 会话失败: {session_id}")
        else:
            console_log("INFO", f"🔄 会话状态: {session_id} - {status}")
        
        console_log("INFO", f"📊 最终统计:")
        console_log("INFO", f"   总项目数: {stats['total_items']}")
        console_log("INFO", f"   成功完成: {stats['completed_items']}")
        console_log("INFO", f"   失败项目: {stats['failed_items']}")
        console_log("INFO", f"   完成率: {stats['completion_percentage']:.1f}%")
        
        if stats.get('duration_seconds'):
            duration_str = self._format_duration(stats['duration_seconds'])
            console_log("INFO", f"   运行时长: {duration_str}")

        print("")  # Empty line for spacingpython
    
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in human readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds <= 0:
            return "--:--:--"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def get_session_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get progress information for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session progress data or None if not found
        """
        return self.progress_data["sessions"].get(session_id)
    
    def get_processed_files(self, session_id: str) -> List[str]:
        """
        Get list of successfully processed files for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of processed file paths
        """
        session = self.get_session_progress(session_id)
        if not session:
            return []
        
        return [item["file"] for item in session.get("processed_files", [])]
    
    def get_failed_files(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get list of failed files for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of failed file information
        """
        session = self.get_session_progress(session_id)
        if not session:
            return []
        
        return session.get("failed_files", [])
    
    def is_file_processed(self, session_id: str, file_path: str) -> bool:
        """
        Check if a file has already been processed in a session.
        
        Args:
            session_id: Session identifier
            file_path: File path to check
            
        Returns:
            True if file was already processed successfully
        """
        processed_files = self.get_processed_files(session_id)
        return str(file_path) in processed_files
    
    def get_remaining_files(self, session_id: str, all_files: List[str]) -> List[str]:
        """
        Get list of files that still need to be processed.
        
        Args:
            session_id: Session identifier
            all_files: Complete list of files to process
            
        Returns:
            List of files that haven't been processed yet
        """
        processed_files = set(self.get_processed_files(session_id))
        return [f for f in all_files if str(f) not in processed_files]
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        session = self.get_session_progress(session_id)
        if not session:
            return {}
        
        total = session["total_items"]
        completed = session["completed_items"]
        failed = session["failed_items"]
        remaining = total - completed - failed
        
        stats = {
            "total_items": total,
            "completed_items": completed,
            "failed_items": failed,
            "remaining_items": remaining,
            "completion_percentage": (completed / total * 100) if total > 0 else 0,
            "failure_percentage": (failed / total * 100) if total > 0 else 0,
            "status": session["status"],
            "start_time": session["start_time"],
            "last_update": session["last_update"],
            "operation_type": session["operation_type"],
            "metadata": session.get("metadata", {})
        }
        
        if "end_time" in session:
            stats["end_time"] = session["end_time"]
            
            # Calculate duration
            start_time = datetime.fromisoformat(session["start_time"])
            end_time = datetime.fromisoformat(session["end_time"])
            duration = end_time - start_time
            stats["duration_seconds"] = duration.total_seconds()
        
        # Add milestone information
        if "percentage_milestones" in session:
            stats["percentage_milestones"] = session["percentage_milestones"]
        
        return stats
    
    def get_progress_summary(self, session_id: str) -> str:
        """
        Get a formatted progress summary for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted progress summary string
        """
        stats = self.get_session_stats(session_id)
        if not stats:
            return f"Session {session_id} not found"
        
        percentage = stats['completion_percentage']
        completed = stats['completed_items']
        total = stats['total_items']
        failed = stats['failed_items']
        status = stats['status']
        
        summary_lines = []
        summary_lines.append(f"会话 {session_id}:")
        summary_lines.append(f"  状态: {status}")
        summary_lines.append(f"  进度: {completed}/{total} ({percentage:.1f}%)")
        
        if failed > 0:
            summary_lines.append(f"  失败: {failed}")
        
        if status == "running" and total > 0:
            # Calculate ETA
            session = self.get_session_progress(session_id)
            start_time = datetime.fromisoformat(session["start_time"])
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            if completed > 0:
                estimated_total_time = elapsed_time * total / completed
                remaining_time = estimated_total_time - elapsed_time
                eta_str = self._format_duration(remaining_time)
                summary_lines.append(f"  预计完成: {eta_str}")
        
        return "\n".join(summary_lines)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions with basic information.
        
        Returns:
            List of session summaries
        """
        sessions = []
        
        for session_id, session_data in self.progress_data["sessions"].items():
            summary = {
                "session_id": session_id,
                "operation_type": session_data["operation_type"],
                "status": session_data["status"],
                "total_items": session_data["total_items"],
                "completed_items": session_data["completed_items"],
                "failed_items": session_data["failed_items"],
                "start_time": session_data["start_time"],
                "last_update": session_data["last_update"],
                "metadata": session_data.get("metadata", {})
            }
            
            if "end_time" in session_data:
                summary["end_time"] = session_data["end_time"]
            
            sessions.append(summary)
        
        return sessions
    
    def cleanup_old_sessions(self, days_old: int = 30) -> None:
        """
        Remove old completed sessions.
        
        Args:
            days_old: Remove sessions older than this many days
        """
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        sessions_to_remove = []
        
        for session_id, session_data in self.progress_data["sessions"].items():
            if session_data["status"] in ["completed", "failed", "cancelled"]:
                last_update = datetime.fromisoformat(session_data["last_update"])
                if last_update.timestamp() < cutoff_date:
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.progress_data["sessions"][session_id]
            logger.info(f"Removed old session: {session_id}")
        
        if sessions_to_remove:
            self.save_progress()
    
    def save_progress(self) -> None:
        """Save progress data to file."""
        try:
            FileUtils.save_json_file(self.progress_data, self.progress_file)
            # 强制刷新日志输出
            from ..utils.logging_utils import UTF8Logger
            UTF8Logger.force_flush_logs()
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def load_progress(self) -> None:
        """Load progress data from file."""
        try:
            if self.progress_file.exists():
                self.progress_data = FileUtils.load_json_file(self.progress_file)
                logger.info(f"Loaded progress data with {len(self.progress_data.get('sessions', {}))} sessions")
            else:
                logger.info("No existing progress file found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            # Reset to default structure
            self.progress_data = {
                "sessions": {},
                "last_updated": None
            }
    
    def reset_session(self, session_id: str) -> None:
        """
        Reset a session to start over.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.progress_data["sessions"]:
            session = self.progress_data["sessions"][session_id]
            session["completed_items"] = 0
            session["failed_items"] = 0
            session["processed_files"] = []
            session["failed_files"] = []
            session["status"] = "running"
            session["start_time"] = datetime.now().isoformat()
            session["last_update"] = datetime.now().isoformat()
            session["percentage_milestones"] = []
            
            if "end_time" in session:
                del session["end_time"]
            
            # Reset percentage tracking
            self.last_percentage_displayed[session_id] = 0
            
            self.save_progress()
            logger.info(f"Reset session: {session_id}")
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete a session completely.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.progress_data["sessions"]:
            del self.progress_data["sessions"][session_id]
            
            # Clean up tracking data
            if session_id in self.last_percentage_displayed:
                del self.last_percentage_displayed[session_id]
            if session_id in self.progress_callbacks:
                del self.progress_callbacks[session_id]
            
            self.save_progress()
            logger.info(f"Deleted session: {session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return uuid.uuid4().hex[:8] 