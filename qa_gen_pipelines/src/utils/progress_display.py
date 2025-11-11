"""Progress display utilities for QA Generation Pipeline."""

import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil

from .console_utils import safe_print


class ProgressBar:
    """Console progress bar with percentage display."""
    
    def __init__(self, total: int, description: str = "", width: int = 50):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of items
            description: Description text
            width: Width of progress bar in characters
        """
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        self.last_update_time = 0
        self.update_interval = 0.1  # Minimum update interval in seconds
        
        # Terminal width detection
        try:
            terminal_width = shutil.get_terminal_size().columns
            # Adjust width based on terminal size
            available_width = terminal_width - len(description) - 30  # Space for percentage and time
            self.width = min(width, max(20, available_width))
        except:
            self.width = width
    
    def update(self, current: int, force_update: bool = False) -> None:
        """
        Update progress bar.
        
        Args:
            current: Current progress value
            force_update: Force update even if within update interval
        """
        current_time = time.time()
        
        # Rate limiting
        if not force_update and current_time - self.last_update_time < self.update_interval:
            return
        
        self.current = min(current, self.total)
        self.last_update_time = current_time
        
        # Calculate percentage
        percentage = (self.current / self.total * 100) if self.total > 0 else 0
        
        # Calculate filled width
        filled_width = int(self.width * self.current / self.total) if self.total > 0 else 0
        
        # Create progress bar
        bar = "█" * filled_width + "░" * (self.width - filled_width)
        
        # Calculate time info
        elapsed_time = current_time - self.start_time
        if self.current > 0 and self.total > 0:
            estimated_total_time = elapsed_time * self.total / self.current
            remaining_time = estimated_total_time - elapsed_time
            eta_str = self._format_time(remaining_time)
        else:
            eta_str = "--:--"
        
        # Format output
        progress_text = (
            f"\r{self.description} |{bar}| "
            f"{self.current}/{self.total} "
            f"({percentage:5.1f}%) "
            f"ETA: {eta_str}"
        )
        
        # Print with safe encoding
        safe_print(progress_text, end="", flush=True)
    
    def increment(self, amount: int = 1) -> None:
        """Increment progress by amount."""
        self.update(self.current + amount)
    
    def finish(self) -> None:
        """Finish progress bar and print newline."""
        self.update(self.total, force_update=True)
        safe_print()  # New line
    
    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format."""
        if seconds < 0:
            return "--:--"
        
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        
        if minutes > 99:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"


class RealTimeProgressMonitor:
    """Real-time progress monitor for QA Generation Pipeline sessions."""
    
    def __init__(self, progress_manager):
        """
        Initialize real-time progress monitor.
        
        Args:
            progress_manager: ProgressManager instance
        """
        self.progress_manager = progress_manager
        self.monitoring = False
        self.monitor_thread = None
        self.session_progress_bars = {}
        self.update_interval = 1.0  # Update every second
        
    def start_monitoring(self, session_ids: List[str] = None, show_all: bool = False) -> None:
        """
        Start monitoring sessions in real-time.
        
        Args:
            session_ids: Specific session IDs to monitor (None for active sessions)
            show_all: Show all sessions including completed ones
        """
        if self.monitoring:
            safe_print("Monitor already running")
            return
        
        self.monitoring = True
        self.session_ids = session_ids
        self.show_all = show_all
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        safe_print("🚀 实时进度监控已启动")
        safe_print("按 Ctrl+C 停止监控\n")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Finish all progress bars
        for progress_bar in self.session_progress_bars.values():
            if progress_bar:
                progress_bar.finish()
        
        safe_print("\n📊 进度监控已停止")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        last_display_time = 0
        
        try:
            while self.monitoring:
                current_time = time.time()
                
                # Update display
                if current_time - last_display_time >= self.update_interval:
                    self._update_display()
                    last_display_time = current_time
                
                time.sleep(0.1)  # Small sleep to prevent high CPU usage
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            safe_print(f"监控错误: {e}")
    
    def _update_display(self) -> None:
        """Update the progress display."""
        # Get sessions to monitor
        if self.session_ids:
            sessions = []
            for session_id in self.session_ids:
                session = self.progress_manager.get_session_progress(session_id)
                if session:
                    sessions.append({"session_id": session_id, **session})
        else:
            sessions = self.progress_manager.list_sessions()
        
        # Filter sessions
        if not self.show_all:
            sessions = [s for s in sessions if s["status"] in ["running", "pending"]]
        
        # Update progress bars
        active_sessions = set()
        
        for session in sessions:
            session_id = session["session_id"]
            active_sessions.add(session_id)
            
            # Get detailed stats
            stats = self.progress_manager.get_session_stats(session_id)
            
            # Create or update progress bar
            if session_id not in self.session_progress_bars:
                description = f"{session['operation_type'][:20]:20s}"
                self.session_progress_bars[session_id] = ProgressBar(
                    total=stats['total_items'],
                    description=description,
                    width=40
                )
            
            progress_bar = self.session_progress_bars[session_id]
            if progress_bar:
                progress_bar.update(stats['completed_items'])
                
                # If session is completed, finish the progress bar
                if session["status"] in ["completed", "failed", "cancelled"]:
                    progress_bar.finish()
                    self.session_progress_bars[session_id] = None
        
        # Clean up finished sessions
        for session_id in list(self.session_progress_bars.keys()):
            if session_id not in active_sessions and self.session_progress_bars[session_id] is None:
                del self.session_progress_bars[session_id]
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a detailed summary of a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session summary with progress information
        """
        session = self.progress_manager.get_session_progress(session_id)
        if not session:
            return {}
        
        stats = self.progress_manager.get_session_stats(session_id)
        
        # Calculate additional metrics
        elapsed_seconds = 0
        if "start_time" in session:
            start_time = datetime.fromisoformat(session["start_time"])
            if session["status"] == "running":
                elapsed_seconds = (datetime.now() - start_time).total_seconds()
            elif "end_time" in session:
                end_time = datetime.fromisoformat(session["end_time"])
                elapsed_seconds = (end_time - start_time).total_seconds()
        
        # Calculate speed
        items_per_second = stats['completed_items'] / elapsed_seconds if elapsed_seconds > 0 else 0
        
        # Estimate remaining time
        remaining_items = stats['remaining_items']
        eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
        
        return {
            "session_id": session_id,
            "operation_type": session["operation_type"],
            "status": session["status"],
            "progress": {
                "total_items": stats['total_items'],
                "completed_items": stats['completed_items'],
                "failed_items": stats['failed_items'],
                "remaining_items": remaining_items,
                "completion_percentage": stats['completion_percentage'],
                "failure_percentage": stats['failure_percentage']
            },
            "timing": {
                "start_time": session.get("start_time"),
                "last_update": session.get("last_update"),
                "end_time": session.get("end_time"),
                "elapsed_seconds": elapsed_seconds,
                "items_per_second": items_per_second,
                "eta_seconds": eta_seconds
            },
            "errors": session.get("failed_files", [])[-5:] if session.get("failed_files") else []  # Last 5 errors
        }


class ProgressDisplayFormatter:
    """Formatter for pretty progress display."""
    
    @staticmethod
    def format_session_summary(summary: Dict[str, Any]) -> str:
        """
        Format session summary for display.
        
        Args:
            summary: Session summary from RealTimeProgressMonitor
            
        Returns:
            Formatted string
        """
        if not summary:
            return "Session not found"
        
        lines = []
        
        # Header
        lines.append(f"📋 会话: {summary['session_id']}")
        lines.append(f"🔧 操作类型: {summary['operation_type']}")
        lines.append(f"📊 状态: {summary['status']}")
        lines.append("")
        
        # Progress
        progress = summary['progress']
        lines.append("📈 进度信息:")
        lines.append(f"   总项目数: {progress['total_items']}")
        lines.append(f"   已完成: {progress['completed_items']}")
        lines.append(f"   失败: {progress['failed_items']}")
        lines.append(f"   剩余: {progress['remaining_items']}")
        lines.append(f"   完成率: {progress['completion_percentage']:.1f}%")
        
        if progress['failed_items'] > 0:
            lines.append(f"   失败率: {progress['failure_percentage']:.1f}%")
        
        lines.append("")
        
        # Timing
        timing = summary['timing']
        if timing['elapsed_seconds'] > 0:
            lines.append("⏱️ 时间信息:")
            lines.append(f"   开始时间: {timing['start_time']}")
            lines.append(f"   运行时长: {ProgressDisplayFormatter._format_duration(timing['elapsed_seconds'])}")
            
            if timing['items_per_second'] > 0:
                lines.append(f"   处理速度: {timing['items_per_second']:.2f} 项/秒")
                
                if timing['eta_seconds'] > 0:
                    lines.append(f"   预计完成: {ProgressDisplayFormatter._format_duration(timing['eta_seconds'])}")
            
            lines.append("")
        
        # Recent errors
        if summary['errors']:
            lines.append("❌ 最近错误:")
            for error in summary['errors'][-3:]:  # Show last 3 errors
                error_file = error.get('file', 'Unknown')
                error_msg = error.get('error', 'Unknown error')
                lines.append(f"   • {error_file}: {error_msg}")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_sessions_overview(sessions: List[Dict[str, Any]]) -> str:
        """
        Format multiple sessions overview.
        
        Args:
            sessions: List of session summaries
            
        Returns:
            Formatted overview string
        """
        if not sessions:
            return "没有找到会话"
        
        lines = []
        lines.append("📊 会话概览")
        lines.append("=" * 60)
        
        # Group by status
        running_sessions = [s for s in sessions if s['status'] == 'running']
        completed_sessions = [s for s in sessions if s['status'] == 'completed']
        failed_sessions = [s for s in sessions if s['status'] == 'failed']
        
        # Running sessions
        if running_sessions:
            lines.append("🔄 正在运行的会话:")
            for session in running_sessions:
                progress = session['progress']
                lines.append(f"   {session['session_id'][:20]:20s} | "
                           f"{progress['completion_percentage']:5.1f}% | "
                           f"{session['operation_type']}")
            lines.append("")
        
        # Completed sessions
        if completed_sessions:
            lines.append("✅ 已完成的会话:")
            for session in completed_sessions:
                progress = session['progress']
                lines.append(f"   {session['session_id'][:20]:20s} | "
                           f"{progress['completed_items']:4d}/{progress['total_items']:4d} | "
                           f"{session['operation_type']}")
            lines.append("")
        
        # Failed sessions
        if failed_sessions:
            lines.append("❌ 失败的会话:")
            for session in failed_sessions:
                progress = session['progress']
                lines.append(f"   {session['session_id'][:20]:20s} | "
                           f"{progress['completed_items']:4d}/{progress['total_items']:4d} | "
                           f"{session['operation_type']}")
            lines.append("")
        
        # Summary
        lines.append(f"总计: {len(sessions)} 个会话 "
                    f"(运行中: {len(running_sessions)}, "
                    f"已完成: {len(completed_sessions)}, "
                    f"失败: {len(failed_sessions)})")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"


# Convenience functions for easy use
def create_progress_bar(total: int, description: str = "") -> ProgressBar:
    """Create a new progress bar."""
    return ProgressBar(total, description)


def monitor_sessions_realtime(progress_manager, session_ids: List[str] = None):
    """Start real-time monitoring of sessions."""
    monitor = RealTimeProgressMonitor(progress_manager)
    try:
        monitor.start_monitoring(session_ids)
        # Keep monitoring until interrupted
        while monitor.monitoring:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()


def display_session_summary(progress_manager, session_id: str) -> None:
    """Display detailed summary of a session."""
    monitor = RealTimeProgressMonitor(progress_manager)
    summary = monitor.get_session_summary(session_id)
    formatted = ProgressDisplayFormatter.format_session_summary(summary)
    safe_print(formatted)


def display_sessions_overview(progress_manager) -> None:
    """Display overview of all sessions."""
    monitor = RealTimeProgressMonitor(progress_manager)
    sessions = progress_manager.list_sessions()
    
    # Get detailed summaries
    summaries = []
    for session in sessions:
        summary = monitor.get_session_summary(session['session_id'])
        if summary:
            summaries.append(summary)
    
    formatted = ProgressDisplayFormatter.format_sessions_overview(summaries)
    safe_print(formatted) 