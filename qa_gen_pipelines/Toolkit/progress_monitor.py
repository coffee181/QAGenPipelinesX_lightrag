#!/usr/bin/env python3
"""
独立的进度监控工具
用于实时监控QA生成数据管道的执行进度

使用方法:
    python Toolkit/progress_monitor.py                    # 监控所有活动会话
    python Toolkit/progress_monitor.py --session my_id    # 监控特定会话
    python Toolkit/progress_monitor.py --all              # 监控所有会话
    python Toolkit/progress_monitor.py --refresh 0.5      # 自定义刷新频率
"""

import sys
import argparse
import time
import signal
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import ConfigManager
from src.services.progress_manager import ProgressManager
from src.utils.progress_display import (
    RealTimeProgressMonitor,
    ProgressDisplayFormatter,
    create_progress_bar
)
from src.utils.console_utils import safe_print


class ProgressMonitorTool:
    """独立的进度监控工具"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化监控工具
        
        Args:
            config_path: 配置文件路径
        """
        try:
            self.config = ConfigManager(config_path)
            self.progress_manager = ProgressManager(self.config)
            self.monitor = RealTimeProgressMonitor(self.progress_manager)
            self.running = False
            
            # 注册信号处理器
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
        except Exception as e:
            safe_print(f"❌ 初始化失败: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """信号处理器，优雅地停止监控"""
        safe_print("\n🛑 收到停止信号，正在停止监控...")
        self.running = False
    
    def show_current_status(self):
        """显示当前状态概览"""
        safe_print("📊 QA生成管道 - 当前状态")
        safe_print("=" * 60)
        
        sessions = self.progress_manager.list_sessions()
        
        if not sessions:
            safe_print("📋 没有找到任何会话")
            return
        
        # 统计不同状态的会话
        running_count = len([s for s in sessions if s['status'] == 'running'])
        completed_count = len([s for s in sessions if s['status'] == 'completed'])
        failed_count = len([s for s in sessions if s['status'] == 'failed'])
        
        # 显示统计
        safe_print(f"📈 会话统计:")
        safe_print(f"   总数: {len(sessions)}")
        safe_print(f"   运行中: {running_count}")
        safe_print(f"   已完成: {completed_count}")
        safe_print(f"   失败: {failed_count}")
        safe_print("")
        
        # 显示活动会话的详细信息
        if running_count > 0:
            safe_print("🔄 活动会话:")
            for session in sessions:
                if session['status'] == 'running':
                    stats = self.progress_manager.get_session_stats(session['session_id'])
                    percentage = stats['completion_percentage']
                    
                    # 创建小型进度条
                    bar_width = 20
                    filled = int(bar_width * percentage / 100)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    
                    safe_print(f"   {session['session_id'][:20]:20s} |{bar}| {percentage:5.1f}%")
            safe_print("")
    
    def show_session_detail(self, session_id: str):
        """显示指定会话的详细信息"""
        summary = self.monitor.get_session_summary(session_id)
        
        if not summary:
            safe_print(f"❌ 会话 {session_id} 不存在")
            return
        
        formatted = ProgressDisplayFormatter.format_session_summary(summary)
        safe_print(formatted)
    
    def start_realtime_monitoring(self, session_ids=None, show_all=False, refresh_interval=1.0):
        """启动实时监控"""
        safe_print("🚀 启动实时进度监控")
        safe_print(f"⏱️ 刷新间隔: {refresh_interval}秒")
        safe_print("📋 按 Ctrl+C 停止监控\n")
        
        self.running = True
        last_update = 0
        
        try:
            while self.running:
                current_time = time.time()
                
                # 按指定间隔更新显示
                if current_time - last_update >= refresh_interval:
                    self._update_realtime_display(session_ids, show_all)
                    last_update = current_time
                
                time.sleep(0.1)  # 防止CPU使用率过高
                
        except KeyboardInterrupt:
            pass
        finally:
            safe_print("\n📊 实时监控已停止")
    
    def _update_realtime_display(self, session_ids=None, show_all=False):
        """更新实时显示"""
        # 清屏 (可选)
        # import os
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        # 获取要监控的会话
        if session_ids:
            sessions = []
            for session_id in session_ids:
                session = self.progress_manager.get_session_progress(session_id)
                if session:
                    sessions.append({"session_id": session_id, **session})
        else:
            sessions = self.progress_manager.list_sessions()
        
        # 过滤会话
        if not show_all:
            sessions = [s for s in sessions if s["status"] in ["running", "pending"]]
        
        if not sessions:
            safe_print("📋 没有活动会话")
            return
        
        # 显示时间戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        safe_print(f"📊 实时进度监控 - {timestamp}")
        safe_print("=" * 60)
        
        # 显示每个会话的进度
        for session in sessions:
            session_id = session["session_id"]
            stats = self.progress_manager.get_session_stats(session_id)
            
            # 基本信息
            operation_type = session["operation_type"]
            status = session["status"]
            percentage = stats['completion_percentage']
            completed = stats['completed_items']
            total = stats['total_items']
            
            # 创建进度条
            bar_width = 30
            filled = int(bar_width * percentage / 100) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            
            # 计算速度和ETA
            if session["status"] == "running" and completed > 0:
                from datetime import datetime
                start_time = datetime.fromisoformat(session["start_time"])
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = completed / elapsed if elapsed > 0 else 0
                
                if speed > 0:
                    remaining_items = total - completed
                    eta_seconds = remaining_items / speed
                    eta_str = self._format_time(eta_seconds)
                    speed_str = f"{speed:.1f} 项/秒"
                else:
                    eta_str = "--:--"
                    speed_str = "-- 项/秒"
            else:
                eta_str = "--:--"
                speed_str = "-- 项/秒"
            
            # 显示会话信息
            safe_print(f"🔄 {session_id}")
            safe_print(f"   类型: {operation_type} | 状态: {status}")
            safe_print(f"   |{bar}| {percentage:5.1f}% ({completed}/{total})")
            safe_print(f"   速度: {speed_str} | ETA: {eta_str}")
            
            # 显示错误信息
            if stats['failed_items'] > 0:
                safe_print(f"   ❌ 失败: {stats['failed_items']} 项")
            
            safe_print("")
        
        safe_print("按 Ctrl+C 停止监控")
    
    def _format_time(self, seconds):
        """格式化时间显示"""
        if seconds <= 0:
            return "--:--"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def export_progress_report(self, output_file="progress_report.txt"):
        """导出进度报告"""
        safe_print(f"📄 导出进度报告到: {output_file}")
        
        sessions = self.progress_manager.list_sessions()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("QA生成管道 - 进度报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"报告时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if not sessions:
                f.write("没有找到任何会话\n")
                return
            
            # 总体统计
            running_count = len([s for s in sessions if s['status'] == 'running'])
            completed_count = len([s for s in sessions if s['status'] == 'completed'])
            failed_count = len([s for s in sessions if s['status'] == 'failed'])
            
            f.write("总体统计:\n")
            f.write(f"  总会话数: {len(sessions)}\n")
            f.write(f"  运行中: {running_count}\n")
            f.write(f"  已完成: {completed_count}\n")
            f.write(f"  失败: {failed_count}\n\n")
            
            # 详细会话信息
            f.write("详细会话信息:\n")
            f.write("-" * 50 + "\n")
            
            for session in sessions:
                session_id = session['session_id']
                summary = self.monitor.get_session_summary(session_id)
                
                if summary:
                    formatted = ProgressDisplayFormatter.format_session_summary(summary)
                    f.write(formatted)
                    f.write("\n" + "-" * 50 + "\n")
        
        safe_print(f"✅ 报告已导出到: {output_file}")


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="QA生成管道 - 实时进度监控工具")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--session", help="监控特定会话ID")
    parser.add_argument("--all", action="store_true", help="显示所有会话（包括已完成的）")
    parser.add_argument("--monitor", action="store_true", help="启动实时监控模式")
    parser.add_argument("--refresh", type=float, default=1.0, help="实时监控刷新间隔（秒）")
    parser.add_argument("--detail", help="显示指定会话的详细信息")
    parser.add_argument("--export", help="导出进度报告到文件")
    
    args = parser.parse_args()
    
    # 创建监控工具实例
    tool = ProgressMonitorTool(args.config)
    
    try:
        if args.detail:
            # 显示指定会话详情
            tool.show_session_detail(args.detail)
        
        elif args.export:
            # 导出进度报告
            tool.export_progress_report(args.export)
        
        elif args.monitor:
            # 启动实时监控
            session_ids = [args.session] if args.session else None
            tool.start_realtime_monitoring(session_ids, args.all, args.refresh)
        
        else:
            # 显示当前状态
            tool.show_current_status()
            
            # 如果有指定会话，显示其详细信息
            if args.session:
                safe_print("")
                tool.show_session_detail(args.session)
    
    except Exception as e:
        safe_print(f"❌ 运行错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 