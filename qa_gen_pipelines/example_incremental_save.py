#!/usr/bin/env python3
"""
示例脚本：演示增量保存和程序中断恢复功能

此脚本展示了如何使用QA生成管道的增量保存功能，
包括程序中断时的自动保存和从中断点恢复的能力。
"""

import sys
import signal
import time
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config import ConfigManager
from src.services.progress_manager import ProgressManager
from src.services.answer_service import AnswerService
from src.implementations.lightrag_rag import LightRAGImplementation
from src.implementations.simple_markdown_processor import SimpleMarkdownProcessor


def simulate_answer_generation_with_interruption():
    """模拟答案生成过程，并展示中断和恢复功能"""
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 加载配置
    config = ConfigManager("config.yaml")
    
    # 创建服务
    progress_manager = ProgressManager(config)
    rag = LightRAGImplementation(config)
    markdown_processor = SimpleMarkdownProcessor()
    
    answer_service = AnswerService(
        rag=rag,
        markdown_processor=markdown_processor,
        progress_manager=progress_manager,
        logger=logger
    )
    
    # 设置路径
    questions_file = Path("test_questions.jsonl")
    working_dir = Path("test_kb")
    output_file = Path("test_incremental_output.jsonl")
    session_id = "incremental_test_session"
    
    logger.info("=" * 60)
    logger.info("增量保存和恢复功能演示")
    logger.info("=" * 60)
    
    # 第一次运行 - 正常生成（可能被中断）
    logger.info("1. 开始生成答案（可以按 Ctrl+C 中断）...")
    try:
        # 设置知识库
        if working_dir.exists():
            # 使用现有知识库
            answer_service.rag.use_existing_knowledge_base(working_dir)
            logger.info(f"使用现有知识库: {working_dir}")
        else:
            logger.warning(f"知识库不存在: {working_dir}，请先创建知识库")
            return
        
        # 生成答案
        result = answer_service.generate_answers_from_existing_kb(
            questions_file=questions_file,
            working_dir=working_dir,
            output_file=output_file,
            session_id=session_id,
            resume=False  # 第一次运行不使用恢复（显式指定）
        )
        
        logger.info(f"完成生成 {len(result.qa_pairs)} 个QA对")
        
    except KeyboardInterrupt:
        logger.warning("检测到中断信号，正在保存进度...")
        # AnswerService会自动保存当前进度
        logger.info("进度已保存，可以稍后恢复")
        return
    except Exception as e:
        logger.error(f"生成过程中出错: {e}")
        return
    
    logger.info("演示完成！")


def resume_answer_generation():
    """恢复被中断的答案生成过程"""
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 加载配置
    config = ConfigManager("config.yaml")
    
    # 创建服务
    progress_manager = ProgressManager(config)
    rag = LightRAGImplementation(config)
    markdown_processor = SimpleMarkdownProcessor()
    
    answer_service = AnswerService(
        rag=rag,
        markdown_processor=markdown_processor,
        progress_manager=progress_manager,
        logger=logger
    )
    
    # 设置路径
    questions_file = Path("test_questions.jsonl")
    working_dir = Path("test_kb")
    output_file = Path("test_incremental_output.jsonl")
    session_id = "incremental_test_session"
    
    logger.info("=" * 60)
    logger.info("恢复被中断的答案生成")
    logger.info("=" * 60)
    
    # 检查是否有未完成的进度
    if not output_file.exists():
        logger.warning("没有发现中断的进度文件")
        return
    
    logger.info(f"发现进度文件: {output_file}")
    
    try:
        # 设置知识库
        answer_service.rag.use_existing_knowledge_base(working_dir)
        
        # 恢复生成答案（现在是默认行为）
        result = answer_service.generate_answers_from_existing_kb(
            questions_file=questions_file,
            working_dir=working_dir,
            output_file=output_file,
            session_id=session_id
            # resume=True 现在是默认值，无需显式指定
        )
        
        logger.info(f"恢复完成！总共生成 {len(result.qa_pairs)} 个QA对")
        
        # 显示进度统计
        stats = progress_manager.get_session_stats(session_id)
        if stats:
            logger.info("会话统计:")
            logger.info(f"  总项目数: {stats['total_items']}")
            logger.info(f"  已完成: {stats['completed_items']}")
            logger.info(f"  失败: {stats['failed_items']}")
            logger.info(f"  完成百分比: {stats['completion_percentage']:.2f}%")
        
    except Exception as e:
        logger.error(f"恢复过程中出错: {e}")


def show_progress():
    """显示当前进度"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 加载配置
    config = ConfigManager("config.yaml")
    progress_manager = ProgressManager(config)
    
    logger.info("=" * 60)
    logger.info("当前进度状态")
    logger.info("=" * 60)
    
    # 列出所有会话
    sessions = progress_manager.list_sessions()
    
    if not sessions:
        logger.info("没有找到任何会话")
        return
    
    for session in sessions:
        session_id = session.get('session_id') if isinstance(session, dict) else session.session_id
        stats = progress_manager.get_session_stats(session_id)
        
        logger.info(f"会话: {session_id}")
        logger.info(f"  操作类型: {stats.get('operation_type', 'unknown')}")
        logger.info(f"  状态: {stats.get('status', 'unknown')}")
        logger.info(f"  进度: {stats.get('completed_items', 0)}/{stats.get('total_items', 0)} "
                   f"({stats.get('completion_percentage', 0):.1f}%)")
        logger.info(f"  开始时间: {stats.get('start_time', 'unknown')}")
        logger.info(f"  最后更新: {stats.get('last_update', 'unknown')}")
        logger.info("-" * 40)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "generate":
            simulate_answer_generation_with_interruption()
        elif command == "resume":
            resume_answer_generation()
        elif command == "progress":
            show_progress()
        else:
            print("使用方法:")
            print("  python example_incremental_save.py generate   # 开始新的生成任务")
            print("  python example_incremental_save.py resume    # 恢复被中断的任务")
            print("  python example_incremental_save.py progress  # 显示当前进度")
    else:
        print("增量保存和恢复功能演示")
        print()
        print("使用方法:")
        print("  python example_incremental_save.py generate   # 开始新的生成任务")
        print("  python example_incremental_save.py resume    # 恢复被中断的任务")
        print("  python example_incremental_save.py progress  # 显示当前进度")
        print()
        print("功能特点:")
        print("- 每生成5个QA对自动保存一次")
        print("- 程序中断时自动保存当前进度")
        print("- 支持从中断点恢复继续生成")
        print("- 实时进度跟踪和统计") 