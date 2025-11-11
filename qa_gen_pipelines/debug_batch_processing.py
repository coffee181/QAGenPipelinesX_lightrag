#!/usr/bin/env python3
"""
批量处理调试工具

用于调试批量导入时的日志和进度问题
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_utils import setup_project_logging, UTF8Logger
from src.utils.config import ConfigManager
from src.services.progress_manager import ProgressManager
from src.services.pdf_processor import PDFProcessor

def test_logging_flush():
    """测试日志刷新功能"""
    print("🔍 测试日志刷新功能...")
    
    logger = setup_project_logging("DEBUG")
    
    for i in range(5):
        logger.info(f"测试日志消息 {i+1}")
        logger.debug(f"调试消息 {i+1}")
        logger.warning(f"警告消息 {i+1}")
        
        # 强制刷新
        UTF8Logger.force_flush_logs()
        
        print(f"已发送日志消息 {i+1}")
        time.sleep(1)
    
    print("✅ 日志刷新测试完成")

def test_progress_manager():
    """测试进度管理器"""
    print("🔍 测试进度管理器...")
    
    config = ConfigManager("config.yaml")
    progress_manager = ProgressManager(config)
    
    # 创建测试会话
    session_id = f"test_session_{int(time.time())}"
    progress_manager.create_session(
        session_id=session_id,
        operation_type="test_processing",
        total_items=5,
        metadata={"test": True}
    )
    
    print(f"创建测试会话: {session_id}")
    
    # 模拟处理进度
    for i in range(5):
        progress_manager.update_progress(session_id, 1)
        print(f"更新进度: {i+1}/5")
        time.sleep(1)
    
    # 完成会话
    progress_manager.complete_session(session_id, "completed")
    
    # 获取统计信息
    stats = progress_manager.get_session_stats(session_id)
    print(f"会话统计: {stats}")
    
    print("✅ 进度管理器测试完成")

def test_pdf_processor():
    """测试PDF处理器"""
    print("🔍 测试PDF处理器...")
    
    config = ConfigManager("config.yaml")
    progress_manager = ProgressManager(config)
    
    # 创建PDF处理器
    pdf_processor = PDFProcessor(config, progress_manager)
    
    # 检查是否有测试PDF文件
    test_pdf_dir = Path("example_pdfs")
    if not test_pdf_dir.exists():
        print("⚠️  未找到测试PDF目录，跳过PDF处理器测试")
        return
    
    pdf_files = list(test_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  测试PDF目录中没有PDF文件，跳过PDF处理器测试")
        return
    
    print(f"找到 {len(pdf_files)} 个测试PDF文件")
    
    # 创建测试会话
    session_id = f"pdf_test_{int(time.time())}"
    
    # 处理第一个PDF文件
    test_pdf = pdf_files[0]
    print(f"处理测试PDF: {test_pdf.name}")
    
    try:
        document = pdf_processor.process_single_pdf(test_pdf, session_id)
        if document:
            print(f"✅ PDF处理成功: {document.name}")
            print(f"   文本长度: {len(document.content)} 字符")
        else:
            print("❌ PDF处理失败")
    except Exception as e:
        print(f"❌ PDF处理异常: {e}")
    
    print("✅ PDF处理器测试完成")

def test_batch_processing():
    """测试批量处理"""
    print("🔍 测试批量处理...")
    
    config = ConfigManager("config.yaml")
    progress_manager = ProgressManager(config)
    pdf_processor = PDFProcessor(config, progress_manager)
    
    # 检查测试目录
    test_pdf_dir = Path("example_pdfs")
    if not test_pdf_dir.exists():
        print("⚠️  未找到测试PDF目录，跳过批量处理测试")
        return
    
    pdf_files = list(test_pdf_dir.glob("*.pdf"))
    if len(pdf_files) < 2:
        print("⚠️  测试PDF文件不足，跳过批量处理测试")
        return
    
    print(f"开始批量处理 {len(pdf_files)} 个PDF文件")
    
    # 创建测试会话
    session_id = f"batch_test_{int(time.time())}"
    
    try:
        # 使用前2个文件进行测试
        test_files = pdf_files[:2]
        documents = pdf_processor.process_pdf_list(test_files, resume_session=False)
        
        print(f"✅ 批量处理完成: {len(documents)} 个文档")
        for doc in documents:
            print(f"   - {doc.name}: {len(doc.content)} 字符")
            
    except Exception as e:
        print(f"❌ 批量处理异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ 批量处理测试完成")

def main():
    """主函数"""
    print("🚀 开始批量处理调试测试")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 测试日志刷新
        test_logging_flush()
        print()
        
        # 测试进度管理器
        test_progress_manager()
        print()
        
        # 测试PDF处理器
        test_pdf_processor()
        print()
        
        # 测试批量处理
        test_batch_processing()
        print()
        
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎉 批量处理调试测试完成")

if __name__ == "__main__":
    main()
