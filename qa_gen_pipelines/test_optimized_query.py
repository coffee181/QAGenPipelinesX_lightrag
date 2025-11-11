#!/usr/bin/env python3
"""Test script for optimized LightRAG query performance."""

import time
import json
from pathlib import Path
from src.utils.config import ConfigManager
from src.implementations.lightrag_rag import LightRAGImplementation

def test_query_performance():
    """Test query performance with a few sample questions."""
    
    # Load configuration
    config = ConfigManager("config.yaml")
    
    # Initialize LightRAG
    rag = LightRAGImplementation(config)
    
    # Use existing knowledge base
    working_dir = Path("D:/Project/lightrag/selection_workdir")
    rag.use_existing_knowledge_base(working_dir)
    
    # Sample questions for testing
    test_questions = [
        "如何在GSK 27i数控系统初次安装时正确配置多通道参数？",
        "当系统出现伺服驱动单元GR/GM系列总线通信异常时，应如何诊断？",
        "在刚性攻丝加工过程中出现跟随误差报警，如何调整参数？"
    ]
    
    print("开始测试查询性能...")
    print(f"测试问题数量: {len(test_questions)}")
    print("-" * 50)
    
    total_time = 0
    successful_queries = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question[:50]}...")
        
        start_time = time.time()
        try:
            answer = rag.query_single_question(question)
            end_time = time.time()
            
            query_time = end_time - start_time
            total_time += query_time
            successful_queries += 1
            
            print(f"✓ 查询成功 - 耗时: {query_time:.2f}秒")
            print(f"  答案长度: {len(answer)} 字符")
            print(f"  答案预览: {answer[:100]}...")
            
        except Exception as e:
            end_time = time.time()
            query_time = end_time - start_time
            total_time += query_time
            
            print(f"✗ 查询失败 - 耗时: {query_time:.2f}秒")
            print(f"  错误: {str(e)}")
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print(f"成功查询: {successful_queries}/{len(test_questions)}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每个问题耗时: {total_time/len(test_questions):.2f}秒")
    
    if successful_queries > 0:
        avg_successful_time = total_time / successful_queries
        print(f"成功查询平均耗时: {avg_successful_time:.2f}秒")
        
        # Estimate total time for 680 questions
        estimated_total = avg_successful_time * 680
        print(f"预计处理680个问题需要: {estimated_total/60:.1f}分钟 ({estimated_total/3600:.1f}小时)")

if __name__ == "__main__":
    test_query_performance() 