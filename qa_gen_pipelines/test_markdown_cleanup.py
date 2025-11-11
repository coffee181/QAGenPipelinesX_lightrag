#!/usr/bin/env python3
"""测试markdown处理的换行清理功能"""

from src.implementations.simple_markdown_processor import SimpleMarkdownProcessor

def test_newline_cleanup():
    processor = SimpleMarkdownProcessor()
    
    # 测试用户提供的包含\n的文本
    test_text = """系统诊断页面监控与反向间隙补偿操作指南\n进入诊断页面\n按 L2SN 键进入诊断页面集，通过软键切换至 系统诊断页面。该页面提供键盘诊断、轴状态监控等功能（见[DC3]）。\n监控反向间隙\n在诊断页面中，可通过以下步骤监控轴状态：\n使用 位移偏差监视 功能（按软键 位移偏差），实时显示各轴的位移偏差波形（见[DC2]）。\n结合 同步性监视（软键 同步性）检查多轴运动的同步误差（见[DC2]）。"""
    
    print("原始文本:")
    print(repr(test_text))
    print("\n原始显示:")
    print(test_text)
    
    print("\n" + "="*60)
    
    result = processor.clean_llm_response(test_text)
    
    print("清理后的文本:")
    print(repr(result))
    print("\n清理后显示:")
    print(result)
    
    print("\n" + "="*60)
    print("清理效果总结:")
    print(f"原始长度: {len(test_text)} 字符")
    print(f"清理后长度: {len(result)} 字符")
    newline_count = test_text.count('\\n')
    print(f"去除了 {newline_count} 个 \\n 字符")
    ref_count = len([ref for ref in ['[DC1]', '[DC2]', '[DC3]'] if ref in test_text])
    print(f"去除了 {ref_count} 个参考文献标记")

if __name__ == "__main__":
    test_newline_cleanup() 