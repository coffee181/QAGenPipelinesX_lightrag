#!/usr/bin/env python3
"""测试参考资料清理功能"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.implementations.simple_markdown_processor import SimpleMarkdownProcessor

def test_references_cleanup():
    """测试参考资料清理功能"""
    print("🧪 测试参考资料清理功能...")
    
    # 创建处理器实例
    processor = SimpleMarkdownProcessor()
    
    # 测试文本包含各种参考资料格式
    test_text = """
    根据文档内容，数控系统是一种计算机控制系统，主要用于控制机床的加工过程。
    
    数控系统具有以下特点：
    1. 高精度
    2. 高效率
    3. 自动化程度高
    
    **参考资料**
    [DC1] 数控系统技术手册.pdf
    [DC2] 机床操作指南.pdf
    
    **References**
    [KG1] 数控系统实体
    [KG2] 机床设备实体
    
    参考资料：
    - 技术文档1
    - 技术文档2
    
    参考：
    见[DC3]相关章节
    """
    
    # 测试清理功能
    cleaned_text = processor.clean_llm_response(test_text)
    
    print("原始文本:")
    print(test_text)
    print("\n清理后文本:")
    print(cleaned_text)
    
    # 验证参考资料被移除
    assert "参考资料" not in cleaned_text, "❌ 参考资料部分没有被清理"
    assert "References" not in cleaned_text, "❌ References部分没有被清理"
    assert "[DC1]" not in cleaned_text, "❌ [DC1]引用没有被清理"
    assert "[DC2]" not in cleaned_text, "❌ [DC2]引用没有被清理"
    assert "[KG1]" not in cleaned_text, "❌ [KG1]引用没有被清理"
    assert "[KG2]" not in cleaned_text, "❌ [KG2]引用没有被清理"
    assert "见[DC3]" not in cleaned_text, "❌ 见[DC3]引用没有被清理"
    
    # 验证主要内容被保留
    assert "数控系统是一种计算机控制系统" in cleaned_text, "❌ 主要内容被错误清理"
    assert "高精度" in cleaned_text, "❌ 列表内容被错误清理"
    assert "高效率" in cleaned_text, "❌ 列表内容被错误清理"
    assert "自动化程度高" in cleaned_text, "❌ 列表内容被错误清理"
    
    print("✅ 参考资料清理测试通过")

def test_chinese_references_cleanup():
    """测试中文参考资料清理"""
    print("\n🧪 测试中文参考资料清理...")
    
    # 创建处理器实例
    processor = SimpleMarkdownProcessor()
    
    # 测试文本包含中文参考资料
    test_text = """
    根据文档内容，AE-100的PWM控制信号频率设置方法如下：
    
    1. 确定PWM频率范围
    2. 使用寄存器配置频率
    3. 计算具体数值
    4. 配置占空比和相位
    5. 测试和验证
    
    参考资料：
    根据文档内容，AE-100的默认频率通常为1.58kHz，但可以调整到其他值以满足特定要求。
    
    参考：
    见[DC1]相关章节，G2 No.5寄存器用于指定PWM频率。
    """
    
    # 测试清理功能
    cleaned_text = processor.clean_llm_response(test_text)
    
    print("原始文本:")
    print(test_text)
    print("\n清理后文本:")
    print(cleaned_text)
    
    # 验证参考资料被移除
    assert "参考资料：" not in cleaned_text, "❌ 参考资料部分没有被清理"
    assert "参考：" not in cleaned_text, "❌ 参考部分没有被清理"
    assert "见[DC1]" not in cleaned_text, "❌ 见[DC1]引用没有被清理"
    
    # 验证主要内容被保留
    assert "AE-100的PWM控制信号频率设置方法" in cleaned_text, "❌ 主要内容被错误清理"
    assert "确定PWM频率范围" in cleaned_text, "❌ 列表内容被错误清理"
    assert "使用寄存器配置频率" in cleaned_text, "❌ 列表内容被错误清理"
    
    print("✅ 中文参考资料清理测试通过")

if __name__ == "__main__":
    try:
        test_references_cleanup()
        test_chinese_references_cleanup()
        print("\n🎉 所有参考资料清理测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
