#!/usr/bin/env python3
"""测试deepseek-r1:32b模型"""

import requests
import json
from pathlib import Path

def test_ollama_connection():
    """测试Ollama连接"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("✓ Ollama服务连接成功")
            print("已安装的模型:")
            for model in models.get("models", []):
                print(f"  - {model['name']}")
            return True
        else:
            print(f"✗ Ollama服务连接失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 无法连接到Ollama服务: {e}")
        return False

def test_deepseek_r1_32b():
    """测试deepseek-r1:32b模型"""
    print("\n=== 测试deepseek-r1:32b模型 ===")
    
    # 测试提示词
    test_prompt = """你是一个专业的问答对生成助手。
你的任务是严格地、仅仅地根据用户在 <Context> 标签中提供的文本内容，生成高质量的问题。
确保每个问题的答案都能在提供的 <Context> 文本中直接找到或明确推断出来。
不要使用任何 <Context> 之外的知识。

这是你需要分析的文本内容：
<Context>
数控机床是一种高精度、高效率的自动化机床。它通过数字控制系统来控制机床的运动，
能够加工各种复杂形状的零件。数控机床的主要组成部分包括：
1. 机床本体：提供机械支撑和运动机构
2. 数控系统：控制机床运动的电子系统
3. 伺服系统：执行数控系统指令的驱动系统
4. 检测系统：检测机床位置和状态的反馈系统

数控机床具有加工精度高、生产效率高、适应性强等优点，广泛应用于航空航天、
汽车制造、模具加工等领域。
</Context>

请严格按照以下要求执行：
1. 根据上面的 <Context> 内容，生成5个高质量的、相关的专业问题。
2. 每个问题必须以"问题N："开头（N为从1开始的数字）。
3. 问题应该由浅入深，覆盖文本中的核心概念和关键信息。
4. 不要生成与 <Context> 无关的问题。
5. 不要生成答案，只生成问题。"""

    try:
        payload = {
            "model": "deepseek-r1:32b",
            "prompt": test_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 2048
            }
        }

        print("正在调用deepseek-r1:32b模型...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120  # 32B模型可能需要更长时间
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            
            print("✓ 模型调用成功!")
            print(f"响应长度: {len(response_text)} 字符")
            print("\n模型响应:")
            print("-" * 50)
            print(response_text)
            print("-" * 50)
            
            # 检查是否生成了问题
            if "问题1：" in response_text and "问题2：" in response_text:
                print("\n✓ 成功生成了结构化问题!")
                return True
            else:
                print("\n⚠ 响应格式可能不符合预期")
                return False
        else:
            print(f"✗ 模型调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("✗ 模型调用超时（可能需要更长时间）")
        return False
    except Exception as e:
        print(f"✗ 模型调用异常: {e}")
        return False

def check_model_availability():
    """检查模型是否可用"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            for model in models.get("models", []):
                if "deepseek-r1:32b" in model["name"]:
                    print("✓ deepseek-r1:32b模型已安装")
                    return True
            print("✗ deepseek-r1:32b模型未找到")
            print("请先运行: ollama pull deepseek-r1:32b")
            return False
    except:
        return False

def main():
    """主函数"""
    print("=== deepseek-r1:32b 模型测试 ===")
    
    # 1. 测试Ollama连接
    if not test_ollama_connection():
        print("\n请确保Ollama服务正在运行:")
        print("ollama serve")
        return
    
    # 2. 检查模型是否可用
    if not check_model_availability():
        return
    
    # 3. 测试模型
    if test_deepseek_r1_32b():
        print("\n🎉 恭喜! deepseek-r1:32b模型工作正常!")
        print("\n现在您可以:")
        print("1. 修改config.yaml，添加本地模型配置")
        print("2. 更新main.py使用本地模型")
        print("3. 享受免费的本地AI服务!")
    else:
        print("\n❌ 模型测试失败")
        print("请检查:")
        print("1. 模型是否正确下载")
        print("2. GPU内存是否充足")
        print("3. Ollama服务是否正常")

if __name__ == "__main__":
    main()
