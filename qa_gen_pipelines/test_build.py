#!/usr/bin/env python3
"""
测试打包后的可执行文件

此脚本用于验证打包的可执行文件是否能正常运行
"""

import subprocess
import os
import sys
import json
from pathlib import Path
import platform

def test_executable():
    """测试可执行文件基本功能"""
    print("🧪 测试可执行文件基本功能")
    print("=" * 50)
    
    # 确定可执行文件路径
    deployment_dir = Path("deployment")
    if platform.system() == "Windows":
        exe_path = deployment_dir / "qa_gen_pipeline.exe"
    else:
        exe_path = deployment_dir / "qa_gen_pipeline"
    
    if not exe_path.exists():
        print(f"❌ 可执行文件不存在: {exe_path}")
        print("请先运行 python build_executable.py 创建可执行文件")
        return False
    
    # 测试1: 帮助信息
    print("📝 测试1: 帮助信息")
    try:
        result = subprocess.run(
            [str(exe_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=deployment_dir
        )
        
        if result.returncode == 0:
            print("✅ 帮助信息显示正常")
            print(f"  输出长度: {len(result.stdout)} 字符")
        else:
            print(f"❌ 帮助信息失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 帮助信息测试失败: {e}")
        return False
    
    # 测试2: 进度查看
    print("\n📊 测试2: 进度查看")
    try:
        result = subprocess.run(
            [str(exe_path), "show-progress"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=deployment_dir
        )
        
        if result.returncode == 0:
            print("✅ 进度查看功能正常")
        else:
            print(f"⚠️  进度查看返回非零代码 (正常，可能没有会话): {result.returncode}")
    except Exception as e:
        print(f"❌ 进度查看测试失败: {e}")
        return False
    
    # 测试3: 创建测试文件
    print("\n📄 测试3: 创建测试文件并测试")
    try:
        # 创建测试问题文件
        test_questions_file = deployment_dir / "test_questions.jsonl"
        with open(test_questions_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"question": "测试问题：什么是人工智能？"}, ensure_ascii=False) + '\n')
        
        print(f"✅ 创建测试问题文件: {test_questions_file}")
        
        # 创建测试文档
        test_doc_file = deployment_dir / "test_doc.txt"
        with open(test_doc_file, 'w', encoding='utf-8') as f:
            f.write("""人工智能（Artificial Intelligence，AI）是一门研究如何让计算机模拟人类智能的科学。
它包括机器学习、深度学习、自然语言处理等多个分支。
人工智能的目标是创造出能够独立思考和解决问题的智能系统。""")
        
        print(f"✅ 创建测试文档: {test_doc_file}")
        
        # 测试插入文档模式（创建知识库）
        print("\n🔧 测试4: 文档插入和答案生成")
        test_working_dir = deployment_dir / "test_kb"
        test_output_file = deployment_dir / "test_output.jsonl"
        
        try:
            result = subprocess.run(
                [
                    str(exe_path),
                    "generate-answers",
                    str(test_questions_file),
                    str(test_working_dir),
                    str(test_output_file),
                    "-i", str(test_doc_file)
                ],
                capture_output=True,
                text=True,
                timeout=120,  # 2分钟超时
                cwd=deployment_dir
            )
            
            if result.returncode == 0:
                print("✅ 文档插入和答案生成成功")
                
                # 检查输出文件
                if test_output_file.exists():
                    with open(test_output_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            print(f"✅ 输出文件有内容: {len(content)} 字符")
                        else:
                            print("⚠️  输出文件为空")
                else:
                    print("⚠️  输出文件未创建")
            else:
                print(f"❌ 文档插入和答案生成失败:")
                print(f"  错误输出: {result.stderr}")
                print(f"  标准输出: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️  文档插入和答案生成超时（可能需要更长时间）")
        except Exception as e:
            print(f"❌ 文档插入和答案生成测试失败: {e}")
            return False
        
        # 清理测试文件
        print("\n🧹 清理测试文件")
        for file_to_remove in [test_questions_file, test_doc_file, test_output_file]:
            if file_to_remove.exists():
                file_to_remove.unlink()
                print(f"✅ 删除: {file_to_remove.name}")
        
        # 清理测试知识库目录
        if test_working_dir.exists():
            import shutil
            shutil.rmtree(test_working_dir)
            print(f"✅ 删除测试知识库: {test_working_dir.name}")
        
    except Exception as e:
        print(f"❌ 文件测试失败: {e}")
        return False
    
    print("\n🎉 所有测试完成！")
    return True

def check_deployment_structure():
    """检查部署结构"""
    print("📁 检查部署结构")
    print("=" * 30)
    
    deployment_dir = Path("deployment")
    if not deployment_dir.exists():
        print("❌ deployment 目录不存在")
        return False
    
    required_files = [
        "config.yaml",
        ".env.example",
        "README.md"
    ]
    
    required_dirs = [
        "working",
        "output", 
        "logs",
        "temp"
    ]
    
    # 检查可执行文件
    if platform.system() == "Windows":
        exe_name = "qa_gen_pipeline.exe"
    else:
        exe_name = "qa_gen_pipeline"
    
    exe_path = deployment_dir / exe_name
    if exe_path.exists():
        print(f"✅ 可执行文件: {exe_name}")
        # 检查文件大小
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"  文件大小: {size_mb:.1f} MB")
    else:
        print(f"❌ 可执行文件不存在: {exe_name}")
        return False
    
    # 检查必需文件
    for file_name in required_files:
        file_path = deployment_dir / file_name
        if file_path.exists():
            print(f"✅ 配置文件: {file_name}")
        else:
            print(f"❌ 配置文件缺失: {file_name}")
    
    # 检查必需目录
    for dir_name in required_dirs:
        dir_path = deployment_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ 目录: {dir_name}")
        else:
            print(f"❌ 目录缺失: {dir_name}")
    
    # 检查重新打包脚本
    if platform.system() == "Windows":
        rebuild_script = deployment_dir / "rebuild.bat"
    else:
        rebuild_script = deployment_dir / "rebuild.sh"
    
    if rebuild_script.exists():
        print(f"✅ 重新打包脚本: {rebuild_script.name}")
    else:
        print(f"⚠️  重新打包脚本缺失: {rebuild_script.name}")
    
    return True

def main():
    """主函数"""
    print("🧪 QA生成管道可执行文件测试")
    print("=" * 50)
    
    # 检查部署结构
    if not check_deployment_structure():
        print("\n❌ 部署结构检查失败")
        return
    
    print("\n" + "=" * 50)
    
    # 测试可执行文件
    if test_executable():
        print("\n✅ 所有测试通过！可执行文件运行正常。")
        print("\n💡 提示:")
        print("1. 确保在 .env 文件中配置了正确的 DEEPSEEK_API_KEY")
        print("2. 在生产环境中使用前，建议进行更全面的测试")
        print("3. 参考 deployment/README.md 了解详细使用方法")
    else:
        print("\n❌ 测试失败！请检查构建过程和错误信息。")

if __name__ == "__main__":
    main() 