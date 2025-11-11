#!/usr/bin/env python3
"""测试本地模型构建脚本"""

import subprocess
import sys
from pathlib import Path

def test_build_script():
    """测试构建脚本"""
    print("=== 测试本地模型构建脚本 ===")
    
    # 检查构建脚本是否存在
    build_script = Path("quick_build_local.py")
    if not build_script.exists():
        print("❌ 构建脚本不存在: quick_build_local.py")
        return False
    
    print("✓ 构建脚本存在")
    
    # 检查配置文件是否存在
    config_file = Path("config_local.yaml")
    if not config_file.exists():
        print("❌ 本地模型配置文件不存在: config_local.yaml")
        return False
    
    print("✓ 本地模型配置文件存在")
    
    # 检查虚拟环境是否存在
    venv_path = Path('build_venv')
    if not venv_path.exists():
        print("❌ 虚拟环境不存在，请先运行: python build_with_venv.py")
        return False
    
    print("✓ 虚拟环境存在")
    
    # 检查PyInstaller是否安装
    try:
        if sys.platform == "win32":
            venv_python = venv_path / 'Scripts' / 'python.exe'
        else:
            venv_python = venv_path / 'bin' / 'python'
        
        if not venv_python.exists():
            print("❌ 虚拟环境Python不存在")
            return False
        
        # 测试PyInstaller
        result = subprocess.run(
            [str(venv_python), '-m', 'PyInstaller', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✓ PyInstaller已安装: {result.stdout.strip()}")
        else:
            print("❌ PyInstaller未安装或有问题")
            return False
            
    except Exception as e:
        print(f"❌ 检查PyInstaller时出错: {e}")
        return False
    
    print("\n🎉 所有检查通过！")
    print("现在可以运行: python quick_build_local.py")
    return True

def show_build_info():
    """显示构建信息"""
    print("\n=== 构建信息 ===")
    print("构建脚本: quick_build_local.py")
    print("配置文件: config_local.yaml")
    print("输出目录: deployment_local/")
    print("可执行文件: qa_gen_pipeline_local.exe")
    print("默认模型: deepseek-r1:32b")
    print("服务地址: http://localhost:11434")

def main():
    """主函数"""
    print("本地模型构建脚本测试工具")
    print("=" * 50)
    
    if test_build_script():
        show_build_info()
        print("\n📋 构建步骤:")
        print("1. 确保Ollama服务运行: ollama serve")
        print("2. 下载模型: ollama pull deepseek-r1:32b")
        print("3. 运行构建: python quick_build_local.py")
        print("4. 测试可执行文件: deployment_local/qa_gen_pipeline_local.exe")
    else:
        print("\n❌ 构建环境检查失败")
        print("请解决上述问题后重新运行")

if __name__ == "__main__":
    main()
