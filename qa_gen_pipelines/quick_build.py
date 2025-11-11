#!/usr/bin/env python3
"""快速构建脚本"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

# 添加src到路径以便导入工具模块
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from utils.console_utils import ConsoleOutputFixer, safe_print, print_with_emoji
    # 修复控制台编码
    ConsoleOutputFixer.fix_console_encoding()
    use_safe_print = True
except ImportError:
    # 如果无法导入，使用标准print
    def safe_print(*args, **kwargs):
        print(*args, **kwargs)
    def print_with_emoji(emoji, message, level="DEBUG"):
        print(f"{emoji} {message}")
    use_safe_print = False

def main():
    print_with_emoji("🚀", "快速构建可执行文件...")
    
    # 检查虚拟环境
    venv_path = Path('build_venv')
    if os.name == 'nt':
        venv_python = venv_path / 'Scripts' / 'python.exe'
    else:
        venv_python = venv_path / 'bin' / 'python'
    
    if not venv_python.exists():
        print_with_emoji("❌", "虚拟环境不存在，请先运行 python build_with_venv.py")
        sys.exit(1)
    
    # 构建命令
    cmd = [
        str(venv_python), '-m', 'PyInstaller',
        '--onefile', '--clean', '--noconfirm',
        '--name=qa_gen_pipeline',
        '--add-data=config.yaml;.',
        '--add-data=src;src',
        '--hidden-import=lightrag',
        '--hidden-import=lightrag.utils',
        '--hidden-import=lightrag.llm',
        '--hidden-import=lightrag.storage',
        '--hidden-import=lightrag.operate',
        '--hidden-import=lightrag.base',
        '--hidden-import=lightrag.kg',
        '--hidden-import=lightrag.kg.json_kv_impl',
        '--hidden-import=lightrag.kg.neo4j_impl',
        '--hidden-import=lightrag.kg.networkx_impl',
        '--hidden-import=lightrag.kg.nano_vector_db_impl',
        '--hidden-import=lightrag.kg.age_impl',
        '--hidden-import=lightrag.kg.chroma_impl',
        '--hidden-import=lightrag.kg.faiss_impl',
        '--hidden-import=lightrag.kg.gremlin_impl',
        '--hidden-import=lightrag.kg.json_doc_status_impl',
        '--hidden-import=lightrag.kg.milvus_impl',
        '--hidden-import=lightrag.kg.mongo_impl',
        '--hidden-import=lightrag.kg.postgres_impl',
        '--hidden-import=lightrag.kg.qdrant_impl',
        '--hidden-import=lightrag.kg.redis_impl',
        '--hidden-import=lightrag.kg.shared_storage',
        '--hidden-import=lightrag.kg.tidb_impl',
        '--hidden-import=lightrag.graph',
        '--hidden-import=lightrag.memory',
        '--hidden-import=lightrag.retrieve',
        '--hidden-import=openai',
        '--hidden-import=requests',
        '--hidden-import=loguru',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        '--hidden-import=networkx',
        '--hidden-import=networkx.algorithms',
        '--hidden-import=networkx.algorithms.community',
        '--hidden-import=graspologic',
        '--hidden-import=tiktoken',
        '--hidden-import=tiktoken.registry',
        '--hidden-import=tiktoken_ext',
        '--hidden-import=tiktoken_ext.openai_public',
        '--collect-data=tiktoken',
        '--hidden-import=nano_vectordb',
        '--hidden-import=nest_asyncio',
        '--hidden-import=jinja2',
        '--hidden-import=markdown',
        '--hidden-import=jsonlines',
        'main.py'
    ]
    
    safe_print("执行构建命令...")
    # 获取当前系统的完整环境变量，并复制一份
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print_with_emoji("✓", "构建成功")
        # 创建部署包
        print_with_emoji("📦", "创建部署包...")
        deploy_dir = Path("deployment")
        if deploy_dir.exists():
            try:
                shutil.rmtree(deploy_dir)
            except PermissionError:
                safe_print("⚠️  deployment目录被占用，使用新目录名...")
                import time
                deploy_dir = Path(f"deployment_new_{int(time.time())}")
                safe_print(f"✓ 使用新目录: {deploy_dir}")
        deploy_dir.mkdir()
        
        # 复制可执行文件
        exe_name = "qa_gen_pipeline.exe" if os.name == 'nt' else "qa_gen_pipeline"
        exe_source = Path("dist") / exe_name
        exe_target = deploy_dir / exe_name
        
        if exe_source.exists():
            shutil.copy2(exe_source, exe_target)
            if os.name != 'nt':
                os.chmod(exe_target, 0o755)
            safe_print(f"✓ 复制可执行文件: {exe_name}")
        else:
            print_with_emoji("❌", f"可执行文件不存在: {exe_source}")
            return False
        
        # 复制配置文件
        shutil.copy2("config.yaml", deploy_dir / "config.yaml")
        safe_print("✓ 复制配置文件")
        
        # 创建环境变量示例
        env_content = """# QA生成管道环境变量配置
# 复制此文件为 .env 并填入实际的API密钥

# 必需配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 可选配置
OPENAI_API_KEY=your_openai_api_key_here
"""
        with open(deploy_dir / ".env.example", 'w', encoding='utf-8') as f:
            f.write(env_content)
        safe_print("✓ 创建环境变量示例")
        
        # 创建目录结构
        for dirname in ["working", "output", "logs", "temp"]:
            (deploy_dir / dirname).mkdir()
            (deploy_dir / dirname / ".gitkeep").touch()
        safe_print("✓ 创建目录结构")
        
        safe_print("")
        print_with_emoji("🎉", "构建完成!")
        safe_print(f"✓ 可执行文件: deployment/{exe_name}")
        
    else:
        print_with_emoji("❌", f"构建失败: {result.stderr}")
        if result.stdout:
            safe_print(f"标准输出: {result.stdout}")
        sys.exit(1)

if __name__ == "__main__":
    main() 