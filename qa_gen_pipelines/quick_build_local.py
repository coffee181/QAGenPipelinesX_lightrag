#!/usr/bin/env python3
"""快速构建支持本地模型的可执行文件脚本"""

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
    print_with_emoji("🚀", "快速构建支持本地模型的可执行文件...")
    
    # 检查虚拟环境
    venv_path = Path('build_venv')
    if os.name == 'nt':
        venv_python = venv_path / 'Scripts' / 'python.exe'
    else:
        venv_python = venv_path / 'bin' / 'python'
    
    if not venv_python.exists():
        print_with_emoji("❌", "虚拟环境不存在，请先运行 python build_with_venv.py")
        sys.exit(1)
    
    # 构建命令 - 添加本地模型相关依赖
    cmd = [
        str(venv_python), '-m', 'PyInstaller',
        '--onefile', '--clean', '--noconfirm',
        '--name=qa_gen_pipeline_local',
        '--add-data=config_local.yaml;config.yaml',
        '--add-data=src;src',
        # LightRAG相关
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
        # 基础依赖
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
        # 本地模型相关依赖
        '--hidden-import=ollama',
        '--hidden-import=vllm',
        '--hidden-import=transformers',
        '--hidden-import=torch',
        '--hidden-import=fastapi',
        '--hidden-import=uvicorn',
        '--hidden-import=pydantic',
        'main.py'
    ]
    
    safe_print("执行构建命令...")
    # 获取当前系统的完整环境变量，并复制一份
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print_with_emoji("✓", "构建成功")
        # 创建部署包
        print_with_emoji("📦", "创建本地模型部署包...")
        deploy_dir = Path("deployment_local")
        if deploy_dir.exists():
            try:
                shutil.rmtree(deploy_dir)
            except PermissionError:
                safe_print("⚠️  deployment_local目录被占用，使用新目录名...")
                import time
                deploy_dir = Path(f"deployment_local_{int(time.time())}")
                safe_print(f"✓ 使用新目录: {deploy_dir}")
        deploy_dir.mkdir()
        
        # 复制可执行文件
        exe_name = "qa_gen_pipeline_local.exe" if os.name == 'nt' else "qa_gen_pipeline_local"
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
        shutil.copy2("config_local.yaml", deploy_dir / "config.yaml")
        safe_print("✓ 复制本地模型配置文件")
        
        # 创建本地模型环境变量示例
        env_content = """# QA生成管道本地模型环境变量配置
# 复制此文件为 .env 并填入实际的API密钥

# 本地模型配置（推荐）
# 使用本地模型时，以下API密钥不是必需的
# DEEPSEEK_API_KEY=your_deepseek_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here

# 本地模型服务配置
# Ollama服务地址（默认）
OLLAMA_BASE_URL=http://localhost:11434

# 本地模型名称
LOCAL_MODEL_NAME=deepseek-r1:32b

# 可选：vLLM服务配置
# VLLM_BASE_URL=http://localhost:8000
"""
        with open(deploy_dir / ".env.example", 'w', encoding='utf-8') as f:
            f.write(env_content)
        safe_print("✓ 创建本地模型环境变量示例")
        
        # 创建目录结构
        for dirname in ["working", "output", "logs", "temp"]:
            (deploy_dir / dirname).mkdir()
            (deploy_dir / dirname / ".gitkeep").touch()
        safe_print("✓ 创建目录结构")
        
        # 创建本地模型使用说明
        readme_content = """# 本地模型部署包使用说明

## 🎯 概述
这是支持本地模型的QA生成管道可执行文件，使用deepseek-r1:32b模型进行问答生成。

## 📋 使用前准备

### 1. 安装Ollama
```bash
# Windows: 下载并安装 https://ollama.ai/download
# Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. 启动Ollama服务
```bash
ollama serve
```

### 3. 下载模型
```bash
ollama pull deepseek-r1:32b
```

### 4. 测试模型
```bash
ollama run deepseek-r1:32b "你好，请介绍一下你自己"
```

## 🚀 使用方法

### 1. 配置环境
```bash
# 复制环境变量示例
copy .env.example .env

# 编辑.env文件（可选，使用默认配置即可）
```

### 2. 运行程序
```bash
# Windows
qa_gen_pipeline_local.exe

# Linux/macOS
./qa_gen_pipeline_local
```

## ⚙️ 配置说明

### config.yaml配置
程序会自动使用本地模型配置：
```yaml
question_generator:
  provider: "local"
  local:
    model_name: "deepseek-r1:32b"
    base_url: "http://localhost:11434"
    max_tokens: 2048
    temperature: 0.7
    timeout: 120
    questions_per_chunk: 30
```

### 切换回API模式
如需使用API模式，修改config.yaml：
```yaml
question_generator:
  provider: "deepseek"  # 改为deepseek使用API
```

## 🔧 故障排除

### 1. 模型连接失败
- 检查Ollama服务是否运行：`ollama serve`
- 检查模型是否下载：`ollama list`
- 检查端口是否被占用：`netstat -an | grep 11434`

### 2. GPU内存不足
- 使用更小的模型：`ollama pull deepseek-r1:7b`
- 检查GPU状态：`nvidia-smi`

### 3. 程序运行缓慢
- 检查GPU利用率：`nvidia-smi`
- 调整超时时间：修改config.yaml中的timeout值

## 💡 优势

- ✅ 完全免费，无API费用
- ✅ 数据安全，不离开本地
- ✅ 响应速度快，无网络延迟
- ✅ 无使用限制
- ✅ 完全离线运行

## 📞 技术支持

如遇问题，请检查：
1. Ollama服务状态
2. 模型下载情况
3. GPU内存使用
4. 配置文件设置

享受免费的本地AI服务！🎉
"""
        with open(deploy_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        safe_print("✓ 创建使用说明")
        
        # 创建启动脚本
        if os.name == 'nt':
            # Windows批处理脚本
            bat_content = """@echo off
echo 启动QA生成管道（本地模型版本）...
echo.
echo 请确保：
echo 1. Ollama服务正在运行 (ollama serve)
echo 2. deepseek-r1:32b模型已下载 (ollama pull deepseek-r1:32b)
echo.
pause
qa_gen_pipeline_local.exe
pause
"""
            with open(deploy_dir / "start.bat", 'w', encoding='utf-8') as f:
                f.write(bat_content)
            safe_print("✓ 创建Windows启动脚本")
        else:
            # Linux/macOS shell脚本
            sh_content = """#!/bin/bash
echo "启动QA生成管道（本地模型版本）..."
echo ""
echo "请确保："
echo "1. Ollama服务正在运行 (ollama serve)"
echo "2. deepseek-r1:32b模型已下载 (ollama pull deepseek-r1:32b)"
echo ""
read -p "按回车键继续..."
./qa_gen_pipeline_local
"""
            with open(deploy_dir / "start.sh", 'w', encoding='utf-8') as f:
                f.write(sh_content)
            os.chmod(deploy_dir / "start.sh", 0o755)
            safe_print("✓ 创建Linux/macOS启动脚本")
        
        safe_print("")
        print_with_emoji("🎉", "本地模型构建完成!")
        safe_print(f"✓ 可执行文件: deployment_local/{exe_name}")
        safe_print("✓ 包含完整的本地模型支持")
        safe_print("✓ 包含使用说明和启动脚本")
        safe_print("")
        safe_print("📋 使用步骤:")
        safe_print("1. 安装Ollama: https://ollama.ai/download")
        safe_print("2. 启动服务: ollama serve")
        safe_print("3. 下载模型: ollama pull deepseek-r1:32b")
        safe_print("4. 运行程序: deployment_local/start.bat (Windows) 或 ./start.sh (Linux/macOS)")
        
    else:
        print_with_emoji("❌", f"构建失败: {result.stderr}")
        if result.stdout:
            safe_print(f"标准输出: {result.stdout}")
        sys.exit(1)

if __name__ == "__main__":
    main()
