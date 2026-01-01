#!/usr/bin/env python3
"""
快速构建可执行文件（参考现有 build 脚本，但精简为一键打包）。
运行前请先执行 `python build_with_venv.py` 以创建 build_venv。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# 项目根目录（qa_gen_pipelines）
PROJECT_ROOT = Path(__file__).resolve().parent

# 将 src 加入路径，尝试使用现有的输出工具
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from utils.console_utils import ConsoleOutputFixer, safe_print, print_with_emoji

    ConsoleOutputFixer.fix_console_encoding()
except Exception:  # noqa: BLE001
    # 回退到标准输出
    def safe_print(*args, **kwargs) -> None:
        print(*args, **kwargs)

    def print_with_emoji(emoji: str, message: str, level: str = "INFO") -> None:
        print(f"{emoji} {message}")


def main() -> None:
    print_with_emoji("🚀", "快速构建可执行文件...")

    # 检查虚拟环境
    venv_path = PROJECT_ROOT / "build_venv"
    if os.name == "nt":
        venv_python = venv_path / "Scripts" / "python.exe"
    else:
        venv_python = venv_path / "bin" / "python"

    if not venv_python.exists():
        print_with_emoji("❌", "虚拟环境不存在，请先运行 python build_with_venv.py")
        sys.exit(1)

    # 根据平台选择 add-data 分隔符
    data_sep = ";" if os.name == "nt" else ":"

    # 基础命令
    cmd = [
        str(venv_python),
        "-m",
        "PyInstaller",
        "--onefile",
        "--clean",
        "--noconfirm",
        "--name=qa_gen_pipeline",
        f"--add-data={PROJECT_ROOT / 'src'}{data_sep}src",
    ]

    # 可选配置文件
    for cfg_name in ("config.yaml", "config_local.yaml"):
        cfg_path = PROJECT_ROOT / cfg_name
        if cfg_path.exists():
            cmd.append(f"--add-data={cfg_path}{data_sep}.")

    # 隐藏导入（主要针对 lightrag 及依赖）
    hidden_imports = [
        "lightrag",
        "lightrag.utils",
        "lightrag.llm",
        "lightrag.storage",
        "lightrag.operate",
        "lightrag.base",
        "lightrag.kg",
        "lightrag.kg.json_kv_impl",
        "lightrag.kg.neo4j_impl",
        "lightrag.kg.networkx_impl",
        "lightrag.kg.nano_vector_db_impl",
        "lightrag.kg.age_impl",
        "lightrag.kg.chroma_impl",
        "lightrag.kg.faiss_impl",
        "lightrag.kg.gremlin_impl",
        "lightrag.kg.json_doc_status_impl",
        "lightrag.kg.milvus_impl",
        "lightrag.kg.mongo_impl",
        "lightrag.kg.postgres_impl",
        "lightrag.kg.qdrant_impl",
        "lightrag.kg.redis_impl",
        "lightrag.kg.shared_storage",
        "lightrag.kg.tidb_impl",
        "lightrag.graph",
        "lightrag.memory",
        "lightrag.retrieve",
        "openai",
        "requests",
        "loguru",
        "numpy",
        "pandas",
        "networkx",
        "networkx.algorithms",
        "networkx.algorithms.community",
        "graspologic",
        "tiktoken",
        "tiktoken.registry",
        "tiktoken_ext",
        "tiktoken_ext.openai_public",
        "nano_vectordb",
        "nest_asyncio",
        "jinja2",
        "markdown",
        "jsonlines",
    ]
    for mod in hidden_imports:
        cmd.append(f"--hidden-import={mod}")

    # 额外数据收集
    cmd.append("--collect-data=tiktoken")

    # 入口脚本
    cmd.append(str(PROJECT_ROOT / "main.py"))

    safe_print("执行构建命令...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print_with_emoji("❌", f"构建失败: {result.stderr}")
        if result.stdout:
            safe_print(f"标准输出: {result.stdout}")
        sys.exit(result.returncode)

    print_with_emoji("✓", "构建成功")

    # 创建部署包
    print_with_emoji("📦", "创建部署包...")
    deploy_dir = PROJECT_ROOT / "deployment"
    if deploy_dir.exists():
        try:
            shutil.rmtree(deploy_dir)
        except PermissionError:
            import time

            safe_print("⚠️  deployment 目录被占用，使用新目录名...")
            deploy_dir = PROJECT_ROOT / f"deployment_new_{int(time.time())}"
            safe_print(f"✓ 使用新目录: {deploy_dir}")
    deploy_dir.mkdir(parents=True, exist_ok=True)

    exe_name = "qa_gen_pipeline.exe" if os.name == "nt" else "qa_gen_pipeline"
    exe_source = PROJECT_ROOT / "dist" / exe_name
    exe_target = deploy_dir / exe_name

    if exe_source.exists():
        shutil.copy2(exe_source, exe_target)
        if os.name != "nt":
            os.chmod(exe_target, 0o755)
        safe_print(f"✓ 复制可执行文件: {exe_name}")
    else:
        print_with_emoji("❌", f"可执行文件不存在: {exe_source}")
        sys.exit(1)

    # 复制配置文件（存在则复制）
    for cfg_name in ("config.yaml", "config_local.yaml"):
        cfg_path = PROJECT_ROOT / cfg_name
        if cfg_path.exists():
            shutil.copy2(cfg_path, deploy_dir / cfg_name)
            safe_print(f"✓ 复制配置文件: {cfg_name}")

    # 创建环境变量示例
    env_content = """# QA生成管道环境变量配置
# 复制此文件为 .env 并填入实际的 API 密钥
# 必需配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here
# 可选配置
OPENAI_API_KEY=your_openai_api_key_here
"""
    with (deploy_dir / ".env.example").open("w", encoding="utf-8") as fp:
        fp.write(env_content)
    safe_print("✓ 创建环境变量示例")

    # 创建目录结构
    for dirname in ["working", "output", "logs", "temp"]:
        target_dir = deploy_dir / dirname
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / ".gitkeep").touch()
    safe_print("✓ 创建目录结构")

    safe_print()
    print_with_emoji("🎉", "构建完成!")
    safe_print(f"✓ 可执行文件: {deploy_dir / exe_name}")


if __name__ == "__main__":
    main()

