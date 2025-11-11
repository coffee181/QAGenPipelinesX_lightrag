#!/usr/bin/env python3
"""
使用虚拟环境的QA生成管道可执行文件打包脚本

此脚本创建一个干净的虚拟环境来避免pathlib包冲突
"""

import os
import sys
import subprocess
import shutil
import venv
import yaml
from pathlib import Path

def create_virtual_env():
    """创建虚拟环境"""
    venv_path = Path("build_venv")
    
    if venv_path.exists():
        print("🧹 清理旧的虚拟环境...")
        shutil.rmtree(venv_path)
    
    print("🔧 创建虚拟环境...")
    venv.create(venv_path, with_pip=True)
    
    # 确定虚拟环境中的Python路径
    if os.name == 'nt':  # Windows
        venv_python = venv_path / "Scripts" / "python.exe"
        venv_pip = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        venv_python = venv_path / "bin" / "python"
        venv_pip = venv_path / "bin" / "pip"
    
    return venv_python, venv_pip

def install_dependencies(venv_pip):
    """在虚拟环境中安装依赖"""
    print("📦 安装依赖包...")
    
    # 基础依赖
    deps = [
        "pyinstaller",
        "pyyaml",
        "python-dotenv",
        "requests",
        "loguru",
        "numpy",
        "pandas",
        "openai",
        "lightrag-hku",  # 添加lightrag依赖 (使用正确的包名)
        "markdown",   # 添加markdown依赖
        "jsonlines",  # 添加jsonlines依赖
        "tiktoken>=0.8.0",  # 确保tiktoken版本支持o200k_base
        "networkx",   # LightRAG图计算依赖
        "graspologic", # LightRAG社区检测依赖
        "nano-vectordb", # LightRAG向量数据库依赖
    ]
    
    # 首先安装pyinstaller
    print("  安装PyInstaller...")
    try:
        subprocess.check_call([str(venv_pip), "install", "pyinstaller"])
        print("  ✓ 安装: pyinstaller")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ PyInstaller安装失败: {e}")
        return False
    
    # 检查requirements.txt
    if Path("requirements.txt").exists():
        print("  使用requirements.txt安装依赖...")
        try:
            subprocess.check_call([str(venv_pip), "install", "-r", "requirements.txt"])
        except subprocess.CalledProcessError:
            print("  ⚠️  requirements.txt安装失败，安装基础依赖...")
            for dep in deps[1:]:  # 跳过已安装的pyinstaller
                try:
                    subprocess.check_call([str(venv_pip), "install", dep])
                    print(f"  ✓ 安装: {dep}")
                except subprocess.CalledProcessError as e:
                    print(f"  ⚠️  安装失败: {dep} - {e}")
    else:
        print("  安装基础依赖...")
        for dep in deps[1:]:  # 跳过已安装的pyinstaller
            try:
                subprocess.check_call([str(venv_pip), "install", dep])
                print(f"  ✓ 安装: {dep}")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️  安装失败: {dep} - {e}")
    
    print("✓ 依赖安装完成")

def fix_config_paths():
    """修复配置文件路径"""
    print("🔧 修复配置文件路径...")
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("❌ config.yaml 文件不存在")
        return False
    
    # 备份原始配置
    backup_path = config_path.with_suffix('.yaml.backup')
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"  ✓ 备份配置文件: {backup_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修复工作目录路径
    if 'rag' in config and 'lightrag' in config['rag']:
        old_working_dir = config['rag']['lightrag'].get('working_dir', '')
        if old_working_dir and Path(old_working_dir).is_absolute():
            config['rag']['lightrag']['working_dir'] = './working'
            print(f"  ✓ 修复工作目录: {old_working_dir} -> ./working")
            
            # 保存修改后的配置
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    return True

def prepare_tiktoken(venv_python):
    """准备tiktoken编码"""
    print("🔧 准备tiktoken编码...")
    
    try:
        # 运行tiktoken准备脚本
        result = subprocess.run([str(venv_python), "prepare_tiktoken.py"], 
                              capture_output=True, text=True, check=True)
        print("✓ tiktoken编码准备完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  tiktoken准备失败: {e}")
        print(f"错误输出: {e.stderr}")
        # 不要因为这个失败就停止构建，让用户决定
        return False
    except Exception as e:
        print(f"⚠️  tiktoken准备过程出错: {e}")
        return False

def build_executable(venv_python):
    """构建可执行文件"""
    print("🚀 构建可执行文件...")
    
    # 构建命令
    cmd = [
        str(venv_python), '-m', 'PyInstaller',
        '--onefile',
        '--clean',
        '--noconfirm',
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
        '--hidden-import=lightrag.utils.hashing',
        '--hidden-import=lightrag.utils.text_processing',
        '--hidden-import=lightrag.utils.vector_store',
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
        '--hidden-import=pytesseract',
        '--hidden-import=PIL',
        '--hidden-import=pdf2image',
        '--hidden-import=PyPDF2',
        '--hidden-import=markdown',
        '--hidden-import=jsonlines',
        '--exclude-module=tkinter',
        '--exclude-module=matplotlib',
        '--exclude-module=IPython',
        '--exclude-module=jupyter',
        '--exclude-module=pytest',
        'main.py'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ 构建成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 构建失败: {e}")
        print(f"错误输出: {e.stderr}")
        if e.stdout:
            print(f"标准输出: {e.stdout}")
        return False

def create_deployment():
    """创建部署包"""
    print("📦 创建部署包...")
    
    deploy_dir = Path("deployment")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    # 复制可执行文件
    exe_name = "qa_gen_pipeline.exe" if os.name == 'nt' else "qa_gen_pipeline"
    exe_source = Path("dist") / exe_name
    exe_target = deploy_dir / exe_name
    
    if exe_source.exists():
        shutil.copy2(exe_source, exe_target)
        if os.name != 'nt':
            os.chmod(exe_target, 0o755)
        print(f"✓ 复制可执行文件: {exe_name}")
    else:
        print(f"❌ 可执行文件不存在: {exe_source}")
        return False
    
    # 复制配置文件
    shutil.copy2("config.yaml", deploy_dir / "config.yaml")
    print("✓ 复制配置文件")
    
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
    print("✓ 创建环境变量示例")
    
    # 创建目录结构
    for dirname in ["working", "output", "logs", "temp"]:
        (deploy_dir / dirname).mkdir()
        (deploy_dir / dirname / ".gitkeep").touch()
    print("✓ 创建目录结构")
    
    # 创建README
    readme_content = f"""# QA生成管道可执行文件

## 使用说明

1. 配置环境变量：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，设置 DEEPSEEK_API_KEY
   ```

2. 运行程序：
   ```bash
   # Windows
   qa_gen_pipeline.exe generate-answers questions.jsonl ./working output.jsonl
   
   # Linux/macOS
   ./qa_gen_pipeline generate-answers questions.jsonl ./working output.jsonl
   ```

## 目录说明

- `{exe_name}`: 主程序
- `config.yaml`: 配置文件（已优化为相对路径）
- `.env`: 环境变量（从 .env.example 复制）
- `working/`: 知识库工作目录
- `output/`: 输出目录
- `logs/`: 日志目录
- `temp/`: 临时文件目录

## 注意事项

1. 必须配置 DEEPSEEK_API_KEY
2. 确保有网络连接
3. 首次运行可能需要较长时间下载模型
"""
    
    with open(deploy_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✓ 创建使用说明")
    
    return True

def cleanup():
    """清理临时文件"""
    print("🧹 清理临时文件...")
    
    cleanup_items = ["build", "dist", "build_venv", "__pycache__"]
    for item in cleanup_items:
        path = Path(item)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
                print(f"✓ 删除: {path}")
            else:
                path.unlink()
                print(f"✓ 删除: {path}")
    
    # 删除spec文件
    import glob
    for spec_file in glob.glob("*.spec"):
        try:
            os.remove(spec_file)
            print(f"✓ 删除spec文件: {spec_file}")
        except:
            pass

def restore_config():
    """恢复原始配置文件"""
    backup_path = Path("config.yaml.backup")
    config_path = Path("config.yaml")
    
    if backup_path.exists():
        print("🔄 恢复原始配置文件...")
        shutil.copy2(backup_path, config_path)
        backup_path.unlink()
        print("✓ 配置文件已恢复")

def main():
    """主函数"""
    print("🔧 QA生成管道虚拟环境构建工具")
    print("=" * 50)
    
    if not Path("main.py").exists():
        print("❌ 错误: 请在项目根目录运行此脚本")
        sys.exit(1)
    
    try:
        # 1. 创建虚拟环境
        venv_python, venv_pip = create_virtual_env()
        
        # 2. 安装依赖
        install_dependencies(venv_pip)
        
        # 3. 准备tiktoken编码
        prepare_tiktoken(venv_python)
        
        # 4. 修复配置路径
        if not fix_config_paths():
            print("❌ 配置文件处理失败")
            sys.exit(1)
        
        # 5. 构建可执行文件
        if not build_executable(venv_python):
            print("❌ 构建失败")
            restore_config()
            sys.exit(1)
        
        # 6. 创建部署包
        if not create_deployment():
            print("❌ 部署包创建失败")
            restore_config()
            sys.exit(1)
        
        # 7. 恢复配置（但保留虚拟环境用于调试）
        restore_config()
        
        print("\n🎉 打包完成!")
        exe_name = "qa_gen_pipeline.exe" if os.name == 'nt' else "qa_gen_pipeline"
        print(f"✓ 可执行文件: deployment/{exe_name}")
        print("✓ 使用说明: deployment/README.md")
        print("\n💡 测试: python test_build.py")
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断")
        restore_config()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        restore_config()
        sys.exit(1)

if __name__ == "__main__":
    main() 