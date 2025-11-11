#!/usr/bin/env python3
"""
QA生成管道可执行文件打包脚本

此脚本使用PyInstaller将QA生成管道打包为独立的可执行文件，
便于在后端系统中部署和调用。
"""

import os
import sys
import subprocess
import shutil
import yaml
from pathlib import Path

def check_dependencies():
    """检查打包依赖"""
    try:
        import PyInstaller
        print("✓ PyInstaller 已安装")
    except ImportError:
        print("❌ PyInstaller 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller 安装完成")

def create_production_config():
    """创建适用于生产环境的配置文件，修复硬编码路径问题"""
    print("🔧 创建生产环境配置文件...")
    
    # 读取现有配置
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("❌ config.yaml 文件不存在")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修复硬编码路径问题
    changes_made = False
    
    # 修复 RAG working_dir 的硬编码路径
    if 'rag' in config and 'lightrag' in config['rag']:
        old_working_dir = config['rag']['lightrag'].get('working_dir', '')
        if old_working_dir and Path(old_working_dir).is_absolute():
            config['rag']['lightrag']['working_dir'] = './working'
            print(f"  ✓ 修复 RAG working_dir: {old_working_dir} -> ./working")
            changes_made = True
    
    # 确保所有路径都是相对路径
    path_configs = [
        ('file_processing.output_dir', './output'),
        ('file_processing.temp_dir', './temp'),
        ('progress.progress_file', './progress.json'),
        ('logging.file', './logs/qa_gen.log')
    ]
    
    for path_key, default_path in path_configs:
        keys = path_key.split('.')
        config_section = config
        
        # 导航到配置部分
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # 检查并修复路径
        final_key = keys[-1]
        if final_key in config_section:
            current_path = config_section[final_key]
            if current_path and Path(current_path).is_absolute():
                config_section[final_key] = default_path
                print(f"  ✓ 修复路径 {path_key}: {current_path} -> {default_path}")
                changes_made = True
    
    # 如果有修改，保存新的配置文件
    if changes_made:
        # 创建备份
        backup_path = config_path.with_suffix('.yaml.backup')
        shutil.copy2(config_path, backup_path)
        print(f"  ✓ 原配置文件备份至: {backup_path}")
        
        # 保存修改后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        print("  ✓ 生产环境配置文件已更新")
    else:
        print("  ✓ 配置文件路径已是相对路径，无需修改")
    
    return True

def create_spec_file():
    """创建PyInstaller配置文件"""
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

block_cipher = None

# 获取项目根目录
project_root = Path.cwd()

# 数据文件配置 - 确保包含所有必要的文件
datas = [
    ('config.yaml', '.'),
    ('src', 'src'),
]

# 确保requirements.txt存在时也包含进去
if os.path.exists('requirements.txt'):
    datas.append(('requirements.txt', '.'))

# 隐藏导入 - 包含所有可能需要的模块
hiddenimports = [
    # 核心依赖
    'lightrag',
    'openai', 
    'requests',
    'loguru',
    'markdown',
    'numpy',
    'pandas',
    'pathlib',
    'json',
    'yaml',
    'dotenv',
    'python-dotenv',
    
    # 科学计算和ML相关
    'scipy',
    'sklearn',
    'transformers',
    'torch',
    'tiktoken',
    
    # 异步和网络
    'asyncio',
    'aiohttp',
    'httpx',
    
    # 数据处理
    'pydantic',
    'dataclasses',
    'typing',
    'typing_extensions',
    
    # 文件处理
    'PyPDF2',
    'pdf2image',
    'PIL',
    'pytesseract',
    
    # 可选的web框架
    'fastapi',
    'uvicorn',
    'flask',
    
    # 图数据处理（LightRAG可能需要）
    'networkx',
    'neo4j',
]

# 排除不需要的模块以减小体积
excludes = [
    'tkinter',
    'matplotlib',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'unittest',
    'doctest',
    'pathlib',  # 排除pathlib包，因为它是Python 3.4+的内置模块
]

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='qa_gen_pipeline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open('qa_gen_pipeline.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content.strip())
    
    print("✓ 创建 PyInstaller 配置文件")

def build_executable():
    """构建可执行文件"""
    print("🚀 开始构建可执行文件...")
    
    # 使用spec文件构建
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--clean',
        '--noconfirm',  # 不询问确认
        'qa_gen_pipeline.spec'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ 可执行文件构建完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 构建失败: {e}")
        print(f"错误输出: {e.stderr}")
        print(f"标准输出: {e.stdout}")
        return False

def create_deployment_package():
    """创建部署包"""
    print("📦 创建部署包...")
    
    # 创建部署目录
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
        # 设置可执行权限 (Unix/Linux)
        if os.name != 'nt':
            os.chmod(exe_target, 0o755)
        print(f"✓ 复制可执行文件: {exe_target}")
    else:
        print(f"❌ 可执行文件不存在: {exe_source}")
        return False
    
    # 复制配置文件
    config_files = ["config.yaml"]
    for config_file in config_files:
        if Path(config_file).exists():
            shutil.copy2(config_file, deploy_dir / config_file)
            print(f"✓ 复制配置文件: {config_file}")
    
    # 复制requirements.txt（如果存在）
    if Path("requirements.txt").exists():
        shutil.copy2("requirements.txt", deploy_dir / "requirements.txt")
        print("✓ 复制依赖文件: requirements.txt")
    
    # 创建示例环境变量文件
    env_example = deploy_dir / ".env.example"
    with open(env_example, 'w', encoding='utf-8') as f:
        f.write("""# QA生成管道环境变量配置
# 复制此文件为 .env 并填入实际的API密钥

# 必需配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 可选配置（用于OpenAI嵌入向量，如不配置将使用哈希嵌入）
OPENAI_API_KEY=your_openai_api_key_here
""")
    print("✓ 创建环境变量示例文件")
    
    # 创建目录结构
    dirs_to_create = ["working", "output", "logs", "temp"]
    for dir_name in dirs_to_create:
        (deploy_dir / dir_name).mkdir(exist_ok=True)
        # 创建 .gitkeep 文件
        (deploy_dir / dir_name / ".gitkeep").touch()
    print("✓ 创建目录结构")
    
    # 创建部署说明
    readme_content = f"""# QA生成管道部署包

## 快速开始

1. 配置环境变量：
   ```bash
   # 复制环境变量模板
   cp .env.example .env
   
   # 编辑 .env 文件，填入你的API密钥
   nano .env
   ```

2. 运行可执行文件：
   ```bash
   # Linux/macOS
   ./qa_gen_pipeline generate-answers questions.jsonl ./working output.jsonl
   
   # Windows
   qa_gen_pipeline.exe generate-answers questions.jsonl ./working output.jsonl
   ```

## 目录说明

- `{exe_name}`: 主程序可执行文件
- `config.yaml`: 系统配置文件（已优化为相对路径）
- `.env.example`: 环境变量模板
- `working/`: 知识库工作目录
- `output/`: 输出文件目录
- `logs/`: 日志文件目录
- `temp/`: 临时文件目录

## 重要说明

1. **路径配置**: 配置文件已自动调整为相对路径，确保跨环境兼容性
2. **API密钥**: 必须配置DEEPSEEK_API_KEY才能正常使用
3. **知识库**: working目录将作为LightRAG知识库存储位置
4. **日志**: 所有日志将保存在logs目录下

## 使用说明

请参考主项目的 `Docs/API_USAGE_GUIDE.md` 文件获取详细的使用说明。

## 系统要求

- Windows 10/11 或 Linux (x64) 或 macOS
- 至少 2GB 内存
- 至少 1GB 磁盘空间
- 网络连接（用于API调用）

## 故障排除

1. **权限问题 (Linux/macOS)**: 运行 `chmod +x qa_gen_pipeline`
2. **API密钥错误**: 检查 .env 文件中的配置
3. **路径问题**: 确保在可执行文件目录下运行命令
4. **依赖问题**: 可执行文件已包含所有依赖，无需额外安装
"""
    
    with open(deploy_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✓ 创建部署说明")
    
    # 创建快速打包脚本
    create_rebuild_script(deploy_dir)
    
    print(f"📦 部署包创建完成: {deploy_dir.absolute()}")
    return True

def create_rebuild_script(deploy_dir: Path):
    """创建快速重新打包脚本"""
    print("📜 创建重新打包脚本...")
    
    # 为不同平台创建脚本
    if os.name == 'nt':  # Windows
        script_content = '''@echo off
echo 🔧 QA生成管道快速重新打包
echo ====================================

cd /d "%~dp0.."

echo 📁 当前目录: %CD%

echo 🚀 开始重新打包...
python build_executable.py

if %ERRORLEVEL% EQU 0 (
    echo ✅ 打包完成！
    echo 📦 可执行文件位于: deployment\\qa_gen_pipeline.exe
) else (
    echo ❌ 打包失败！
    pause
)
'''
        script_path = deploy_dir / "rebuild.bat"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        print("✓ 创建 Windows 重新打包脚本: rebuild.bat")
    
    else:  # Unix/Linux/macOS
        script_content = '''#!/bin/bash
echo "🔧 QA生成管道快速重新打包"
echo "===================================="

# 切换到项目根目录
cd "$(dirname "$0")/.."

echo "📁 当前目录: $(pwd)"

echo "🚀 开始重新打包..."
python build_executable.py

if [ $? -eq 0 ]; then
    echo "✅ 打包完成！"
    echo "📦 可执行文件位于: deployment/qa_gen_pipeline"
else
    echo "❌ 打包失败！"
    exit 1
fi
'''
        script_path = deploy_dir / "rebuild.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        # 设置可执行权限
        os.chmod(script_path, 0o755)
        print("✓ 创建 Unix/Linux 重新打包脚本: rebuild.sh")

def cleanup():
    """清理临时文件"""
    print("🧹 清理临时文件...")
    
    cleanup_items = [
        "build",
        "dist", 
        "__pycache__",
        "qa_gen_pipeline.spec",
    ]
    
    for item in cleanup_items:
        path = Path(item)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
                print(f"✓ 删除目录: {path}")
            else:
                path.unlink()
                print(f"✓ 删除文件: {path}")
    
    # 清理Python缓存文件
    import glob
    for pyc_file in glob.glob("**/*.pyc", recursive=True):
        try:
            os.remove(pyc_file)
        except:
            pass
    
    for pycache_dir in glob.glob("**/__pycache__", recursive=True):
        try:
            shutil.rmtree(pycache_dir)
        except:
            pass

def restore_original_config():
    """恢复原始配置文件"""
    backup_path = Path("config.yaml.backup")
    config_path = Path("config.yaml")
    
    if backup_path.exists():
        print("🔄 恢复原始配置文件...")
        shutil.copy2(backup_path, config_path)
        backup_path.unlink()
        print("✓ 原始配置文件已恢复")

def main():
    """主函数"""
    print("🔧 QA生成管道可执行文件打包工具")
    print("=" * 50)
    
    # 检查当前目录
    if not Path("main.py").exists():
        print("❌ 错误: 请在项目根目录运行此脚本")
        sys.exit(1)
    
    try:
        # 1. 检查依赖
        check_dependencies()
        
        # 2. 创建适用于生产环境的配置文件
        if not create_production_config():
            print("❌ 配置文件处理失败")
            sys.exit(1)
        
        # 3. 创建spec文件
        create_spec_file()
        
        # 4. 构建可执行文件
        if not build_executable():
            print("❌ 构建失败")
            restore_original_config()
            sys.exit(1)
        
        # 5. 创建部署包
        if not create_deployment_package():
            print("❌ 部署包创建失败")
            restore_original_config()
            sys.exit(1)
        
        # 6. 清理临时文件
        cleanup()
        
        # 7. 恢复原始配置文件
        restore_original_config()
        
        print("\n🎉 打包完成!")
        exe_name = "qa_gen_pipeline.exe" if os.name == 'nt' else "qa_gen_pipeline"
        print(f"✓ 可执行文件位于: deployment/{exe_name}")
        print("✓ 请参考 deployment/README.md 进行部署")
        print("✓ 使用 deployment/rebuild.sh 或 deployment/rebuild.bat 快速重新打包")
        
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断")
        restore_original_config()
        cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        restore_original_config()
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main() 