#!/usr/bin/env python3
"""
QA生成管道可执行文件打包脚本 v2

此脚本自动处理pathlib兼容性问题并使用PyInstaller将QA生成管道打包为独立的可执行文件
"""

import os
import sys
import subprocess
import shutil
import yaml
import tempfile
from pathlib import Path

def check_dependencies():
    """检查打包依赖"""
    try:
        import PyInstaller
        print("✓ PyInstaller 已安装")
        return True
    except ImportError:
        print("❌ PyInstaller 未安装，正在安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("✓ PyInstaller 安装完成")
            return True
        except Exception as e:
            print(f"❌ PyInstaller 安装失败: {e}")
            return False

def handle_pathlib_conflict():
    """处理pathlib包冲突"""
    print("🔧 处理pathlib包兼容性...")
    
    # 查找可能的pathlib包位置
    import site
    pathlib_files = []
    
    # 检查标准site-packages位置
    try:
        for site_dir in site.getsitepackages():
            pathlib_file = Path(site_dir) / "pathlib.py"
            if pathlib_file.exists():
                pathlib_files.append(pathlib_file)
    except:
        pass
    
    # 检查用户site-packages
    try:
        user_site = site.getusersitepackages()
        if user_site:
            pathlib_file = Path(user_site) / "pathlib.py"
            if pathlib_file.exists():
                pathlib_files.append(pathlib_file)
    except:
        pass
    
    # 直接检查可能的路径
    possible_paths = [
        "D:/Tools/Anaconda3/lib/site-packages/pathlib.py",
        "D:/Tools/Anaconda3/Lib/site-packages/pathlib.py",
        str(Path(sys.executable).parent.parent / "lib" / "site-packages" / "pathlib.py"),
        str(Path(sys.executable).parent / "lib" / "site-packages" / "pathlib.py"),
        str(Path(sys.executable).parent / "Lib" / "site-packages" / "pathlib.py"),
    ]
    
    for path_str in possible_paths:
        pathlib_file = Path(path_str)
        if pathlib_file.exists() and pathlib_file not in pathlib_files:
            pathlib_files.append(pathlib_file)
    
    if not pathlib_files:
        print("✓ 未发现冲突的pathlib包")
        return [], []
    
    print(f"发现 {len(pathlib_files)} 个pathlib包:")
    for pf in pathlib_files:
        print(f"  - {pf}")
    
    # 临时重命名pathlib文件
    renamed_files = []
    for pathlib_file in pathlib_files:
        backup_file = pathlib_file.with_suffix(".py.disabled")
        try:
            if backup_file.exists():
                backup_file.unlink()  # 删除旧的备份
            pathlib_file.rename(backup_file)
            renamed_files.append((pathlib_file, backup_file))
            print(f"✓ 临时重命名: {pathlib_file.name}")
        except PermissionError:
            print(f"⚠️  权限不足，跳过: {pathlib_file}")
        except Exception as e:
            print(f"⚠️  重命名失败: {pathlib_file} - {e}")
    
    return pathlib_files, renamed_files

def restore_pathlib_files(renamed_files):
    """恢复pathlib文件"""
    if not renamed_files:
        return
    
    print("🔄 恢复pathlib文件...")
    for original_file, backup_file in renamed_files:
        try:
            if backup_file.exists():
                backup_file.rename(original_file)
                print(f"✓ 恢复: {backup_file} -> {original_file}")
        except Exception as e:
            print(f"⚠️  恢复失败: {backup_file} - {e}")

def create_production_config():
    """创建适用于生产环境的配置文件"""
    print("🔧 创建生产环境配置文件...")
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("❌ config.yaml 文件不存在")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    changes_made = False
    
    # 修复 RAG working_dir 的硬编码路径
    if 'rag' in config and 'lightrag' in config['rag']:
        old_working_dir = config['rag']['lightrag'].get('working_dir', '')
        if old_working_dir and Path(old_working_dir).is_absolute():
            config['rag']['lightrag']['working_dir'] = './working'
            print(f"  ✓ 修复 RAG working_dir: {old_working_dir} -> ./working")
            changes_made = True
    
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

def build_with_pyinstaller():
    """使用PyInstaller构建可执行文件"""
    print("🚀 开始构建可执行文件...")
    
    # 创建简化的构建命令
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',  # 打包成单个文件
        '--clean',
        '--noconfirm',
        '--name=qa_gen_pipeline',
        '--add-data=config.yaml;.',
        '--add-data=src;src',
        '--hidden-import=lightrag',
        '--hidden-import=openai',
        '--hidden-import=requests',
        '--hidden-import=loguru',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        '--hidden-import=transformers',
        '--hidden-import=torch',
        '--hidden-import=PyPDF2',
        '--hidden-import=pdf2image',
        '--hidden-import=PIL',
        '--hidden-import=pytesseract',
        '--hidden-import=networkx',
        '--hidden-import=asyncio',
        '--hidden-import=aiohttp',
        '--hidden-import=httpx',
        '--hidden-import=pydantic',
        '--exclude-module=tkinter',
        '--exclude-module=matplotlib',
        '--exclude-module=IPython',
        '--exclude-module=jupyter',
        '--exclude-module=pytest',
        'main.py'
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
        if os.name != 'nt':
            os.chmod(exe_target, 0o755)
        print(f"✓ 复制可执行文件: {exe_target}")
    else:
        print(f"❌ 可执行文件不存在: {exe_source}")
        return False
    
    # 复制配置文件
    if Path("config.yaml").exists():
        shutil.copy2("config.yaml", deploy_dir / "config.yaml")
        print("✓ 复制配置文件: config.yaml")
    
    # 创建环境变量示例文件
    env_example = deploy_dir / ".env.example"
    with open(env_example, 'w', encoding='utf-8') as f:
        f.write("""# QA生成管道环境变量配置
# 复制此文件为 .env 并填入实际的API密钥

# 必需配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 可选配置
OPENAI_API_KEY=your_openai_api_key_here
""")
    print("✓ 创建环境变量示例文件")
    
    # 创建目录结构
    dirs_to_create = ["working", "output", "logs", "temp"]
    for dir_name in dirs_to_create:
        (deploy_dir / dir_name).mkdir(exist_ok=True)
        (deploy_dir / dir_name / ".gitkeep").touch()
    print("✓ 创建目录结构")
    
    # 创建使用说明
    readme_content = f"""# QA生成管道部署包

## 快速开始

1. **配置环境变量**：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填入你的API密钥
   ```

2. **运行程序**：
   ```bash
   # Windows
   qa_gen_pipeline.exe generate-answers questions.jsonl ./working output.jsonl
   
   # Linux/macOS
   ./qa_gen_pipeline generate-answers questions.jsonl ./working output.jsonl
   ```

## 命令说明

- `generate-answers`: 生成答案
- `show-progress`: 查看进度
- `--help`: 显示帮助信息

## 文件说明

- `{exe_name}`: 主程序
- `config.yaml`: 配置文件
- `.env`: 环境变量（需要从.env.example复制）
- `working/`: 知识库目录
- `output/`: 输出目录
- `logs/`: 日志目录

## 注意事项

1. 必须配置DEEPSEEK_API_KEY
2. 确保网络连接正常
3. 首次运行可能需要较长时间
"""
    
    with open(deploy_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✓ 创建使用说明")
    
    return True

def cleanup():
    """清理临时文件"""
    print("🧹 清理临时文件...")
    
    cleanup_items = ["build", "dist", "__pycache__", "*.spec"]
    
    for item in cleanup_items:
        if item == "*.spec":
            import glob
            for spec_file in glob.glob("*.spec"):
                try:
                    os.remove(spec_file)
                    print(f"✓ 删除spec文件: {spec_file}")
                except:
                    pass
        else:
            path = Path(item)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"✓ 删除目录: {path}")
                else:
                    path.unlink()
                    print(f"✓ 删除文件: {path}")

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
    print("🔧 QA生成管道可执行文件打包工具 v2")
    print("=" * 50)
    
    # 检查当前目录
    if not Path("main.py").exists():
        print("❌ 错误: 请在项目根目录运行此脚本")
        sys.exit(1)
    
    renamed_files = []
    
    try:
        # 1. 检查依赖
        if not check_dependencies():
            sys.exit(1)
        
        # 2. 处理pathlib冲突
        pathlib_files, renamed_files = handle_pathlib_conflict()
        
        # 3. 创建生产环境配置
        if not create_production_config():
            print("❌ 配置文件处理失败")
            sys.exit(1)
        
        # 4. 构建可执行文件
        if not build_with_pyinstaller():
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
        print("\n💡 测试打包结果: python test_build.py")
        
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断")
        restore_original_config()
        restore_pathlib_files(renamed_files)
        cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        restore_original_config()
        restore_pathlib_files(renamed_files)
        cleanup()
        sys.exit(1)
    finally:
        # 确保恢复pathlib文件
        restore_pathlib_files(renamed_files)

if __name__ == "__main__":
    main() 