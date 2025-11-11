#!/usr/bin/env python3
"""
修复pathlib包兼容性问题的脚本

处理PyInstaller与过时的pathlib包的冲突
"""

import os
import sys
import shutil
from pathlib import Path

def find_pathlib_packages():
    """查找系统中的pathlib包"""
    pathlib_locations = []
    
    # 检查常见的Python包路径
    possible_paths = [
        Path(sys.executable).parent / "lib" / "site-packages",
        Path(sys.executable).parent.parent / "lib" / "site-packages", 
        Path(sys.executable).parent / "Lib" / "site-packages",
        Path("D:/Tools/Anaconda3/lib/site-packages"),
    ]
    
    for base_path in possible_paths:
        if base_path.exists():
            pathlib_pkg = base_path / "pathlib.py"
            pathlib_dir = base_path / "pathlib"
            
            if pathlib_pkg.exists():
                pathlib_locations.append(pathlib_pkg)
                print(f"发现pathlib包: {pathlib_pkg}")
            
            if pathlib_dir.exists() and pathlib_dir.is_dir():
                pathlib_locations.append(pathlib_dir)
                print(f"发现pathlib目录: {pathlib_dir}")
    
    return pathlib_locations

def backup_and_remove_pathlib():
    """备份并移除pathlib包"""
    locations = find_pathlib_packages()
    
    if not locations:
        print("✓ 未发现冲突的pathlib包")
        return True
    
    backup_dir = Path("pathlib_backup")
    backup_dir.mkdir(exist_ok=True)
    
    for location in locations:
        try:
            print(f"正在处理: {location}")
            
            # 创建备份
            backup_name = f"{location.name}_{location.parent.name}"
            backup_path = backup_dir / backup_name
            
            if location.is_file():
                shutil.copy2(location, backup_path)
                location.unlink()
                print(f"✓ 已移除文件: {location}")
            elif location.is_dir():
                shutil.copytree(location, backup_path, dirs_exist_ok=True)
                shutil.rmtree(location)
                print(f"✓ 已移除目录: {location}")
            
            print(f"✓ 已备份到: {backup_path}")
            
        except PermissionError:
            print(f"❌ 权限不足，无法移除: {location}")
            print("尝试以管理员身份运行，或手动删除")
            return False
        except Exception as e:
            print(f"❌ 移除失败 {location}: {e}")
            return False
    
    return True

def restore_pathlib():
    """恢复pathlib包"""
    backup_dir = Path("pathlib_backup")
    if not backup_dir.exists():
        print("❌ 备份目录不存在")
        return False
    
    print("🔄 恢复pathlib包...")
    try:
        # 这里可以添加恢复逻辑
        print("✓ 备份文件保存在 pathlib_backup 目录")
        print("如需恢复，请手动复制回原位置")
        return True
    except Exception as e:
        print(f"❌ 恢复失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 pathlib包兼容性修复工具")
    print("=" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_pathlib()
        return
    
    print("查找pathlib包...")
    locations = find_pathlib_packages()
    
    if not locations:
        print("✅ 未发现冲突的pathlib包，可以继续构建")
        return
    
    print(f"\n发现 {len(locations)} 个pathlib包")
    print("这些包与PyInstaller不兼容，需要移除")
    
    response = input("\n是否继续移除这些包？(y/N): ")
    if response.lower() in ['y', 'yes']:
        if backup_and_remove_pathlib():
            print("\n✅ pathlib包处理完成")
            print("现在可以运行: python build_executable.py")
            print("如需恢复: python fix_pathlib.py restore")
        else:
            print("\n❌ pathlib包处理失败")
    else:
        print("操作已取消")

if __name__ == "__main__":
    main() 