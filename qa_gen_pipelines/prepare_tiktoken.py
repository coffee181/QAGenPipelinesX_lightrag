#!/usr/bin/env python3
"""
预下载tiktoken编码文件脚本

确保在构建可执行文件前，所有必要的tiktoken编码都已下载并缓存
"""

import os
import sys
import subprocess
from pathlib import Path

def check_and_install_tiktoken():
    """检查并安装/升级tiktoken到最新版本"""
    print("📦 检查tiktoken版本...")
    
    try:
        import tiktoken
        current_version = tiktoken.__version__
        print(f"  当前tiktoken版本: {current_version}")
    except ImportError:
        print("  tiktoken未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken>=0.8.0"])
        import tiktoken
        current_version = tiktoken.__version__
        print(f"  已安装tiktoken版本: {current_version}")
    
    # 检查是否需要升级
    try:
        from packaging import version
        if version.parse(current_version) < version.parse("0.8.0"):
            print("  升级tiktoken到最新版本...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken>=0.8.0", "--upgrade"])
    except ImportError:
        print("  无法检查版本，尝试升级...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken", "--upgrade"])

def download_encodings():
    """预下载所有必要的编码"""
    print("🔄 预下载tiktoken编码...")
    
    import tiktoken
    
    encodings_to_download = [
        "cl100k_base",  # GPT-3.5/GPT-4基础编码
        "o200k_base",   # GPT-4o编码
        "p50k_base",    # 旧版编码
        "r50k_base"     # 旧版编码
    ]
    
    success_count = 0
    for encoding_name in encodings_to_download:
        try:
            print(f"  下载编码: {encoding_name}")
            encoding = tiktoken.get_encoding(encoding_name)
            print(f"    ✓ 成功: {encoding_name} (词汇表大小: {encoding.n_vocab})")
            success_count += 1
        except Exception as e:
            print(f"    ❌ 失败: {encoding_name} - {e}")
    
    print(f"✓ 编码下载完成: {success_count}/{len(encodings_to_download)} 个成功")
    return success_count > 0

def verify_encodings():
    """验证编码是否可用"""
    print("🔍 验证编码可用性...")
    
    import tiktoken
    
    test_text = "Hello, this is a test message for tiktoken encoding."
    
    encodings_to_test = ["cl100k_base", "o200k_base"]
    
    for encoding_name in encodings_to_test:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            tokens = encoding.encode(test_text)
            decoded = encoding.decode(tokens)
            
            print(f"  ✓ {encoding_name}: {len(tokens)} tokens")
            assert decoded == test_text, "编码/解码不匹配"
            
        except Exception as e:
            print(f"  ❌ {encoding_name}: {e}")

def get_cache_info():
    """获取tiktoken缓存信息"""
    print("📁 tiktoken缓存信息:")
    
    import tiktoken
    
    # 尝试获取缓存目录
    cache_dir = None
    if hasattr(tiktoken, 'get_cache_dir'):
        cache_dir = tiktoken.get_cache_dir()
    else:
        # 检查环境变量
        cache_dir = os.environ.get('TIKTOKEN_CACHE_DIR')
        if not cache_dir:
            # 默认缓存位置
            import tempfile
            cache_dir = Path(tempfile.gettempdir()) / "data-gym-cache"
    
    if cache_dir and Path(cache_dir).exists():
        cache_files = list(Path(cache_dir).glob("*"))
        print(f"  缓存目录: {cache_dir}")
        print(f"  缓存文件数: {len(cache_files)}")
        for f in cache_files[:5]:  # 只显示前5个文件
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name} ({size_mb:.1f} MB)")
        if len(cache_files) > 5:
            print(f"    ... 还有 {len(cache_files) - 5} 个文件")
    else:
        print(f"  缓存目录不存在: {cache_dir}")

def test_lightrag_compatibility():
    """测试与LightRAG的兼容性"""
    print("🧪 测试LightRAG兼容性...")
    
    try:
        # 模拟LightRAG的tiktoken使用方式
        import tiktoken
        
        # 测试默认编码
        try:
            encoding = tiktoken.get_encoding("o200k_base")
            print("  ✓ o200k_base编码可用")
        except Exception as e:
            print(f"  ⚠️ o200k_base不可用，尝试cl100k_base: {e}")
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                print("  ✓ cl100k_base编码可用作备用")
            except Exception as e2:
                print(f"  ❌ cl100k_base也不可用: {e2}")
                return False
        
        # 测试编码功能
        test_text = "这是一个中英文混合的测试文本 This is a mixed test text"
        tokens = encoding.encode(test_text)
        decoded = encoding.decode(tokens)
        
        print(f"  ✓ 编码测试通过: {len(tokens)} tokens")
        return True
        
    except Exception as e:
        print(f"  ❌ LightRAG兼容性测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 tiktoken编码准备工具")
    print("=" * 50)
    
    try:
        # 1. 检查和安装tiktoken
        check_and_install_tiktoken()
        print()
        
        # 2. 下载编码
        if download_encodings():
            print()
            
            # 3. 验证编码
            verify_encodings()
            print()
            
            # 4. 显示缓存信息
            get_cache_info()
            print()
            
            # 5. 测试LightRAG兼容性
            if test_lightrag_compatibility():
                print("\n🎉 tiktoken准备完成！现在可以构建可执行文件了。")
                return True
            else:
                print("\n❌ LightRAG兼容性测试失败")
                return False
        else:
            print("\n❌ 编码下载失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 准备过程失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 