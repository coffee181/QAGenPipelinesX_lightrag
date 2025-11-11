#!/usr/bin/env python3
"""
测试lightrag导入
"""

print("测试lightrag导入...")

try:
    import lightrag
    print(f"✓ lightrag导入成功，版本: {lightrag.__version__ if hasattr(lightrag, '__version__') else '未知'}")
    print(f"  lightrag路径: {lightrag.__file__}")
    
    try:
        from lightrag import LightRAG, QueryParam
        print("✓ LightRAG和QueryParam导入成功")
    except ImportError as e:
        print(f"❌ LightRAG或QueryParam导入失败: {e}")
    
    try:
        from lightrag.utils import EmbeddingFunc
        print("✓ EmbeddingFunc导入成功")
    except ImportError as e:
        print(f"❌ EmbeddingFunc导入失败: {e}")
        
except ImportError as e:
    print(f"❌ lightrag导入失败: {e}")
    print("尝试安装lightrag...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightrag"])
        print("✓ lightrag安装成功，重新测试导入...")
        import lightrag
        print(f"✓ lightrag导入成功，版本: {lightrag.__version__ if hasattr(lightrag, '__version__') else '未知'}")
    except Exception as install_error:
        print(f"❌ lightrag安装失败: {install_error}")

print("测试完成") 