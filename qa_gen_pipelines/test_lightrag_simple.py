"""Simple test script for LightRAG functionality."""

import os
from pathlib import Path


def test_lightrag_query():
    """Test LightRAG query functionality using existing working directory."""
    try:
        # Import LightRAG
        from lightrag import LightRAG, QueryParam
        
        print("✅ LightRAG imported successfully")
        
        # Set the working directory to the existing knowledge base
        working_dir = r"D:\Project\lightrag\selection_workdir"
        print(f"📁 Using working directory: {working_dir}")
        
        # Check if working directory exists and has data
        wd_path = Path(working_dir)
        if not wd_path.exists():
            print(f"❌ Working directory does not exist: {working_dir}")
            return
        
        # List files in working directory
        files = list(wd_path.glob("*"))
        print(f"📊 Found {len(files)} files in working directory:")
        for file in files[:5]:  # Show first 5 files
            size_mb = file.stat().st_size / (1024 * 1024) if file.is_file() else 0
            print(f"  - {file.name}: {size_mb:.2f} MB")
        
        # Initialize LightRAG with the existing working directory
        # Note: We'll try different initialization approaches
        print("\n🔧 Initializing LightRAG...")
        
        try:
            # Try with minimal initialization - LightRAG should load existing data
            rag = LightRAG(working_dir=working_dir)
            print("✅ LightRAG initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize LightRAG: {e}")
            print("This might be because LightRAG requires API keys or specific setup")
            return
        
        # Test queries
        test_questions = [
            "这个文档主要讲了什么内容？",
            "有哪些重要的技术特性？",
            "文档中提到了哪些关键概念？"
        ]
        
        print("\n🔍 Testing LightRAG queries:")
        
        for i, question in enumerate(test_questions, 1):
            try:
                print(f"\nQuestion {i}: {question}")
                
                # Try different query modes
                modes = ["local", "global", "hybrid", "naive"]
                
                for mode in modes:
                    try:
                        print(f"  Trying {mode} mode...")
                        
                        if mode == "naive":
                            # Simple query without mode
                            response = rag.query(question)
                        else:
                            # Query with specific mode
                            response = rag.query(question, param=QueryParam(mode=mode))
                        
                        # Show response preview
                        response_preview = response[:200] + "..." if len(response) > 200 else response
                        print(f"  ✅ {mode} mode response ({len(response)} chars): {response_preview}")
                        break  # Success, move to next question
                        
                    except Exception as e:
                        print(f"  ⚠️ {mode} mode failed: {e}")
                        continue
                
                # Limit to 2 questions for initial test
                if i >= 2:
                    break
                    
            except Exception as e:
                print(f"❌ Question {i} failed completely: {e}")
        
        print("\n🎉 LightRAG test completed!")
        
    except ImportError as e:
        print(f"❌ Failed to import LightRAG: {e}")
        print("Please ensure LightRAG is installed with: pip install lightrag-hku")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    test_lightrag_query() 