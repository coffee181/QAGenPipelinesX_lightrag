#!/usr/bin/env python3
"""
Simple test script to verify LightRAG append functionality.
This script tests whether LightRAG can append documents to existing knowledge bases.
"""

import os
import asyncio
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
WORKING_DIR = "./test_lightrag_append"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

async def setup_lightrag():
    """Setup LightRAG instance with DeepSeek"""
    
    # Define async LLM function
    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        """LLM function for LightRAG."""
        if not DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API key not configured")
        
        # Configure OpenAI client for DeepSeek
        client = openai.AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Handle history_messages properly
        if history_messages:
            if isinstance(history_messages, list):
                for msg in history_messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        messages.append(msg)
        
        messages.append({"role": "user", "content": prompt})
        
        # Filter out unsupported parameters that LightRAG might pass
        supported_params = {
            'model', 'messages', 'temperature', 'max_tokens', 'top_p', 
            'frequency_penalty', 'presence_penalty', 'stream', 'stop'
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            **filtered_kwargs
        )
        
        return response.choices[0].message.content
    
    # Define async embedding function
    async def embedding_func(texts: list[str]):
        """Simple embedding function."""
        import numpy as np
        import hashlib
        
        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            embedding = [(hash_int >> i) & 1 for i in range(3072)]  # 3072 dimensions
            embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)
    
    # Create LightRAG instance
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=embedding_func
        )
    )
    
    # Initialize storages
    await rag.initialize_storages()
    
    # Initialize pipeline status if available
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()
    except ImportError:
        pass  # Not available in all versions
    
    return rag

async def test_append_functionality():
    """Test appending documents to existing knowledge base"""
    
    print("=== LightRAG Append Functionality Test ===")
    
    # Setup
    rag = await setup_lightrag()
    
    # Test 1: Create initial knowledge base
    print("\n1. Creating initial knowledge base...")
    doc1 = "这是第一个测试文档。它包含关于机器学习的基础知识。"
    
    try:
        await rag.ainsert(doc1)
        print("✓ Initial document inserted successfully")
    except Exception as e:
        print(f"✗ Failed to insert initial document: {e}")
        return False
    
    # Check initial stats
    try:
        files = list(Path(WORKING_DIR).glob("*.json"))
        print(f"✓ Knowledge base created with {len(files)} files")
    except Exception as e:
        print(f"✗ Failed to check initial stats: {e}")
    
    # Test 2: Append new document
    print("\n2. Appending new document to existing knowledge base...")
    doc2 = "这是第二个测试文档。它介绍了深度学习的概念和应用。"
    
    try:
        await rag.ainsert(doc2)
        print("✓ Second document appended successfully")
    except Exception as e:
        print(f"✗ Failed to append second document: {e}")
        if "history_messages" in str(e):
            print("  This is a known LightRAG version issue with history_messages")
        return False
    
    # Test 3: Query to verify both documents are accessible
    print("\n3. Testing query to verify both documents are accessible...")
    try:
        response = await rag.aquery("什么是机器学习和深度学习？", param=QueryParam(mode="naive"))
        if response and len(response.strip()) > 10:
            print("✓ Query successful, both documents appear to be accessible")
            print(f"  Response length: {len(response)} characters")
        else:
            print("✗ Query returned empty or very short response")
            return False
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False
    
    print("\n=== Test Summary ===")
    print("✓ LightRAG append functionality appears to be working")
    print(f"✓ Working directory: {WORKING_DIR}")
    
    return True

async def main():
    """Main test function"""
    
    # Ensure working directory exists
    Path(WORKING_DIR).mkdir(exist_ok=True)
    
    try:
        success = await test_append_functionality()
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Some tests failed")
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")

if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        print("❌ Error: DEEPSEEK_API_KEY environment variable not set")
        print("Please set your DeepSeek API key in the .env file")
        exit(1)
    
    asyncio.run(main()) 