#!/usr/bin/env python3
"""Check the output file content."""

import json
from pathlib import Path

def check_output_file(file_path):
    """Check the content of output file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File does not exist: {file_path}")
        return
    
    print(f"Checking file: {file_path}")
    print(f"File size: {file_path.stat().st_size} bytes")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Content length: {len(content)} characters")
        
        if content.strip():
            try:
                # Try to parse as JSON
                data = json.loads(content.strip())
                print(f"Valid JSON found")
                print(f"Keys: {list(data.keys())}")
                
                if "messages" in data:
                    messages = data["messages"]
                    print(f"Messages count: {len(messages)}")
                    print("First 3 messages:")
                    for i, msg in enumerate(messages[:3]):
                        print(f"  Message {i+1}: role={msg.get('role')}, content_length={len(msg.get('content', ''))}")
                        print(f"    Content preview: {msg.get('content', '')[:100]}...")
                        print()
                else:
                    print("No 'messages' key found")
                    
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}")
                print(f"Content preview: {content[:200]}...")
        else:
            print("File is empty")
                
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_output_file("C:/Users/26423/Desktop/testoutput/test_fixed.jsonl") 