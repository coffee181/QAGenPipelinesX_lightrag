#!/usr/bin/env python3
"""切换到本地模型的简单脚本"""

import re
from pathlib import Path

def update_config_for_local():
    """更新配置文件以使用本地模型"""
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("❌ 配置文件不存在: config.yaml")
        return False
    
    print("正在更新配置文件以使用本地模型...")
    
    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有本地配置
    if "question_generator:" in content and "local:" in content:
        print("✓ 配置文件已包含本地模型配置")
        
        # 只需要更新provider
        if 'provider: "deepseek"' in content:
            content = content.replace('provider: "deepseek"', 'provider: "local"')
            print("✓ 已切换到本地模型")
        elif 'provider: "local"' in content:
            print("✓ 已在使用本地模型")
        else:
            # 添加provider配置
            content = content.replace(
                'question_generator:',
                'question_generator:\n  provider: "local"'
            )
            print("✓ 已添加本地模型配置")
    else:
        # 添加完整的本地配置
        local_config = '''
  # 本地模型配置
  local:
    model_name: "deepseek-r1:32b"
    base_url: "http://localhost:11434"
    max_tokens: 2048
    temperature: 0.7
    timeout: 120
    questions_per_chunk: 30
'''
        
        # 在question_generator部分添加配置
        if 'question_generator:' in content:
            content = content.replace(
                'question_generator:',
                f'question_generator:\n  provider: "local"{local_config}'
            )
        else:
            # 如果找不到question_generator，在文件末尾添加
            content += f'''
# Question Generation Configuration
question_generator:
  provider: "local"{local_config}
'''
        print("✓ 已添加完整的本地模型配置")
    
    # 写回配置文件
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 配置文件已更新")
    return True

def update_main_py():
    """更新main.py以支持本地模型"""
    main_file = Path("main.py")
    
    if not main_file.exists():
        print("❌ main.py文件不存在")
        return False
    
    print("正在更新main.py以支持本地模型...")
    
    # 读取文件
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经导入了LocalQuestionGenerator
    if "from src.implementations.local_question_generator import LocalQuestionGenerator" not in content:
        # 添加导入
        import_line = "from src.implementations.local_question_generator import LocalQuestionGenerator"
        content = content.replace(
            "from src.implementations.deepseek_question_generator import DeepSeekQuestionGenerator",
            f"from src.implementations.deepseek_question_generator import DeepSeekQuestionGenerator\n{import_line}"
        )
        print("✓ 已添加LocalQuestionGenerator导入")
    
    # 检查是否已经有本地模型选择逻辑
    if "provider = config.get" not in content:
        # 替换简单的question_generator创建
        old_code = "    question_generator = DeepSeekQuestionGenerator(config)"
        new_code = '''    # 根据配置选择问题生成器
    provider = config.get("question_generator.provider", "deepseek")
    if provider == "local":
        question_generator = LocalQuestionGenerator(config)
        console_log(f"使用本地模型: {config.get('question_generator.local.model_name', 'unknown')}")
    else:
        question_generator = DeepSeekQuestionGenerator(config)
        console_log("使用DeepSeek API")'''
        
        content = content.replace(old_code, new_code)
        print("✓ 已添加本地模型选择逻辑")
    
    # 写回文件
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ main.py已更新")
    return True

def main():
    """主函数"""
    print("=== 切换到本地模型 ===")
    print()
    
    # 更新配置文件
    if not update_config_for_local():
        return
    
    print()
    
    # 更新main.py
    if not update_main_py():
        return
    
    print()
    print("🎉 切换完成!")
    print()
    print("现在您可以:")
    print("1. 运行 python test_deepseek_32b.py 测试模型")
    print("2. 运行 python main.py 开始使用本地模型")
    print("3. 享受免费的本地AI服务!")
    print()
    print("如需切换回API，请修改config.yaml中的provider为'deepseek'")

if __name__ == "__main__":
    main()
