"""控制台输出工具模块

解决Windows控制台中文字符乱码问题，提供安全的控制台输出功能。
"""

import sys
import os
import locale
import codecs
from typing import Any, TextIO, Optional


class ConsoleOutputFixer:
    """控制台输出修复工具"""
    
    _console_fixed = False
    _original_encoding = None
    
    @classmethod
    def fix_console_encoding(cls) -> bool:
        """
        修复控制台编码问题，确保中文字符正确显示
        
        Returns:
            是否成功修复编码
        """
        if cls._console_fixed:
            return True
            
        try:
            # 保存原始编码信息
            cls._original_encoding = {
                'stdout': getattr(sys.stdout, 'encoding', 'utf-8'),
                'stderr': getattr(sys.stderr, 'encoding', 'utf-8'),
                'locale': locale.getpreferredencoding()
            }
            
            # Windows系统特殊处理
            if os.name == 'nt':
                # 尝试设置控制台代码页为UTF-8
                try:
                    # Windows 10 1903+支持UTF-8代码页
                    import subprocess
                    subprocess.run(['chcp', '65001'], 
                                 capture_output=True, check=False)
                except Exception:
                    pass
                
                # 重新包装sys.stdout和sys.stderr
                if hasattr(sys.stdout, 'buffer'):
                    sys.stdout = codecs.getwriter('utf-8')(
                        sys.stdout.buffer, errors='replace'
                    )
                if hasattr(sys.stderr, 'buffer'):
                    sys.stderr = codecs.getwriter('utf-8')(
                        sys.stderr.buffer, errors='replace'
                    )
            
            # 设置环境变量
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            cls._console_fixed = True
            return True
            
        except Exception as e:
            # 静默失败，避免影响主程序
            print(f"Warning: Failed to fix console encoding: {e}", file=sys.stderr)
            return False
    
    @classmethod
    def get_encoding_info(cls) -> dict:
        """获取当前编码信息"""
        return {
            'stdout_encoding': getattr(sys.stdout, 'encoding', 'unknown'),
            'stderr_encoding': getattr(sys.stderr, 'encoding', 'unknown'),
            'locale_encoding': locale.getpreferredencoding(),
            'fs_encoding': sys.getfilesystemencoding(),
            'default_encoding': sys.getdefaultencoding(),
            'console_fixed': cls._console_fixed,
            'platform': os.name
        }


def safe_print(*args, sep: str = ' ', end: str = '\n', 
               file: Optional[TextIO] = None, flush: bool = False) -> None:
    """
    安全的打印函数，自动处理编码问题
    
    Args:
        *args: 要打印的参数
        sep: 分隔符
        end: 结束符
        file: 输出文件流
        flush: 是否立即刷新
    """
    if file is None:
        file = sys.stdout
    
    try:
        # 转换所有参数为字符串并处理编码
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg)
            elif isinstance(arg, bytes):
                # 尝试解码bytes
                try:
                    safe_args.append(arg.decode('utf-8'))
                except UnicodeDecodeError:
                    safe_args.append(arg.decode('utf-8', errors='replace'))
            else:
                safe_args.append(str(arg))
        
        # 构建输出字符串
        output = sep.join(safe_args) + end
        
        # 尝试正常输出
        try:
            file.write(output)
            if flush:
                file.flush()
        except UnicodeEncodeError:
            # 如果编码失败，使用replace策略
            safe_output = output.encode(
                file.encoding or 'utf-8', errors='replace'
            ).decode(file.encoding or 'utf-8')
            file.write(safe_output)
            if flush:
                file.flush()
                
    except Exception as e:
        # 最后的备选方案：使用repr
        try:
            fallback_output = f"[ENCODING_ERROR] {repr(args)}{end}"
            file.write(fallback_output)
            if flush:
                file.flush()
        except Exception:
            # 如果连这个都失败了，直接写入字节流
            try:
                if hasattr(file, 'buffer'):
                    file.buffer.write(f"[FATAL_ENCODING_ERROR] {repr(args)}\n".encode('utf-8'))
                    if flush:
                        file.buffer.flush()
            except Exception:
                pass  # 彻底放弃


def console_log(level: str, message: str) -> None:
    """
    控制台专用日志函数
    
    Args:
        level: 日志级别 (INFO, WARNING, ERROR, etc.)
        message: 日志消息
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 根据级别选择输出流
    if level.upper() in ['ERROR', 'CRITICAL']:
        file_stream = sys.stderr
    else:
        file_stream = sys.stdout
    
    # 格式化消息
    formatted_msg = f"[{timestamp}] {level.upper()}: {message}"
    
    # 安全输出
    safe_print(formatted_msg, file=file_stream, flush=True)


def print_with_emoji(emoji: str, message: str, level: str = "INFO") -> None:
    """
    带表情符号的控制台输出
    
    Args:
        emoji: 表情符号
        message: 消息内容
        level: 消息级别
    """
    # 在Windows CMD中，某些emoji可能不能正确显示，提供备选方案
    if os.name == 'nt':
        # Windows下的安全emoji映射
        emoji_map = {
            '🚀': '[START]',
            '✓': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARNING]',
            '📦': '[PACKAGE]',
            '🎉': '[SUCCESS]',
            '🔧': '[FIX]',
            '📁': '[FOLDER]',
            '🔍': '[SEARCH]'
        }
        
        # 尝试使用emoji，如果不支持则使用备选
        try:
            safe_print(f"{emoji} {message}")
        except (UnicodeEncodeError, UnicodeDecodeError):
            safe_emoji = emoji_map.get(emoji, '[INFO]')
            safe_print(f"{safe_emoji} {message}")
    else:
        # Unix系统通常支持emoji
        safe_print(f"{emoji} {message}")


def test_console_output() -> bool:
    """
    测试控制台输出功能
    
    Returns:
        测试是否成功
    """
    try:
        # 输出编码信息
        encoding_info = ConsoleOutputFixer.get_encoding_info()
        safe_print("控制台编码信息:")
        for key, value in encoding_info.items():
            safe_print(f"  {key}: {value}")
        
        # 测试中文输出
        test_messages = [
            "测试中文字符输出",
            "GSK 27i高端多通道系统",
            "文件路径：D:/测试目录/中文文档.pdf",
            "错误：无法处理包含中文的文件名"
        ]
        
        safe_print("\n测试中文消息输出:")
        for i, msg in enumerate(test_messages, 1):
            safe_print(f"{i}. {msg}")
        
        # 测试emoji输出
        safe_print("\n测试emoji输出:")
        print_with_emoji("🚀", "程序启动")
        print_with_emoji("✓", "操作成功") 
        print_with_emoji("❌", "操作失败")
        print_with_emoji("🎉", "任务完成")
        
        # 测试控制台日志
        safe_print("\n测试控制台日志:")
        console_log("INFO", "这是一条信息日志")
        console_log("WARNING", "这是一条警告日志")
        console_log("ERROR", "这是一条错误日志")
        
        return True
        
    except Exception as e:
        safe_print(f"控制台输出测试失败: {e}")
        return False


# 模块初始化时自动修复控制台编码
if not ConsoleOutputFixer._console_fixed:
    ConsoleOutputFixer.fix_console_encoding() 