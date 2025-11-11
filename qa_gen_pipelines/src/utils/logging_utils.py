"""日志工具模块

提供统一的日志配置，确保中文字符正确显示。
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Union


class UTF8Logger:
    """UTF-8编码的日志工具类"""
    
    @staticmethod
    def setup_logger(
        name: str,
        log_file: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        console_output: bool = True,
        file_output: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> logging.Logger:
        """
        设置UTF-8编码的日志记录器
        
        Args:
            name: 日志记录器名称
            log_file: 日志文件路径
            log_level: 日志级别
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            max_bytes: 日志文件最大大小（字节）
            backup_count: 备份文件数量
            
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        
        # 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            # 强制刷新控制台输出
            console_handler.flush = lambda: sys.stdout.flush()
            logger.addHandler(console_handler)
        
        # 文件处理器
        if file_output and log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用RotatingFileHandler支持日志轮转
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'  # 关键：指定UTF-8编码
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            # 强制刷新文件输出
            file_handler.flush = lambda: file_handler.stream.flush()
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def force_flush_logs():
        """强制刷新所有日志输出"""
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        # 刷新所有日志处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
    
    @staticmethod
    def setup_basic_config(
        log_file: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5
    ) -> None:
        """
        设置基本的日志配置（替代logging.basicConfig）
        
        Args:
            log_file: 日志文件路径
            log_level: 日志级别
            max_bytes: 日志文件最大大小
            backup_count: 备份文件数量
        """
        handlers = []
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        handlers.append(console_handler)
        
        # 文件处理器
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            handlers.append(file_handler)
        
        # 配置基本日志
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True  # 强制重新配置
        )
    
    @staticmethod
    def fix_log_encoding(log_file: Union[str, Path]) -> bool:
        """
        修复现有日志文件的编码问题
        
        Args:
            log_file: 日志文件路径
            
        Returns:
            是否成功修复
        """
        try:
            log_path = Path(log_file)
            if not log_path.exists():
                return True  # 文件不存在，不需要修复
            
            # 尝试用不同编码读取文件
            content = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try:
                    with open(log_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return False  # 无法读取文件
            
            # 重新写入文件，使用UTF-8编码
            backup_path = log_path.with_suffix(f'{log_path.suffix}.backup')
            log_path.rename(backup_path)  # 备份原文件
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception:
            return False


def safe_log_message(message: str) -> str:
    """
    安全地处理日志消息，确保中文字符正确显示
    
    Args:
        message: 原始消息
        
    Returns:
        安全的消息字符串
    """
    try:
        # 如果消息包含乱码，尝试修复
        if '�' in message:
            # 尝试用不同编码解码
            for encoding in ['utf-8', 'gbk', 'gb2312']:
                try:
                    # 先编码为bytes再解码
                    fixed = message.encode('latin1').decode(encoding)
                    if '�' not in fixed:
                        return fixed
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
        
        return message
    except Exception:
        return repr(message)  # 如果都失败了，返回repr表示


def setup_project_logging(log_level: str = "INFO") -> logging.Logger:
    """
    为整个项目设置统一的日志配置
    
    Args:
        log_level: 日志级别
        
    Returns:
        主日志记录器
    """
    # 使用UTF8Logger设置基本配置
    UTF8Logger.setup_basic_config(
        log_file='qa_pipeline.log',
        log_level=log_level,
        max_bytes=10 * 1024 * 1024,  # 10MB
        backup_count=5
    )
    
    return logging.getLogger(__name__) 