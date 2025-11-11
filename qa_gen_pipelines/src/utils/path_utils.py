"""路径处理工具模块

处理各种路径编码问题，特别是中文路径的处理。
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional
import locale
import urllib.parse


class PathUtils:
    """路径处理工具类"""
    
    @staticmethod
    def normalize_path(path_input: Union[str, Path]) -> Path:
        """
        标准化路径，处理编码问题
        
        Args:
            path_input: 输入路径（字符串或Path对象）
            
        Returns:
            标准化的Path对象
        """
        if isinstance(path_input, Path):
            return path_input
        
        if isinstance(path_input, str):
            # 处理可能的编码问题
            try:
                # 尝试多种编码解码
                decoded_path = PathUtils._decode_path_string(path_input)
                return Path(decoded_path)
            except Exception:
                # 如果解码失败，直接使用原始字符串
                return Path(path_input)
        
        raise ValueError(f"Unsupported path type: {type(path_input)}")
    
    @staticmethod
    def _decode_path_string(path_str: str) -> str:
        """
        解码路径字符串，处理可能的编码问题
        
        Args:
            path_str: 路径字符串
            
        Returns:
            解码后的路径字符串
        """
        # 如果字符串包含乱码字符，尝试重新解码
        if '�' in path_str or any(ord(c) > 127 and ord(c) < 256 for c in path_str):
            # 尝试常见的编码方式
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp936']
            
            for encoding in encodings:
                try:
                    # 先编码为bytes然后用正确编码解码
                    if isinstance(path_str, str):
                        # 尝试不同的源编码
                        for src_encoding in ['latin1', 'cp1252', 'utf-8']:
                            try:
                                bytes_data = path_str.encode(src_encoding)
                                decoded = bytes_data.decode(encoding)
                                # 检查解码结果是否合理
                                if not ('�' in decoded) and _is_valid_chinese_path(decoded):
                                    return decoded
                            except (UnicodeEncodeError, UnicodeDecodeError):
                                continue
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
        
        return path_str
    
    @staticmethod
    def safe_path_string(path: Union[str, Path]) -> str:
        """
        安全地获取路径字符串表示，避免编码错误
        
        Args:
            path: 路径对象
            
        Returns:
            安全的路径字符串
        """
        try:
            if isinstance(path, Path):
                return str(path)
            elif isinstance(path, str):
                return PathUtils._decode_path_string(path)
            else:
                return str(path)
        except Exception:
            # 如果发生任何错误，返回repr表示
            return repr(path)
    
    @staticmethod
    def validate_path(path: Union[str, Path], 
                     require_exists: bool = False,
                     require_file: bool = False,
                     require_dir: bool = False,
                     allowed_extensions: Optional[list] = None) -> tuple[bool, str]:
        """
        验证路径是否有效
        
        Args:
            path: 要验证的路径
            require_exists: 是否要求路径存在
            require_file: 是否要求是文件
            require_dir: 是否要求是目录
            allowed_extensions: 允许的文件扩展名列表
            
        Returns:
            (是否有效, 错误消息)
        """
        try:
            normalized_path = PathUtils.normalize_path(path)
            
            # 检查路径存在性
            if require_exists and not normalized_path.exists():
                return False, f"Path does not exist: {PathUtils.safe_path_string(normalized_path)}"
            
            # 检查是否为文件
            if require_file and normalized_path.exists() and not normalized_path.is_file():
                return False, f"Path is not a file: {PathUtils.safe_path_string(normalized_path)}"
            
            # 检查是否为目录
            if require_dir and normalized_path.exists() and not normalized_path.is_dir():
                return False, f"Path is not a directory: {PathUtils.safe_path_string(normalized_path)}"
            
            # 检查文件扩展名
            if (allowed_extensions and 
                normalized_path.exists() and 
                normalized_path.is_file() and 
                normalized_path.suffix not in allowed_extensions):
                return False, f"File extension not allowed: {normalized_path.suffix}. Allowed: {allowed_extensions}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Path validation error: {str(e)}"
    
    @staticmethod
    def fix_url_encoded_path(path_str: str) -> str:
        """
        修复URL编码的路径
        
        Args:
            path_str: 可能包含URL编码的路径字符串
            
        Returns:
            解码后的路径字符串
        """
        try:
            # 如果路径包含%编码，尝试URL解码
            if '%' in path_str:
                decoded = urllib.parse.unquote(path_str, encoding='utf-8')
                return decoded
            return path_str
        except Exception:
            return path_str
    
    @staticmethod
    def ensure_utf8_path(path: Union[str, Path]) -> Path:
        """
        确保路径使用UTF-8编码
        
        Args:
            path: 输入路径
            
        Returns:
            UTF-8编码的Path对象
        """
        if isinstance(path, Path):
            return path
        
        try:
            # 处理可能的URL编码
            path_str = PathUtils.fix_url_encoded_path(str(path))
            
            # 处理可能的编码问题
            decoded_str = PathUtils._decode_path_string(path_str)
            
            return Path(decoded_str)
        except Exception:
            return Path(str(path))


def _is_valid_chinese_path(path_str: str) -> bool:
    """
    检查路径字符串是否包含有效的中文字符
    
    Args:
        path_str: 路径字符串
        
    Returns:
        是否包含有效中文字符
    """
    # 检查是否包含中文字符
    for char in path_str:
        if '\u4e00' <= char <= '\u9fff':  # CJK统一汉字
            return True
    return False


def create_safe_filename(filename: str, max_length: int = 255) -> str:
    """
    创建安全的文件名，处理中文字符和特殊字符
    
    Args:
        filename: 原始文件名
        max_length: 最大长度限制
        
    Returns:
        安全的文件名
    """
    # 移除不安全的字符，但保留中文
    import re
    
    # 保留字母、数字、中文、常见符号
    safe_chars = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # 限制长度
    if len(safe_chars.encode('utf-8')) > max_length:
        # 截断时要注意不能截断中文字符
        truncated = safe_chars
        while len(truncated.encode('utf-8')) > max_length:
            truncated = truncated[:-1]
        return truncated
    
    return safe_chars 