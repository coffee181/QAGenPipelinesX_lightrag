"""超时配置管理模块"""

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def configure_global_timeouts():
    """配置全局超时设置"""
    
    # 设置环境变量
    os.environ['REQUESTS_TIMEOUT'] = '30000'  # 30,000秒 = 500分钟
    os.environ['HTTPX_TIMEOUT'] = '30000'
    
    # 配置requests的默认超时
    requests.adapters.DEFAULT_RETRIES = 3
    requests.adapters.DEFAULT_TIMEOUT = 30000
    
    # 创建自定义的HTTPAdapter
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )
    
    # 配置session
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def get_extended_timeout():
    """获取扩展的超时时间"""
    return 30000  # 500分钟

def configure_ollama_timeout():
    """专门为Ollama配置超时"""
    return {
        'timeout': (60, 30000),  # 连接超时60秒，读取超时30000秒
        'stream': False,
        'verify': False  # 如果是自签名证书
    }
