"""
线程本地事件循环管理器
确保每个线程都有独立的事件循环，避免跨线程污染
"""
import asyncio
import threading
from contextlib import contextmanager
from loguru import logger
from typing import Any, Callable, TypeVar

T = TypeVar('T')

# 线程本地存储
_thread_local = threading.local()


def get_or_create_event_loop():
    """获取或创建当前线程的事件循环"""
    try:
        # 尝试获取当前线程的事件循环
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        # 没有运行中的循环，检查是否有已设置的循环
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
            return loop
        except RuntimeError:
            # 创建新的事件循环
            logger.debug(f"Creating new event loop for thread {threading.current_thread().ident}")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


@contextmanager
def thread_safe_event_loop():
    """
    上下文管理器：为当前线程获取事件循环
    
    使用示例:
        with thread_safe_event_loop() as loop:
            result = loop.run_until_complete(my_async_func())
    """
    loop = get_or_create_event_loop()
    try:
        yield loop
    finally:
        # 不关闭循环，只在线程结束时关闭
        pass


def cleanup_thread_event_loop():
    """清理当前线程的事件循环"""
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            # 取消所有待处理的任务
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # 关闭循环
            loop.close()
            logger.debug(f"Cleaned up event loop for thread {threading.current_thread().ident}")
    except Exception as e:
        logger.debug(f"Error cleaning up event loop: {e}")


def run_async_in_thread(coro, timeout=None):
    """
    在当前线程中运行异步协程
    
    Args:
        coro: 要执行的协程
        timeout: 超时时间（秒）
        
    Returns:
        协程的返回值
    """
    loop = get_or_create_event_loop()
    
    # 检查是否已经在运行中的循环内
    try:
        asyncio.get_running_loop()
        # 已经在循环内，直接用 create_task
        return asyncio.ensure_future(coro)
    except RuntimeError:
        # 不在循环内，用 run_until_complete
        try:
            if timeout:
                return asyncio.wait_for(loop.run_until_complete(coro), timeout=timeout)
            else:
                return loop.run_until_complete(coro)
        except asyncio.TimeoutError:
            logger.error(f"Async operation timeout after {timeout} seconds")
            raise
