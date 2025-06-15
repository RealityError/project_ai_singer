# -*- coding: utf-8 -*-
# @Time    : 2025/06/15
# @Author  : Gemini AI
# @File    : thread_utils.py
# @Software: Python 3
"""
该模块提供了简化线程创建和管理的工具。

通过从外部导入，即可方便地将任务异步化，或管理一个线程池来处理多个并发任务。
模块的核心是围绕 Python 内置的 `concurrent.futures.ThreadPoolExecutor` 构建的，
提供了更高级、更安全的线程操作接口。

主要功能:
- @run_in_thread 装饰器: 轻松将一个函数变为异步执行。
- ThreadPoolManager 类: 用于管理一个线程池的生命周期和任务提交。
- thread_pool 全局实例: 一个默认的线程池管理器，开箱即用。

用法:
    ```python
    from thread_utils import run_in_thread, thread_pool
    import time

    # 1. 使用装饰器 (适用于“即发即忘”的任务)
    @run_in_thread
    def long_running_task(name, seconds):
        print(f"任务 '{name}' 开始，将运行 {seconds} 秒...")
        time.sleep(seconds)
        print(f"任务 '{name}' 完成。")

    # 调用后会立即返回，不会阻塞
    long_running_task("装饰器任务", 3)
    print("主线程继续执行...")


    # 2. 使用全局线程池 (适用于需要管理多个任务并获取结果的场景)
    def task_with_return(x, y):
        print(f"线程池任务开始: {x} + {y}")
        time.sleep(1)
        return x + y

    # 提交任务到线程池，会返回一个 Future 对象
    future = thread_pool.submit(task_with_return, 10, 20)
    print("线程池任务已提交。")

    # future.result() 会阻塞直到任务完成并返回结果
    result = future.result()
    print(f"线程池任务结果: {result}")

    # 使用完毕后，建议关闭线程池
    thread_pool.shutdown()
    ```
"""

import functools
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, Optional

# 尝试从你的日志模块导入logger，如果不存在则使用备用方案
try:
    # 假设你的日志模块在项目根目录的 log.py 文件中
    from utils.loguru_logger import *
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def run_in_thread(func: Callable) -> Callable:
    """
    一个装饰器，使被装饰的函数在一个新的守护线程中执行。

    守护线程 (daemon thread) 会随着主程序的退出而退出。
    这适用于不需要等待其完成的后台任务。

    Args:
        func (Callable): 需要在新线程中执行的目标函数。

    Returns:
        Callable: 一个包装后的函数，调用它会启动线程但立即返回。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> threading.Thread:
        # 创建一个线程来执行目标函数
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        # 设置为守护线程，这样主程序退出时它也会退出
        thread.daemon = True
        thread.start()
        logger.info(f"函数 '{func.__name__}' 已在新的守护线程中启动。")
        return thread
    return wrapper


class ThreadPoolManager:
    """
    一个对 `concurrent.futures.ThreadPoolExecutor` 的封装，简化线程池的管理。
    """
    def __init__(self, max_workers: Optional[int] = None, thread_name_prefix: str = 'ThreadPool'):
        """
        初始化线程池管理器。

        Args:
            max_workers (Optional[int]): 线程池中的最大线程数。
                如果为 `None` 或未提供，则默认为 `os.cpu_count() * 5`。
            thread_name_prefix (str): 线程名称的前缀，便于调试。
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._shutdown = False
        logger.info(f"线程池 '{thread_name_prefix}' 已初始化，最大工作线程数: {self._executor._max_workers}。")

    def submit(self, func: Callable, *args: Any, **kwargs: Any) -> Future:
        """
        向线程池提交一个任务以供执行。

        Args:
            func (Callable): 要执行的函数。
            *args: 函数的位置参数。
            **kwargs: 函数的关键字参数。

        Returns:
            Future: 一个 Future 对象，代表这个任务的未来执行结果。

        Raises:
            RuntimeError: 如果线程池已经关闭。
        """
        if self._shutdown:
            raise RuntimeError("线程池已关闭，无法提交新任务。")

        future = self._executor.submit(func, *args, **kwargs)
        future.add_done_callback(self._handle_task_exception)
        return future

    @staticmethod
    def _handle_task_exception(future: Future):
        """在任务完成后检查是否有异常抛出，并记录它。"""
        exception = future.exception()
        if exception:
            logger.error(f"线程池中的任务执行失败: {exception}", exc_info=exception)

    def shutdown(self, wait: bool = True):
        """
        关闭线程池。

        Args:
            wait (bool): 是否等待所有已提交的任务执行完毕再关闭。
        """
        if not self._shutdown:
            logger.info("正在关闭线程池...")
            self._executor.shutdown(wait=wait)
            self._shutdown = True
            logger.info("线程池已成功关闭。")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# --- 全局实例 ---
# 提供一个默认的全局线程池实例，方便在项目的任何地方直接调用。
# 这类似于您示例中的 `logger` 对象。
thread_pool = ThreadPoolManager(max_workers=10, thread_name_prefix='GlobalPool')


# 定义模块的公共API
__all__ = [
    "run_in_thread",
    "ThreadPoolManager",
    "thread_pool"
]


# --- 示例代码 ---
if __name__ == "__main__":
    import time

    print("--- 1. @run_in_thread 装饰器演示 ---")
    @run_in_thread
    def background_job(duration):
        """一个模拟的后台任务。"""
        print(f"后台任务开始，需要 {duration} 秒。")
        time.sleep(duration)
        # 在这个任务中，我们甚至可以向全局线程池提交子任务
        sub_task_future = thread_pool.submit(lambda: "子任务完成")
        print(f"后台任务完成。子任务结果: {sub_task_future.result()}")

    background_job(2)  # 调用后立即返回
    print("主线程没有被阻塞，可以继续执行其他代码。")
    time.sleep(0.5)  # 等待一会，让后台任务的打印信息显示出来
    
    print("\n" + "="*40 + "\n")

    print("--- 2. 全局 thread_pool 演示 ---")
    
    def add(a, b):
        """一个简单的计算任务。"""
        print(f"线程池任务 [{threading.current_thread().name}] 正在计算 {a} + {b}")
        time.sleep(1)
        return a + b

    def fail_task():
        """一个必定失败的任务，用于演示异常处理。"""
        print(f"线程池任务 [{threading.current_thread().name}] 即将抛出异常")
        time.sleep(0.5)
        raise ValueError("这是一个故意的错误")

    # 提交多个任务
    future1 = thread_pool.submit(add, 5, 10)
    future2 = thread_pool.submit(add, 100, 200)
    future3 = thread_pool.submit(fail_task)

    # 等待并获取结果
    # .result() 方法会阻塞，直到该任务完成
    print(f"任务1的结果: {future1.result()}")
    print(f"任务2的结果: {future2.result()}")

    try:
        # 获取失败任务的结果会重新引发异常
        future3.result()
    except ValueError as e:
        print(f"成功捕获到任务3的异常: {e}")

    # 等待所有后台任务完成，然后关闭线程池
    # 这是良好实践，确保程序干净地退出
    print("所有任务已提交，准备关闭线程池。")
    thread_pool.shutdown()
    print("主线程执行结束。")