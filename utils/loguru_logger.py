# -*- coding: utf-8 -*-
# @Time : 2023/1/13 (Revised)
# @Author : 白猫猫
# @File : log.py
# @Software: Vscode|虚拟环境|3.10.6|64-bit
"""
该模块定义了通用的的日志记录Logger。

通过从外部导入 logger 对象，即可在项目的任何地方使用统一配置的日志系统。

默认信息:
- 格式: 控制台带颜色，文件不带颜色。
- 等级: 从环境变量 `LOG_LEVEL` 读取，默认为 `INFO`。
- 输出: 同时输出至控制台和 log/ 文件夹下的日志文件。

用法:
    ```python
    from your_project.log import logger

    logger.info("这是一条普通信息")
    logger.debug("这是一条用于调试的信息")
    ```
"""

import os
import sys
from loguru import logger

# --- 核心配置 ---

# 1. 从环境变量读取日志级别，提供一个安全的默认值
#    在终端中可以通过 `export LOG_LEVEL=DEBUG` 来改变日志级别
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 2. 定义日志格式
#    控制台使用的带颜色的格式
CONSOLE_FORMAT = (
    "<g>{time:MM-DD HH:mm:ss}</g> "
    "[<lvl>{level: <8}</lvl>] "
    "<c><u>{name}</u>:{line}</c> | "
    "{message}"
)
#    日志文件使用的纯文本格式
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# --- 初始化日志设置 ---

# 3. 移除 loguru 的默认 handler，避免日志重复输出
logger.remove()

# 4. 添加输出到控制台的 handler
logger.add(
    sys.stderr,                  # 指定接收器为标准错误流
    level=LOG_LEVEL,             # 设置日志级别
    format=CONSOLE_FORMAT,       # 使用带颜色的格式
    colorize=True                # 启用颜色
)

# 5. 添加输出到文件的 handler
try:
    log_dir = os.path.join(os.getcwd(), 'log')
    os.makedirs(log_dir, exist_ok=True) # 使用 exist_ok=True 简化目录创建

    # 使用更标准的日期格式，便于排序和查找
    log_file_path = os.path.join(log_dir, 'app_{time:YYYY-MM-DD}.log')

    logger.add(
        log_file_path,
        level=LOG_LEVEL,
        format=FILE_FORMAT,
        rotation='00:00',          # 每天零点创建一个新文件
        retention='30 days',       # 日志文件最长保留30天
        compression='zip',         # 使用zip格式压缩旧的日志文件
        enqueue=True,              # 确保多线程/多进程安全
        encoding='utf-8'           # 明确指定编码
    )
    logger.info("日志系统初始化成功。日志将记录在 '{}' 文件夹中。", log_dir)

except Exception as e:
    logger.error("设置文件日志时发生错误: {}", e)
    logger.error("将仅使用控制台日志作为后备。")

# 导出 logger 对象，供其他模块使用
__all__ = ["logger"]
