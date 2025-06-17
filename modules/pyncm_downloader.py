# -*- coding: utf-8 -*-
"""
用于登录网易云音乐获取歌曲信息及下载歌曲的模块。

核心功能:
1.  通过 Pydantic 从环境变量加载登录凭据 (NETEASE_PHONE, NETEASE_PASSWORD)。
2.  在模块首次导入时自动登录，并自动保存/加载登录会话，避免重复登录。
3.  `get_song_id_by_name(name)`，可以通过歌名搜索可播放的歌曲ID。
4.  提供 `get_track_audio` 和 `get_track_detail` 等清晰的接口函数。

涉及到的文件夹文件:
- `.env`: (可选) 用于在项目根目录存储登录凭据，比直接设置环境变量更方便。
- `static/pyncm.session`: 用于缓存登录状态的文件，程序会自动创建和管理。
- `static/music_downloads/`: 下载歌曲时默认创建的保存目录。

如何设置环境变量:
    - 在项目根目录创建一个 `.env` 文件，内容如下:
      NETEASE_PHONE="你的手机号"
      NETEASE_PASSWORD="你的密码" 
    - 或者在操作系统环境中直接设置这些变量。
    
    - 注意!!!!!!!!
      1.根据pyncm的issue反馈,自动登陆功能(使用密码)可能会导致登录时报错：“需要行为验证码验证”,
      所以更推荐的自动登录方式是只在.env文件中填写`NETEASE_PHONE`字段。
      这样程序会自动触发短信验证码登录, 成功登陆后你的session会被缓存到`static/pyncm.session`中。
      2.开启系统代理会导致报错,模块默认不提供代理,如果你需要代理请修改"功能所需外部模块"下面的
      GetCurrentSession().proxies = {'http': None, 'https': None}
      !!!!!!!

函数内容:
- `login()`: 模块的智能登录入口，在首次导入时自动执行，处理所有登录逻辑。
- `resolve_song_input(query)`: 智能解析用户输入，自动判断是歌曲ID还是歌名。
- `get_song_id_by_name(name)`: 通过歌名搜索并返回第一个可播放的歌曲 ID。
- `get_track_detail(song_id)`: 获取指定歌曲的详细信息（如歌名、歌手、专辑图等）。
- `get_track_audio(song_id, bitrate)`: 获取指定歌曲的音频信息（包含下载链接和码率）。
- `download_song_by_id(song_id, save_dir)`: 根据歌曲ID下载音乐文件到指定目录。

"""

# python模块
import os
import sys
import threading
from pprint import pprint
import requests
from pathlib import Path
import re
from typing import Optional, Dict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 自写模块
try:
    from utils.loguru_logger import logger
except ImportError:
    # Fallback to standard logging if loguru is not found
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(threadName)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("自写日志模块`loguru_logger`未找到，已切换至标准`logging`模块。")

# 功能所需外部模块
try:
    import inquirer
    from pydantic_settings import BaseSettings
    from pyncm import apis
    from pyncm import SetCurrentSession, LoadSessionFromString, DumpSessionAsString, GetCurrentSession
    # 禁用 pyncm 的代理，避免部分网络环境下的问题
    GetCurrentSession().proxies = {'http': None, 'https': None}
except ImportError as e:
    print(f"依赖库 `{e.name}` 未找到。请运行: pip install pyncm \"pydantic-settings>=2.0.0\" inquirer requests")
    exit(1)


# --- 配置管理 ---
class Settings(BaseSettings):
    """使用 Pydantic 从环境变量或 .env 文件加载配置"""
    netease_phone: str = ""
    netease_password: str = ""
    netease_ctcode: int = 86
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
config = Settings()


# --- 全局变量与锁 ---
SESSION_FILE = Path("static/pyncm.session")
_session_lock = threading.Lock()
_login_attempted = False


# --- 登录逻辑 ---
def _perform_interactive_login() -> bool:
    """内部函数：执行交互式登录流程"""
    print("─" * 40)
    logger.info("自动登录失败或未配置，已进入交互式登录模式。")
    
    # 智能判断是否推荐使用验证码
    should_default_to_captcha = not config.netease_password

    questions = [
        inquirer.Text("phone", message="请输入手机号", default=config.netease_phone or None),
        inquirer.Text("ctcode", message="请输入国家代码 (默认 86)", default=str(config.netease_ctcode)),
        inquirer.Confirm("use_captcha", message="是否使用【短信验证码】登录?", default=should_default_to_captcha)
    ]
    answers = inquirer.prompt(questions)
    phone, ctcode = answers["phone"], answers["ctcode"]

    result = {}
    if answers["use_captcha"]:
        send_result = apis.login.SetSendRegisterVerifcationCodeViaCellphone(phone, ctcode)
        if send_result.get("code", 0) != 200:
            logger.error(f"发送验证码失败: {send_result}")
            return False
        logger.info("验证码已发送，请查收短信。")
        captcha = inquirer.text("请输入收到的验证码")
        result = apis.login.LoginViaCellphone(phone, captcha=captcha, ctcode=ctcode)
    else:
        password = inquirer.password("请输入密码")
        result = apis.login.LoginViaCellphone(phone, password=password, ctcode=ctcode)
    
    if result.get("code") == 200:
        logger.info("交互式登录成功！")
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SESSION_FILE, "w") as f:
            f.write(DumpSessionAsString(GetCurrentSession()))
        logger.info(f"登录状态已保存到 {SESSION_FILE}")
        return True
    else:
        logger.error(f"交互式登录失败: {result}")
        return False

def login():
    """
    模块的智能登录入口，在首次导入时自动执行。
    会按“会话文件 -> 环境变量 -> 交互式登录”的顺序尝试。
    """
    global _login_attempted
    with _session_lock:
        if _login_attempted: return
        _login_attempted = True

        if SESSION_FILE.exists():
            try:
                with open(SESSION_FILE, "r") as f:
                    SetCurrentSession(LoadSessionFromString(f.read()))
                status = apis.login.GetCurrentLoginStatus()
                if status.get('code') == 200 and status.get('profile'):
                    logger.info(f"从文件加载登录状态成功。欢迎您: {status['profile']['nickname']}")
                    return
            except Exception:
                logger.warning("加载本地 session 失败或已过期。")

        if config.netease_phone and config.netease_password:
            logger.info(f"正在尝试使用手机号和密码进行自动登录...")
            result = apis.login.LoginViaCellphone(
                phone=config.netease_phone, password=config.netease_password, ctcode=config.netease_ctcode
            )
            if result.get("code") == 200:
                logger.info("自动登录成功！")
                SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(SESSION_FILE, "w") as f:
                    f.write(DumpSessionAsString(GetCurrentSession()))
                logger.info(f"登录状态已保存到 {SESSION_FILE}")
                return
            else:
                logger.warning(f"自动登录失败: {result.get('message', '密码错误或账号异常')}。将进入交互模式。")
        
        if not _perform_interactive_login():
            logger.critical("所有登录尝试均已失败。模块功能将受限！")

# --- 核心API函数 ---

def get_song_id_by_name(name: str) -> Optional[int]:
    if not name: return None
    logger.info(f"正在搜索歌曲: '{name}'")
    try:
        res = apis.cloudsearch.GetSearchResult(name, limit=10)
        if res['code'] != 200 or res['result']['songCount'] == 0: return None
        for song in res["result"]["songs"]:
            privilege = song.get("privilege", {})
            if privilege.get('playMaxbr', 0) and privilege.get('fee') != 1:
                song_id = song['id']
                logger.info(f"找到匹配歌曲: '{song['name']}' (ID: {song_id})")
                return song_id
        return None
    except Exception as e:
        logger.error(f"搜索歌曲时发生异常: {e}", exc_info=True)
        return None

def resolve_song_input(query: str) -> Optional[int]:
    if not query:
        return None
    
    query = query.strip()
    if query.isdigit():
        logger.info(f"输入被识别为歌曲 ID: {query}")
        return int(query)
    else:
        logger.info(f"输入被识别为歌名，开始搜索: '{query}'")
        return get_song_id_by_name(query)

def get_track_audio(song_id: int, bitrate: int = 999000) -> Optional[Dict]:
    logger.info(f"正在获取歌曲 ID:{song_id} 的音频信息...")
    try:
        info = apis.track.GetTrackAudio(song_ids=[song_id], bitrate=bitrate)
        if info.get('code') == 200 and info.get('data'):
            track_data = info['data'][0]
            if track_data.get('code') == 200 and track_data.get('url'):
                return track_data
            else:
                logger.error(f"获取歌曲音频失败 (ID:{song_id}): 歌曲无法播放或无有效链接。内部状态码: {track_data.get('code')}")
                return None
        else:
            logger.error(f"获取歌曲音频的API调用失败 (ID:{song_id}): {info.get('message')}")
            return None
    except Exception as e:
        logger.error(f"获取歌曲音频时发生异常 (ID:{song_id}): {e}", exc_info=True)
        return None

def get_track_detail(song_id: int) -> Optional[Dict]:
    logger.info(f"正在获取歌曲 ID:{song_id} 的详细信息...")
    try:
        info = apis.track.GetTrackDetail(song_ids=[song_id])
        return info['songs'][0] if info.get('code') == 200 and info.get('songs') else None
    except Exception as e:
        logger.error(f"获取歌曲详情时发生异常 (ID:{song_id}): {e}", exc_info=True)
        return None

def download_song_by_id(song_id: int, save_dir: str = "static/music_downloads") -> Optional[Path]:
    logger.info(f"准备下载歌曲 ID: {song_id}")
    detail = get_track_detail(song_id)
    if not detail:
        logger.error(f"无法获取歌曲详情 (ID: {song_id})，无法生成文件名。")
        return None
    
    song_name = detail.get('name', 'Unknown Song')
    artists = ", ".join([ar.get('name', 'Unknown Artist') for ar in detail.get('ar', [])])
    safe_filename = re.sub(r'[\\/:*?"<>|]', '_', f"{artists} - {song_name}.mp3")

    audio = get_track_audio(song_id)
    if not audio or not audio.get('url'):
        logger.error(f"无法获取下载链接 (ID: {song_id})。")
        return None
    
    url = audio['url']
    
    try:
        logger.info(f"正在从以下链接下载: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        file_path = save_path / safe_filename
        
        total_size = int(response.headers.get('content-length', 0))
        
        logger.info(f"保存至: {file_path}")
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                print(f"\r下载进度: {downloaded/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB ({progress:.1f}%)", end="")
        
        print("\n下载完成！")
        return file_path

    except requests.exceptions.RequestException as e:
        logger.error(f"下载过程中发生网络错误: {e}")
        return None
    except IOError as e:
        logger.error(f"写入文件时发生错误: {e}")
        return None

# --- 模块导入时自动执行登录 ---
login()

# --- 主程序入口（用作示例和测试）---
if __name__ == "__main__":
    print("\n" + "="*20 + " 网易云音乐下载器模块测试 " + "="*20)
    print("模块已加载，登录流程已自动处理完毕。")
    print("提示: 您可以删除目录下的 .env 和 static/pyncm.session 文件来强制触发交互式登录。")
    
    while True:
        user_input = input("\n请输入歌名或歌曲ID (输入 'exit' 退出): ")
        if user_input.lower() == 'exit': break
        
        found_song_id = resolve_song_input(user_input)

        if found_song_id:
            print(f"\n[成功] 已解析输入，对应歌曲 ID: {found_song_id}")
            detail = get_track_detail(found_song_id)
            if detail:
                print("\n>>> 歌曲详情:"); pprint({"歌名": detail.get('name'), "歌手": [a['name'] for a in detail.get('ar', [])]})
            
            if inquirer.confirm(f"是否下载这首歌曲: '{detail.get('name')}'?", default=True):
                download_path = download_song_by_id(found_song_id)
                if download_path:
                    logger.info(f"文件已成功保存到: {download_path.resolve()}")
                else:
                    logger.error("下载失败，请查看上面的错误信息。")
        else:
            print(f"\n[失败] 未能通过 '{user_input}' 找到可播放的歌曲。")
    print("\n测试程序已退出。")