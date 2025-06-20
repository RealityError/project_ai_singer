import time
import librosa
from tqdm.auto import tqdm
import sys
import os
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn

# --- 路径修正与日志记录器设置 ---
# 这个代码块确保了无论脚本如何运行，都能正确导入其依赖项。

# 将项目根目录添加到搜索路径 (e.g., /path/to/project)
# 使得 'from utils import ...' 等项目级导入能够被正确解析
# 注意：这里的路径计算假设此文件位于 project_root/modules/separator/ 类似结构下
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except:
    PROJECT_ROOT = "" # 如果计算失败，则置空

# 确保本地的 'utils' 模块可以被正确导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
# 导入自定义的 loguru 日志记录器
try:
    from utils.loguru_logger import logger
    logger.debug("成功导入自定义的 loguru_logger。")
except ImportError:
    # 如果找不到自定义的logger，则使用标准的logging作为备用
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.warning("未找到自定义的 loguru_logger，已切换到 Python 标准 logging。")

try:
    from modules.msst.utils_msst import demix, get_model_from_config, prefer_target_instrument
except ImportError as e:
    logger.error(f"导入本地 utils 时出错: {e}")
    logger.error("请确保 'utils.py' 文件与此脚本位于同一目录下。")
    raise

import warnings
warnings.filterwarnings("ignore")


class AudioSeparator:
    """
    一个用于将音频文件分离成不同音轨（如人声、鼓、贝斯等）的类。
    它封装了模型加载和推理的逻辑。
    使用示例:
    # 1. 初始化分离器 (这是一个耗时操作)
    separator = AudioSeparator(
        model_type='mdx23c',
        config_path='path/to/your/model/config.yaml',
        checkpoint_path='path/to/your/model.pth'
    )

    # 2. 调用方法处理单个文件
    output_files = separator.separate_file(
        file_path='path/to/input.wav',
        output_dir='path/to/output_folder'
    )
    logger.info(f"分离完成，文件保存在: {output_files}")
    """
    
    def __init__(self, model_type: str, config_path: str, checkpoint_path: str, device: str = None, force_cpu: bool = False):
        """
        初始化分离器并将模型加载到内存中。

        :param model_type: 要使用的模型类型 (例如, 'mdx23c', 'htdemucs')。
        :param config_path: 模型的配置文件路径。
        :param checkpoint_path: 模型的检查点文件路径 (.pth)。
        :param device: 用于运行推理的设备 ('cuda', 'cpu', 'mps')。如果为 None, 将自动检测。
        :param force_cpu: 如果为 True, 即使 CUDA 可用也强制使用 CPU。
        """
        logger.info("正在初始化音频分离器...")
        load_start_time = time.time()

        # --- 设备选择 ---
        if device is None:
            if force_cpu:
                self.device = "cpu"
            elif torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        logger.info(f"使用设备: {self.device}")

        # --- 模型加载 ---
        self.model, self.config = get_model_from_config(model_type, config_path)
        
        if checkpoint_path:
            logger.info(f'正在从以下路径加载检查点: {checkpoint_path}')
            # weights_only 参数的逻辑取决于模型类型
            weights_only = False if model_type in ['htdemucs', 'apollo'] else True
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=weights_only)
            
            # 处理来自不同预训练模型的不同 state_dict 格式
            if 'state' in state_dict:
                state_dict = state_dict['state']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        torch.backends.cudnn.benchmark = True
        
        self.instruments = prefer_target_instrument(self.config)
        logger.info(f"将要分离的音轨: {self.instruments}")
        logger.success(f"模型加载耗时: {time.time() - load_start_time:.2f} 秒")


    def separate_file(self, file_path: str, output_dir: str, 
                      use_tta: bool = True, 
                      extract_instrumental: bool = True, 
                      save_as_flac: bool = False,
                      pcm_type: str = 'PCM_24',
                      show_progress: bool = True) -> dict:
        """
        将单个音频文件分离成其组成音轨。

        :param file_path: 输入音频文件的路径。
        :param output_dir: 保存输出音轨文件的目录。
        :param use_tta: 如果为 True, 使用测试时增强以获得更好的质量但速度较慢。
        :param extract_instrumental: 如果为 True, 通过减去主音轨来创建伴奏音轨。
        :param save_as_flac: 如果为 True, 将输出保存为 FLAC 文件。否则, 保存为 WAV。
        :param pcm_type: FLAC 文件的 PCM 类型 ('PCM_16' 或 'PCM_24')。
        :param show_progress: 如果为 True, 显示详细的分离进度条。
        :return: 一个将乐器名称映射到其输出文件路径的字典。
        """
        logger.info(f"正在处理音轨: {file_path}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            mix, sr = librosa.load(file_path, sr=44100, mono=False)
        except Exception as e:
            logger.error(f'无法读取音轨: {file_path}')
            logger.error(f'错误详情: {e}')
            return {}

        # 如果需要，将单声道转换为立体声
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=0)

        mix_orig = mix.copy()
        if 'normalize' in self.config.inference and self.config.inference['normalize']:
            mono = mix.mean(0)
            mean, std = mono.mean(), mono.std()
            mix = (mix - mean) / std
        else:
            mean, std = 0, 1

        # --- 测试时增强 (TTA) ---
        if use_tta:
            # 原始版本、通道交换版本和极性反转版本
            track_proc_list = [mix.copy(), mix[::-1].copy(), -1. * mix.copy()]
        else:
            track_proc_list = [mix.copy()]

        full_result = []
        iterable = tqdm(track_proc_list, desc="分离中 (TTA)") if show_progress and use_tta else track_proc_list
        for track_mix in iterable:
            waveforms = demix(self.config, self.model, track_mix, self.device, pbar=show_progress)
            full_result.append(waveforms)
        
        # 对 TTA 的结果进行平均
        waveforms = full_result[0]
        for i in range(1, len(full_result)):
            d = full_result[i]
            for el in d:
                if i == 1: # 通道交换
                    waveforms[el] += d[el][::-1].copy()
                elif i == 2: # 极性反转
                    waveforms[el] += -1.0 * d[el]
                else:
                    waveforms[el] += d[el]
        
        for el in waveforms:
            waveforms[el] /= len(full_result)

        # --- 伴奏提取 ---
        current_instruments = list(self.instruments)
        if extract_instrumental:
            instr_to_invert = 'vocals' if 'vocals' in current_instruments else current_instruments[0]
            if 'instrumental' not in current_instruments:
                current_instruments.append('instrumental')
            waveforms['instrumental'] = mix_orig - waveforms[instr_to_invert]

        # --- 保存文件 ---
        output_paths = {}
        file_name, _ = os.path.splitext(os.path.basename(file_path))
        for instr in current_instruments:
            estimates = waveforms[instr].T
            estimates = estimates * std + mean # 反归一化

            if save_as_flac:
                output_file = os.path.join(output_dir, f"{file_name}_{instr}.flac")
                subtype = 'PCM_16' if pcm_type == 'PCM_16' else 'PCM_24'
                sf.write(output_file, estimates, sr, subtype=subtype)
            else:
                output_file = os.path.join(output_dir, f"{file_name}_{instr}.wav")
                sf.write(output_file, estimates, sr, subtype='FLOAT')
            output_paths[instr] = output_file
        
        logger.success(f"成功分离文件: {os.path.basename(file_path)}")
        return output_paths


if __name__ == '__main__':
    """
    模块的自我测试例程。
    可以直接运行此脚本来测试 AudioSeparator 类的基本功能。
    请在使用前，务必修改下面的4个路径变量。
    """
    def create_dummy_audio(path, sr=44100, duration=5):
        """如果测试音频不存在，则创建一个虚拟WAV文件"""
        if os.path.exists(path):
            return
        try:
            logger.info(f"未找到测试音频，正在创建虚拟文件: {path}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            t = np.linspace(0., duration, int(sr * duration))
            amplitude = np.iinfo(np.int16).max * 0.3
            data = amplitude * (np.sin(2. * np.pi * 220.0 * t) + np.sin(2. * np.pi * 440.0 * t))
            sf.write(path, data.astype(np.int16), sr)
            logger.info("虚拟音频创建成功。")
        except Exception as e:
            logger.error(f"创建虚拟音频失败: {e}")

    def run_test():
        # --- 1. 请在此处修改为您的实际路径 ---
        # !! 修改这里 !!
        MODEL_TYPE = 'mel_band_roformer'
        CONFIG_PATH = 'static\model\msst\config_karaoke_becruily.yaml'
        CHECKPOINT_PATH = 'static\model\msst\mel_band_roformer_karaoke_becruily.ckpt'
        INPUT_AUDIO = 'outputs\separated_output\明天你好_vocals.wav'
        OUTPUT_DIR = 'outputs/separated_output/results'
        
        logger.info("-" * 40)
        logger.info("开始执行音频分离模块测试...")

        # --- 2. 检查文件是否存在 ---

        if not all(os.path.exists(p) for p in [CONFIG_PATH, CHECKPOINT_PATH, INPUT_AUDIO]):
            logger.error("测试无法进行: 请确保模型配置、检查点和输入音频文件都存在。")
            if not os.path.exists(CONFIG_PATH): logger.error(f"找不到配置文件: {CONFIG_PATH}")
            if not os.path.exists(CHECKPOINT_PATH): logger.error(f"找不到检查点文件: {CHECKPOINT_PATH}")
            if not os.path.exists(INPUT_AUDIO): logger.error(f"找不到输入音频: {INPUT_AUDIO}")
            return
            
        try:
            # --- 3. 初始化分离器 ---
            separator = AudioSeparator(
                model_type=MODEL_TYPE,
                config_path=CONFIG_PATH,
                device="cuda",
                checkpoint_path=CHECKPOINT_PATH
            )

            # --- 4. 执行分离 ---
            output_files = separator.separate_file(
                file_path=INPUT_AUDIO,
                output_dir=OUTPUT_DIR
            )

            # --- 5. 打印结果 ---
            if output_files:
                logger.success("测试成功完成！")
                logger.info("输出文件:")
                for instrument, path in output_files.items():
                    logger.info(f"  - {instrument}: {path}")
            else:
                logger.warning("测试执行完毕，但没有生成任何文件。请检查日志中的错误信息。")
            
            logger.info("-" * 40)

        except Exception as e:
            logger.error(f"测试过程中发生未捕获的错误: {e}", exc_info=True)

    # 运行测试函数
    run_test()