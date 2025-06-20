# svc_converter.py
import os,sys
import torch
import soundfile
import numpy as np

# --- Path Correction ---
# 这个代码块确保了无论脚本如何运行，
# 它都能正确地导入 so-vits-svc 库内部的模块和项目顶层的 'utils' 模块。

# 将项目根目录添加到搜索路径 (e.g., /path/to/project)
# 这使得 'from utils import ...' 能够被正确解析
# 修正了此处的路径计算，确保它指向正确的项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 将 so-vits-svc 库的根目录添加到搜索路径 (e.g., /path/to/project/modules/svc)
# 使用 insert(0,...) 确保该路径被优先搜索, 解决内部模块的相对导入问题
SVC_ROOT = os.path.dirname(os.path.abspath(__file__))
if SVC_ROOT not in sys.path:
    sys.path.insert(0, SVC_ROOT)
    
    
# --- End of Path Correction ---


# 假设您的项目结构中有 `utils/loguru_logger.py`
# 如果路径不同，请相应修改
try:
    from utils.loguru_logger import logger
    logger.debug("成功导入自定义的 loguru_logger。")
    logger.debug(f"项目根目录 (PROJECT_ROOT) 设置为: {PROJECT_ROOT}")
    logger.debug(f"SVC 库根目录 (SVC_ROOT) 设置为: {SVC_ROOT}")
except ImportError:
    # 如果找不到自定义的logger，则使用标准的logging作为备用
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.warning("未找到自定义的 loguru_logger，已切换到 Python 标准 logging。")
# --- PyTorch 2.x/Fairseq Compatibility Patch ---
# 新版 PyTorch (2.x+) 为了安全，默认不允许加载包含未知类对象的 checkpoint。
# fairseq 的 Hubert 模型 checkpoint 包含一个 'Dictionary' 类对象，默认会被阻止。
# 我们需要在此处将其显式地添加到 PyTorch 的安全列表。
# 这是 PyTorch 官方推荐的、更直接和稳妥的解决方案。
try:
    import torch.serialization
    import torchaudio
    torchaudio.set_audio_backend("soundfile")
    from fairseq.data import Dictionary
    logger.info("正在应用 PyTorch/Fairseq 兼容性补丁...")
    
    # 将 'fairseq.data.dictionary.Dictionary' 添加到全局安全列表
    torch.serialization.add_safe_globals([Dictionary])
    
    logger.info("兼容性补丁应用成功。")
except ImportError:
    logger.warning("未找到 fairseq 库，跳过兼容性补丁。如果遇到模型加载错误，请确保已安装 fairseq。")
except Exception as e:
    logger.error(f"应用 PyTorch/Fairseq 兼容性补丁时出错: {e}")
# --- End of Compatibility Patch ---

# 从 so-vits-svc 4.1 稳定版项目中导入必要的组件
try:
    from inference import infer_tool
    from inference.infer_tool import Svc
except ImportError as e:
    logger.error(f"无法导入 so-vits-svc 依赖: {e}", exc_info=True)
    logger.error("请确保您已正确安装 so-vits-svc 4.1-Stable 分支及其依赖项。")
    raise

class Sovits41Converter:
    """
    so-vits-svc 4.1 稳定版推理模块封装。
    
    该类封装了模型的加载和推理过程，旨在提供一个干净、易于调用的接口。
    
    使用方法:
    1. 初始化类，加载模型:
       converter = Sovits41Converter(
           model_path="path/to/G_xxxxx.pth",
           config_path="path/to/config.json"
       )
    
    2. 调用 infer 方法进行推理:
       audio_data, sample_rate = converter.infer(
           input_audio_path="path/to/input.wav",
           speaker="speaker_name",
           transpose=0
       )
    
    3. 保存音频:
       soundfile.write("output.wav", audio_data, sample_rate)
    """

    def __init__(self, 
                 model_path: str, 
                 config_path: str, 
                 device: str = "cpu",
                 # --- 可选的模型增强组件 ---
                 enhance: bool = False,
                 cluster_model_path: str = "",
                 feature_retrieval: bool = False,
                 # --- 浅扩散相关 ---
                 shallow_diffusion: bool = False, 
                 diffusion_model_path: str = "", 
                 diffusion_config_path: str = "",
                 # --- 其他高级选项 ---
                 only_diffusion: bool = False,
                 use_spk_mix: bool = False):
        """
        初始化转换器并加载核心模型。
        这是一个耗时操作，建议在程序启动时执行一次。
        
        :param model_path: Sovits 主模型路径 (.pth)
        :param config_path: 主模型配置文件路径 (.json)
        :param device: 推理设备 ('cpu', 'cuda', None for auto-detect)。若为 None，则自动检测。
        :param enhance: 是否使用 NSF_HIFIGAN 增强器。
        :param cluster_model_path: 聚类模型或特征检索索引的路径。
        :param feature_retrieval: 是否启用特征检索模式。
        :param shallow_diffusion: 是否启用浅层扩散。
        :param diffusion_model_path: 扩散模型路径 (.pt)。
        :param diffusion_config_path: 扩散模型配置文件路径 (.yaml)。
        :param only_diffusion: 是否只使用纯扩散模式。
        :param use_spk_mix: 是否启用说话人融合模式。
        """
        logger.info("正在初始化 Sovits.4.1 推理转换器...")
        
        # 自动检测设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"未指定设备，自动选择为: {device}")
        
        self.device = device

        # 实例化 Svc 核心类时，使用与您提供的 `infer_tool.py` 完全匹配的关键字参数。
        self.svc_model = Svc(
            net_g_path=model_path,
            config_path=config_path,
            device=self.device,
            cluster_model_path=cluster_model_path,
            nsf_hifigan_enhance=enhance,
            diffusion_model_path=diffusion_model_path,
            diffusion_config_path=diffusion_config_path,
            shallow_diffusion=shallow_diffusion,
            only_diffusion=only_diffusion,
            spk_mix_enable=use_spk_mix,
            feature_retrieval=feature_retrieval
        )
        
        self.target_sample = self.svc_model.target_sample
        logger.success(f"模型加载完成。目标采样率: {self.target_sample}Hz, 运行设备: {self.device}")

    def get_speakers(self) -> list:
        """获取模型支持的所有说话人名称"""
        # *** 最终修正点 ***
        # 根据用户提供的 infer_tool.py，正确的属性名是 hps_ms
        return list(self.svc_model.hps_ms.spk.keys())

    def infer(self, 
              input_audio_path: str, 
              speaker: str, 
              transpose: int = 0,
              # --- 核心推理参数 ---
              auto_predict_f0: bool = False,
              cluster_infer_ratio: float = 0.0,
              f0_predictor: str = "rmvpe", # rmvpe 是目前效果和速度综合最佳的选择
              loudness_envelope_adjustment: float = 1.0,
              # --- 音频切片参数 ---
              slice_db: int = -40,
              clip_seconds: float = 30,
              pad_seconds: float = 0.5,
              # --- 浅扩散参数 ---
              k_step: int = 100,
              second_encoding: bool = False,
              # --- 其他玄学/微调参数 ---
              noice_scale: float = 0.4,
              linear_gradient: float = 0.0,
              linear_gradient_retain: float = 0.75,
              enhancer_adaptive_key: int = 0,
              f0_filter_threshold: float = 0.05
              ) -> tuple[any, int]:
        """
        执行语音转换。

        :param input_audio_path: 待转换的源音频WAV文件路径。
        :param speaker: 目标说话人名称 (str) 或说话人融合字典 (dict)。
        :param transpose: 音高调整（半音数），正负整数。
        :param auto_predict_f0: 是否自动预测F0，用于说话转换，**转换歌声时请务必保持为False**。
        :param cluster_infer_ratio: 特征检索或聚类占比 (范围0-1)，越高越接近目标音色，越低越保留原始情感。
        :param f0_predictor: F0预测器 ('crepe', 'pm', 'dio', 'harvest', 'rmvpe', 'fcpe')。
        :param loudness_envelope_adjustment: 响度包络调整比例 (范围0-1)。
        :param slice_db: 自动切片分贝阈值，值越小越灵敏。
        :param clip_seconds: 强制切片秒数，0为自动切片。
        :param pad_seconds: 在音频首尾填充的静音秒数，可防止爆音。
        :param k_step: 浅扩散步数，越大越接近扩散模型的结果。
        :param second_encoding: 是否二次编码（浅扩散的玄学选项）。
        :param noice_scale: 噪音比例，影响音质和咬字。
        :param linear_gradient: 交叉淡入长度(秒)，用于强制切片。
        :param linear_gradient_retain: 自动切片交叉长度保留比例。
        :param enhancer_adaptive_key: 使增强器适应更高的音域。
        :param f0_filter_threshold: F0过滤阈值，仅crepe有效。
        
        :return: 元组 (音频数据 (numpy.ndarray), 采样率 (int))
        """
        logger.debug(f"收到推理任务: 音频='{os.path.basename(input_audio_path)}', 角色='{speaker}', 变调={transpose}")
        
        # 准备推理所需的参数字典
        kwarg = {
            "raw_audio_path": input_audio_path,
            "spk": speaker,
            "tran": transpose,
            "slice_db": slice_db,
            "cluster_infer_ratio": cluster_infer_ratio,
            "auto_predict_f0": auto_predict_f0,
            "noice_scale": noice_scale,
            "pad_seconds": pad_seconds,
            "clip_seconds": clip_seconds,
            "lg_num": linear_gradient,
            "lgr_num": linear_gradient_retain,
            "f0_predictor": f0_predictor,
            "enhancer_adaptive_key": enhancer_adaptive_key,
            "cr_threshold": f0_filter_threshold,
            "k_step": k_step,
            "use_spk_mix": isinstance(speaker, dict), # 如果speaker是字典，则自动启用混音模式
            "second_encoding": second_encoding,
            "loudness_envelope_adjustment": loudness_envelope_adjustment
        }

        # 确保输入音频是正确的WAV格式 (这一步很重要)
        try:
            infer_tool.format_wav(input_audio_path)
        except Exception as e:
            logger.error(f"格式化WAV文件失败: {e}")
            raise IOError(f"无法处理音频文件: {input_audio_path}") from e

        # 执行核心的切片推理
        logger.info("开始执行切片推理...")
        audio_data = self.svc_model.slice_inference(**kwarg)
        
        # 清理推理过程中产生的临时文件 (好习惯)
        self.svc_model.clear_empty()
        
        logger.success("推理成功完成。")
        return audio_data, self.target_sample
    
    
if __name__ == '__main__':
    """
    模块的自我测试例程。
    可以直接运行此脚本来测试 Sovits41Converter 类的基本功能。
    请在使用前，务必修改下面的模型路径、配置路径和输入音频路径。
    """

    def run_test():
        # --- 1. 修改为您的实际路径 ---
        # 模型文件夹通常包含 G_xxxxx.pth, config.json 等文件
        MODEL_DIR = "static\\model\\sovits4.1\\paimeng" # !! 修改这里 !!
        
        model_path = os.path.join(MODEL_DIR, "G_41600.pth") # !! 修改 G_xxxxx.pth 为您的模型文件名 !!
        config_path = os.path.join(MODEL_DIR, "config.json")
        
        # 测试用的输入和输出文件
        TEST_INPUT_DIR = "outputs/separated_output/results"
        TEST_OUTPUT_DIR = "outputs/test_outputs"
        os.makedirs(TEST_INPUT_DIR, exist_ok=True)
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        
        input_audio_path = os.path.join(TEST_INPUT_DIR, "明天你好_vocals_Vocals.wav")
        output_audio_path = os.path.join(TEST_OUTPUT_DIR, "明天你好_output.wav")

        # --- 2. 检查文件是否存在 ---
        if not all(os.path.exists(p) for p in [model_path, config_path, input_audio_path]):
            logger.error("测试无法进行: 请确保模型、配置和输入音频文件都存在。")
            if not os.path.exists(model_path): logger.error(f"找不到模型: {model_path}")
            if not os.path.exists(config_path): logger.error(f"找不到配置: {config_path}")
            if not os.path.exists(input_audio_path): logger.error(f"找不到输入音频: {input_audio_path}")
            return
            
        
        logger.info("-" * 40)
        logger.info("开始执行模块测试...")
        
        try:
            # --- 3. 初始化转换器 ---
            # 这是重量级操作，只需执行一次
            converter = Sovits41Converter(
                model_path=model_path,
                config_path=config_path,
                device="cuda" 
                # 如果有浅扩散模型，在这里填入路径
                # shallow_diffusion=True,
                # diffusion_model_path="path/to/diffusion_model.pt",
                # diffusion_config_path="path/to/diffusion_config.yaml"
            )

            # --- 4. 执行推理 ---
            logger.info(f"准备转换音频: '{input_audio_path}'")
            
            # 从加载的模型配置中安全地获取说话人列表
            speakers = converter.get_speakers()
            if not speakers:
                logger.error("模型配置中没有找到任何说话人 (speaker)！")
                return
            
            target_speaker = speakers[0] # 使用列表中的第一个说话人进行测试
            logger.info(f"模型支持的说话人: {speakers}")
            logger.info(f"将使用目标说话人: {target_speaker}")

            audio_data, sample_rate = converter.infer(
                input_audio_path=input_audio_path,
                speaker=target_speaker,
                transpose=0,
                f0_predictor="rmvpe",
            )

            # --- 5. 保存输出 ---
            soundfile.write(output_audio_path, audio_data, sample_rate)
            logger.success(f"测试成功！转换后的音频已保存至: '{output_audio_path}'")
            logger.info("-" * 40)

        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}", exc_info=True)

    # 运行测试函数
    run_test()
