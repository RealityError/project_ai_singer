# app.py
import os
import threading
import time
import uuid
import subprocess
from flask import Flask, render_template, request, jsonify, Response, url_for, send_from_directory
import json
import torch
from pathlib import Path
import queue

# --- 配置日志 ---
try:
    from utils.loguru_logger import logger
    logger.info("成功导入自定义的 loguru_logger。")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.warning("未找到自定义的 loguru_logger，已切换到 Python 标准 logging。")


# --- 导入您的模块 ---
try:
    from modules import pyncm_downloader
except ImportError: logger.error("无法导入 pyncm_downloader.py 模块。"); pyncm_downloader = None
try:
    from modules.vocal_separater import AudioSeparator
except ImportError: logger.error("无法导入 audio_separator.py 模块。"); AudioSeparator = None
try:
    from modules.svc.svc_converter import Sovits41Converter
except ImportError: logger.error("无法导入 svc_converter.py 模块。"); Sovits41Converter = None
try:
    from pydub import AudioSegment
except ImportError: logger.error("pydub 库未安装。请运行: pip install pydub"); AudioSegment = None
try:
    import librosa
    import soundfile as sf
except ImportError: logger.error("librosa 或 soundfile 库未安装。"); librosa, sf = None, None

# --- 初始化 Flask 应用 ---
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'outputs/downloaded_music'
app.config['OUTPUT_FOLDER'] = 'outputs/separated_music_outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


# --- 全局变量与任务队列 ---
tasks = {}
model_cache = None
model_cache_lock = threading.Lock()
task_queue = queue.Queue()
GPU_WORKER_STATE = 'idle'
worker_state_lock = threading.Lock()

# --- 模型扫描与加载逻辑 ---

def _scan_separation_models(msst_base_dir_abs: str) -> dict:
    logger.info(f"开始通过清单文件加载分离模型: {msst_base_dir_abs}")
    categorized_models = {"vocal_models": [], "kara_models": [], "reverb_models": [], "other_models": []}
    manifest_path = os.path.join(msst_base_dir_abs, "model_config_zh.json")
    try:
        if not os.path.isfile(manifest_path):
            logger.error(f"分离模型清单文件未找到: {manifest_path}"); return categorized_models
        with open(manifest_path, 'r', encoding='utf-8') as f: manifest = json.load(f)
        
        config_map = manifest.get("config_paths", {})
        type_map = manifest.get("model_types", {})
        model_to_category = {model_name: category for category, models in manifest.items() if category.endswith("_models") for model_name in models.keys()}
        
        for ckpt_filename, config_path_list in config_map.items():
            ckpt_full_path_abs = os.path.join(msst_base_dir_abs, ckpt_filename)
            if not os.path.isfile(ckpt_full_path_abs):
                logger.warning(f"清单模型文件不存在，跳过: {ckpt_full_path_abs}"); continue
            
            category_key = model_to_category.get(ckpt_filename)
            if not category_key: continue
            base_description = manifest.get(category_key, {}).get(ckpt_filename, ckpt_filename)

            for config_relative_path in config_path_list:
                config_full_path_abs = os.path.join(msst_base_dir_abs, config_relative_path)
                if not os.path.isfile(config_full_path_abs):
                    logger.warning(f"清单配置文件不存在，跳过: {config_full_path_abs}"); continue

                model_type = type_map.get(ckpt_filename, "unknown")
                config_basename = os.path.splitext(os.path.basename(config_relative_path))[0]
                ckpt_basename = os.path.splitext(ckpt_filename)[0]
                unique_id = f"{ckpt_basename}_{config_basename}"
                display_name = f"{ckpt_basename} - {base_description}"
                if "-fast" in config_basename.lower(): display_name += " (快速版)"

                model_entry = {"id": unique_id, "name": display_name, "type": model_type, "config_path": config_full_path_abs.replace("\\", "/"), "checkpoint_path": ckpt_full_path_abs.replace("\\", "/")}
                if category_key in categorized_models: categorized_models[category_key].append(model_entry)
        
        for category in categorized_models: categorized_models[category].insert(0, {"id": "None", "name": "不使用此模块"})
        logger.info("成功加载分离模型。")
    except Exception as e:
        logger.error(f"加载或解析分离模型清单时发生未知错误: {e}", exc_info=True)
    return categorized_models

def _scan_detailed_conversion_models(svc_base_dir_abs: str) -> list:
    logger.info(f"开始扫描SVC转换模型目录: {svc_base_dir_abs}")
    conversion_models_list = [{"id": "None", "name": "不进行声线转换"}]
    try:
        if not os.path.isdir(svc_base_dir_abs):
            logger.warning(f"SVC模型目录未找到: {svc_base_dir_abs}")
            return conversion_models_list
            
        for model_id in sorted(os.listdir(svc_base_dir_abs)):
            model_dir_abs = os.path.join(svc_base_dir_abs, model_id)
            if os.path.isdir(model_dir_abs):
                pth_path, json_path = None, None
                for item in os.listdir(model_dir_abs):
                    item_abs = os.path.join(model_dir_abs, item)
                    if item.lower().endswith(('.pth', '.safetensors')): pth_path = item_abs
                    if item.lower().endswith('.json'): json_path = item_abs
                
                if pth_path and json_path:
                    enhancements = {"cluster_model_path": None, "feature_model_path": None, "diffusion_model_path": None, "diffusion_config_path": None}
                    model_name = model_id.capitalize()
                    conversion_models_list.append({
                        "id": model_id, "name": model_name,
                        "checkpoint_path": pth_path.replace("\\", "/"), "config_path": json_path.replace("\\", "/"),
                        "base_dir": model_dir_abs.replace("\\", "/"), "enhancements": enhancements
                    })
    except Exception as e:
        logger.error(f"扫描SVC模型时发生未知错误: {e}", exc_info=True)
    logger.info(f"扫描到 {len(conversion_models_list)-1} 个有效的SVC模型。")
    return conversion_models_list


def get_all_application_models():
    global model_cache
    with model_cache_lock:
        if model_cache: return model_cache
        logger.info("开始扫描所有应用模型...")
        MSST_MODELS_DIR_ABS = os.path.abspath(os.path.join("static", "model", "msst"))
        SOVITS_MODELS_DIR_ABS = os.path.abspath(os.path.join("static", "model", "sovits4.1"))
        separation_models_by_cat = _scan_separation_models(MSST_MODELS_DIR_ABS)
        conversion_models = _scan_detailed_conversion_models(SOVITS_MODELS_DIR_ABS)
        model_cache = {**separation_models_by_cat, "conversion_models": conversion_models}
        return model_cache

# --- 后端核心处理函数 ---

def download_audio_from_url(task_id, url, platform):
    logger.info(f"[{task_id}] 正在从 {platform} 下载: {url}")
    send_progress(task_id, f"正在从 {platform} 下载音频...", 5)
    if platform == 'netease':
        if not pyncm_downloader: raise ModuleNotFoundError("网易云音乐下载模块未导入。")
        song_id = pyncm_downloader.resolve_song_input(url)
        if not song_id: raise ValueError(f"无法从 '{url}' 解析出歌曲ID。")
        downloaded_path_obj = pyncm_downloader.download_song_by_id(song_id, save_dir=app.config['UPLOAD_FOLDER'])
        if not downloaded_path_obj: raise ConnectionError("下载失败，可能是VIP或付费歌曲。")
        return str(downloaded_path_obj.resolve())
    elif platform in ['bilibili', 'youtube']:
        output_template = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}.%(ext)s")
        command = ['yt-dlp', '-x', '--audio-format', 'wav', '-o', output_template, url]
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        return os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}.wav")
    raise NotImplementedError(f"不支持的平台: {platform}")

def separate_audio(task_id, input_path, output_dir, model_params):
    logger.info(f"[{task_id}] 音频分离中, 模型: {model_params.get('name')}")
    if not AudioSeparator: raise ModuleNotFoundError("音频分离模块未导入。")
    separator = AudioSeparator(model_type=model_params.get('type'), config_path=model_params.get('config_path'), checkpoint_path=model_params.get('checkpoint_path'))
    output_files = separator.separate_file(file_path=input_path, output_dir=output_dir)
    if not output_files: raise RuntimeError("分离失败，未能生成任何文件。")
    main_stem_key = 'vocals'
    if main_stem_key not in output_files:
        main_stem_key = list(output_files.keys())[0]
        logger.warning(f"在输出中未找到 'vocals'，将使用 '{main_stem_key}' 作为下一阶段的输入。")
    main_stem_path = output_files.pop(main_stem_key)
    return {"main_stem": main_stem_path, "other_stems": list(output_files.values())}

def convert_voice(task_id, input_path, output_dir, params):
    logger.info(f"[{task_id}] 开始声线转换，参数: {params}")
    send_progress(task_id, "正在进行声线转换...", 80)
    if not Sovits41Converter: raise ModuleNotFoundError("声线转换模块 (Sovits41Converter) 未成功导入。")
    try:
        converter = Sovits41Converter(model_path=params.get('checkpoint_path'), config_path=params.get('config_path'), device="cuda" if torch.cuda.is_available() else "cpu", enhance=params.get('enhance', False), cluster_model_path=params.get('cluster_model_path') or params.get('feature_model_path', ""), feature_retrieval=bool(params.get('feature_model_path')), shallow_diffusion=params.get('shallow_diffusion', False), diffusion_model_path=params.get('diffusion_model_path', ""), diffusion_config_path=params.get('diffusion_config_path', ""))
        speakers = converter.get_speakers()
        if not speakers: raise ValueError("模型配置中没有找到可用的说话人。")
        target_speaker = speakers[0]
        logger.info(f"[{task_id}] 自动选择目标说话人: {target_speaker}")
        audio_data, sample_rate = converter.infer(input_audio_path=input_path, speaker=target_speaker, transpose=params.get('transpose', 0), auto_predict_f0=params.get('auto_predict_f0', False), cluster_infer_ratio=params.get('cluster_infer_ratio', 0.0), f0_predictor=params.get('f0_predictor', 'rmvpe'), k_step=params.get('k_step', 100))
        output_filename = f"converted_vocals_{task_id}.wav"
        output_path = os.path.join(output_dir, output_filename)
        sf.write(output_path, audio_data, sample_rate)
        logger.info(f"[{task_id}] 声线转换成功，文件保存在: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"[{task_id}] 声线转换过程中出错: {e}", exc_info=True)
        raise e

def pitch_shift_instrumentals(task_id, instrumental_paths, n_steps, output_dir):
    if n_steps == 0: return instrumental_paths
    if not librosa or not sf: logger.error("Librosa 或 soundfile 未安装，无法进行变调。"); return instrumental_paths
    logger.info(f"[{task_id}] 开始对 {len(instrumental_paths)} 个伴奏音轨进行变调，步数: {n_steps}...")
    send_progress(task_id, "正在调整伴奏音高...", 85)
    shifted_paths = []
    for i, path in enumerate(instrumental_paths):
        try:
            y, sr = librosa.load(path, sr=None)
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            shifted_filename = f"{Path(path).stem}_shifted_{n_steps}key.wav"
            shifted_path = os.path.join(output_dir, shifted_filename)
            sf.write(shifted_path, y_shifted, sr)
            shifted_paths.append(shifted_path)
            logger.debug(f"[{task_id}] 伴奏变调完成: {shifted_path}")
        except Exception as e:
            logger.error(f"[{task_id}] 对伴奏文件 {path} 进行变调时失败: {e}，将使用原文件代替。", exc_info=True)
            shifted_paths.append(path)
    return shifted_paths

def combine_audio(task_id, vocal_path, instrumental_paths, output_dir):
    logger.info(f"[{task_id}] 开始最终混音，包含 {len(instrumental_paths)} 个伴奏部分...")
    send_progress(task_id, "正在合成最终音频...", 90)
    if not AudioSegment: raise ModuleNotFoundError("音频混合所需的 pydub 库未安装。")
    if not instrumental_paths:
        logger.warning(f"[{task_id}] 没有找到伴奏文件，最终成品将只有转换后的人声。")
        return vocal_path
    try:
        converted_vocal = AudioSegment.from_file(vocal_path)
        
        
        logger.debug(f"[{task_id}] 加载基础伴奏: {instrumental_paths[0]}")
        combined_instrumental = AudioSegment.from_file(instrumental_paths[0])

        for ins_path in instrumental_paths[1:]:
            logger.debug(f"[{task_id}] 叠加伴奏部分: {ins_path}")
            instrumental_part = AudioSegment.from_file(ins_path)
            combined_instrumental = combined_instrumental.overlay(instrumental_part)
        vocal_dbfs = converted_vocal.dBFS
        if combined_instrumental.dBFS != -float('inf'):
            gain_diff = vocal_dbfs - combined_instrumental.dBFS
            logger.info(f"[{task_id}] 人声音量: {vocal_dbfs:.2f} dBFS, 合并后伴奏音量: {combined_instrumental.dBFS:.2f} dBFS. 增益调整: {gain_diff:.2f} dB")
            combined_instrumental = combined_instrumental.apply_gain(gain_diff)
        
        final_mix = combined_instrumental.overlay(converted_vocal)
        output_filename = f"final_mix_{task_id}.wav"
        output_path = os.path.join(output_dir, output_filename)
        final_mix.export(output_path, format="wav")
        logger.info(f"[{task_id}] 混音成功，文件保存在: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"[{task_id}] 音频混合过程中出错: {e}", exc_info=True)
        raise e

def processing_task(task_id, input_data, params):
    with app.test_request_context():
        temp_file_path = None
        try:
            task_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
            os.makedirs(task_dir, exist_ok=True)
            if input_data.get('type') == 'url':
                temp_file_path = download_audio_from_url(task_id, input_data['value'], input_data['platform'])
            else:
                temp_file_path = input_data['value']
            if not temp_file_path or not os.path.exists(temp_file_path):
                raise FileNotFoundError("未能获取到有效的输入音频文件。")
            current_processing_file = temp_file_path
            all_other_stems = []
            separation_pipeline = params.get('separation_pipeline', [])
            total_stages = len([p for p in separation_pipeline if p.get('id') != 'None'])
            stages_done = 0
            for model_params in separation_pipeline:
                if model_params.get('id') == 'None': continue
                stages_done += 1
                stage_progress = 20 + int(stages_done / total_stages * 50) if total_stages > 0 else 70
                send_progress(task_id, f"第 {stages_done}/{total_stages} 步分离: {model_params.get('name')}...", stage_progress)
                separation_result = separate_audio(task_id, current_processing_file, task_dir, model_params)
                current_processing_file = separation_result['main_stem']
                all_other_stems.extend(separation_result['other_stems'])
                logger.info(f"[{task_id}] 第 {stages_done} 步分离完成。")
            
            final_vocals_path = current_processing_file
            conversion_params = params.get('conversion_model', {})
            shifted_instrumentals = all_other_stems
            
            if conversion_params.get('id') != 'None':
                converted_vocals_path = convert_voice(task_id, final_vocals_path, task_dir, conversion_params)
                transpose_steps = conversion_params.get('transpose', 0)
                if transpose_steps != 0:
                    shifted_instrumentals = pitch_shift_instrumentals(task_id, all_other_stems, transpose_steps, task_dir)
            else:
                logger.info(f"[{task_id}] 跳过声线转换步骤。")
                send_progress(task_id, "跳过声线转换", 85)
                converted_vocals_path = final_vocals_path

            final_mix_path = combine_audio(task_id, converted_vocals_path, shifted_instrumentals, task_dir)
            
            final_result = {
                'final_mix': url_for('serve_result_file', task_id=task_id, filename=os.path.basename(final_mix_path)),
                'converted_vocals': url_for('serve_result_file', task_id=task_id, filename=os.path.basename(converted_vocals_path)),
                'instrumental': url_for('serve_result_file', task_id=task_id, filename=os.path.basename(all_other_stems[0])) if all_other_stems else None
            }
            send_progress(task_id, "处理完成！", 100, final_result)
        except Exception as e:
            logger.error(f"[{task_id}] 处理过程中发生错误: {e}", exc_info=True)
            send_progress(task_id, f"错误: {e}", -1)
        finally:
            if input_data.get('type') in ['file', 'url'] and temp_file_path and os.path.exists(temp_file_path):
                 os.remove(temp_file_path)

# --- 新增: GPU 工人线程 ---
def gpu_worker():
    global GPU_WORKER_STATE
    logger.info("GPU 工人线程已启动，等待任务...")
    while True:
        try:
            task_id, input_data, params = task_queue.get()
            
            # *** 关键修正点: 设置GPU状态为“处理中” ***
            with worker_state_lock:
                GPU_WORKER_STATE = 'processing'
            
            logger.info(f"工人线程从队列中获取到新任务: {task_id}")
            processing_task(task_id, input_data, params)

        except Exception as e:
            logger.critical(f"GPU工人线程发生致命错误: {e}", exc_info=True)
            # 即使发生错误，也要确保状态被重置
            send_progress(task_id, f"处理失败，发生严重错误: {e}", -1)
        finally:
            # *** 关键修正点: 任务完成后，设置GPU状态为“空闲” ***
            with worker_state_lock:
                GPU_WORKER_STATE = 'idle'

            task_queue.task_done()
            logger.info(f"工人线程完成任务 {task_id}，返回空闲状态。")

# --- Flask API 路由 ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/get_models', methods=['GET'])
def api_get_models(): return jsonify(get_all_application_models())

@app.route('/api/queue_status', methods=['GET'])
def api_queue_status():
    """
    *** 关键修正点: 返回更精确的系统负载状态 ***
    """
    with worker_state_lock:
        is_processing = 1 if GPU_WORKER_STATE == 'processing' else 0
    
    # 总任务数 = 正在处理的任务(0或1) + 在队列中等待的任务
    total_tasks_in_system = is_processing + task_queue.qsize()
    
    return jsonify({'tasks_in_system': total_tasks_in_system})

@app.route('/results/<task_id>/<path:filename>')
def serve_result_file(task_id, filename):
    directory = os.path.abspath(os.path.join(app.config['OUTPUT_FOLDER'], task_id))
    return send_from_directory(directory, filename, as_attachment=False)

@app.route('/api/get_enhancements/<model_id>')
def api_get_enhancements(model_id):
    all_models = get_all_application_models()
    model_info = next((m for m in all_models.get('conversion_models', []) if m['id'] == model_id), None)
    if model_info and 'enhancements' in model_info: return jsonify(model_info['enhancements'])
    return jsonify({}), 404

@app.route('/api/process', methods=['POST'])
def process_audio_file():
    task_id = str(uuid.uuid4())
    try:
        params = json.loads(request.form.get('params', '{}'))
        input_type = request.form.get('input_type')
        if input_type == 'file':
            file = request.files['audio_file']
            temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{file.filename}")
            file.save(temp_file_path)
            input_data = {'type': 'file', 'value': temp_file_path}
        elif input_type == 'url':
            input_data = {'type': 'url', 'value': request.form.get('url'), 'platform': request.form.get('platform')}
        else: return jsonify({"error": "无效的输入类型"}), 400
        
        task_queue.put((task_id, input_data, params))
        queue_position = task_queue.qsize()
        logger.info(f"新任务 {task_id} 已加入队列，当前队列长度: {queue_position}")
        send_progress(task_id, f"任务已提交，您正在排队，前方还有 {queue_position - 1} 个任务。", 0)
    except Exception as e:
        logger.error(f"处理请求时出错: {e}", exc_info=True)
        return jsonify({"error": f"请求参数错误: {e}"}), 400
    return jsonify({"task_id": task_id})


def send_progress(task_id, message, progress, data=None):
    if task_id not in tasks: tasks[task_id] = []
    tasks[task_id].append(json.dumps({"message": message, "progress": progress, "data": data}))

@app.route('/stream/<task_id>')
def stream(task_id):
    def event_stream(): 
        last_sent_index = 0
        while True:
            if task_id in tasks and len(tasks[task_id]) > last_sent_index:
                for i in range(last_sent_index, len(tasks[task_id])): yield f"data: {tasks[task_id][i]}\n\n"
                last_sent_index = len(tasks[task_id])
                last_event = json.loads(tasks[task_id][-1])
                if last_event['progress'] >= 100 or last_event['progress'] < 0:
                    tasks.pop(task_id, None); break
            time.sleep(0.5)
    return Response(event_stream(), mimetype='text/event-stream')


if __name__ == '__main__':
    worker_thread = threading.Thread(target=gpu_worker, daemon=True)
    worker_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True)
