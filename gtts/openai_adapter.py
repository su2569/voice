import os
import tempfile
import io
import logging
import shutil
import gc
import inspect
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
import genie_tts as genie

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# ================= 配置 =================
# 限制单次请求最大文本长度（防止生成过长的音频导致内存爆炸）
MAX_TEXT_LENGTH = 500

# ================= 扫描模型目录 =================
def find_genie_data():
    candidates = [
        os.path.abspath("./GenieData"),
        os.path.abspath("./genie_data"),
        os.path.abspath("./data"),
        os.getcwd(),
        os.path.expanduser("~/.genie_tts"),
    ]
    for path in candidates:
        model_subdir = os.path.join(path, "model")
        if os.path.isdir(model_subdir):
            files = os.listdir(model_subdir)
            if any(f.endswith((".ckpt", ".pth", ".onnx", ".bin", ".wav")) for f in files):
                logger.info(f"✅ 找到模型目录: {path}")
                return path
    return None

detected = find_genie_data()
if not detected:
    logger.error("❌ 未找到模型目录")
    exit(1)

model_dir = os.path.join(detected, "model")
os.environ["GENIE_DATA_DIR"] = detected
logger.info(f"📁 GENIE_DATA_DIR = {detected}")

# ================= 配置角色和参考音频 =================
character_name = "Elysia"
ref_text = ""
ref_audio_path = ""

wav_files = [f for f in os.listdir(model_dir) if f.lower().endswith('.wav') and f.lower() != 'sl.wav']
if wav_files:
    ref_audio = wav_files[0]
    ref_audio_path = os.path.join(model_dir, ref_audio)
    character_name = os.path.splitext(ref_audio)[0]
    ref_text = character_name
    
    sl_path = os.path.join(model_dir, "sl.wav")
    if not os.path.exists(sl_path):
        shutil.copy2(ref_audio_path, sl_path)
        logger.info(f"📄 已复制参考音频为 sl.wav")
    ref_audio_path = sl_path
else:
    logger.error("❌ 未找到 .wav 参考音频")
    exit(1)

logger.info(f"📌 角色名: {character_name}")

# ================= 全局单例：只加载一次模型 =================
try:
    # 加载角色
    if hasattr(genie, 'load_character'):
        genie.load_character(
            character_name=character_name,
            onnx_model_dir=model_dir,
            language="zh"
        )
        logger.info(f"✅ load_character 成功")
except Exception as e:
    logger.warning(f"⚠️ load_character 失败: {e}")

# 设置参考音频（只设置一次，全局生效）
try:
    genie.set_reference_audio(
        character_name=character_name,
        audio_path=ref_audio_path,
        audio_text=ref_text,
        language="zh"
    )
    logger.info(f"✅ set_reference_audio 成功（全局）")
except Exception as e:
    logger.error(f"❌ set_reference_audio 失败: {e}")
    exit(1)

# ================= 获取 tts 函数签名 =================
sig_tts = inspect.signature(genie.tts)
tts_params = list(sig_tts.parameters.keys())
logger.info(f"genie.tts 参数: {tts_params}")

def find_param(candidates):
    for p in candidates:
        if p in tts_params:
            return p
    return None

text_param = find_param(['text', 'input', 'txt'])
out_param = find_param(['save_path', 'output_file', 'output_path', 'out'])
char_param = find_param(['character_name', 'character', 'voice', 'speaker'])

logger.info(f"text={text_param}, out={out_param}, char={char_param}")

# ================= 创建 OpenAI 兼容 API =================
app = FastAPI(title="Genie-TTS OpenAI Compatible API")

class TTSRequest(BaseModel):
    model: str = "genie-tts"
    input: str
    voice: str = ""
    response_format: str = "wav"
    speed: float = 1.0
    model_config = ConfigDict(extra="allow")

@app.get("/v1/models")
async def list_models():
    return JSONResponse(content={
        "object": "list",
        "data": [{"id": "genie-tts", "object": "model", "created": 1683588000, "owned_by": "genie-tts"}]
    })

@app.post("/v1/audio/speech")
async def speech_endpoint(request: TTSRequest):
    # ========== 1. 输入验证 ==========
    if not request.input.strip():
        raise HTTPException(400, "输入文本不能为空")
    
    # 限制文本长度，防止内存爆炸
    if len(request.input) > MAX_TEXT_LENGTH:
        raise HTTPException(400, f"文本过长，最大支持 {MAX_TEXT_LENGTH} 字符")
    
    char_to_use = request.voice.strip() if request.voice.strip() else character_name

    # ========== 2. 构建调用参数 ==========
    kwargs = {}
    if text_param:
        kwargs[text_param] = request.input
    else:
        raise HTTPException(500, "未找到 text 参数")
    
    if char_param:
        kwargs[char_param] = char_to_use

    # ========== 3. 生成音频（直接返回 bytes，避免中间文件） ==========
    try:
        # 尝试直接获取 bytes 返回值（最省内存）
        if out_param is None:
            result = genie.tts(**kwargs)
            if isinstance(result, bytes) and len(result) > 0:
                audio_bytes = result
                # 手动触发垃圾回收
                gc.collect()
                return Response(audio_bytes, media_type="audio/wav")
            elif isinstance(result, str) and os.path.exists(result):
                with open(result, "rb") as f:
                    audio_bytes = f.read()
                os.remove(result)
                gc.collect()
                return Response(audio_bytes, media_type="audio/wav")
        
        # 如果有 save_path 参数，使用临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        kwargs[out_param] = temp_path
        result = genie.tts(**kwargs)
        
        # 检查是否直接返回了 bytes
        if isinstance(result, bytes) and len(result) > 0:
            audio_bytes = result
            if os.path.exists(temp_path):
                os.remove(temp_path)
            gc.collect()
            return Response(audio_bytes, media_type="audio/wav")
        
        # 检查文件
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()
            os.remove(temp_path)
            gc.collect()
        else:
            raise Exception("生成的音频为空")

    except Exception as e:
        logger.error(f"TTS 失败: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()
        raise HTTPException(500, f"TTS 引擎错误: {str(e)}")

    # ========== 4. 返回音频 ==========
    if request.response_format == "wav":
        return Response(audio_bytes, media_type="audio/wav")
    else:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
            buffer = io.BytesIO()
            audio.export(buffer, format=request.response_format)
            result_bytes = buffer.getvalue()
            # 释放大对象
            del audio
            del buffer
            gc.collect()
            return Response(result_bytes, media_type=f"audio/{request.response_format}")
        except ImportError:
            raise HTTPException(400, "未安装 pydub，请使用 wav")

# ================= 健康检查 =================
@app.get("/health")
async def health_check():
    return {"status": "ok", "memory_optimized": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=11559,
        workers=1,  # 单 worker 减少内存开销
        limit_concurrency=1,  # 限制并发，防止内存爆炸
        timeout_keep_alive=30
    )
