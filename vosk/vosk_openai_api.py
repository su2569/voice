import os
import json
import io
import time
from collections import OrderedDict
from typing import Optional, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import traceback

# ---------- 应用初始化 ----------
app = FastAPI(
    title="Vosk OpenAI Compatible API",
    version="1.0.0",
    description="本地离线语音识别服务，兼容 OpenAI /v1/audio/transcriptions"
)

# ---------- 配置 ----------
MODEL_ROOT = "models"               # 模型存放根目录
MAX_CACHED_MODELS = 1               # 内存中最多缓存几个模型（1 = 最省内存）
DEFAULT_SAMPLING_RATE = 16000       # Vosk 要求的采样率

# ---------- 模型缓存（LRU） ----------
model_cache = OrderedDict()

# ---------- 辅助函数 ----------
def get_available_models() -> list:
    """返回所有模型ID（文件夹名）"""
    if not os.path.exists(MODEL_ROOT):
        return []
    return [d for d in os.listdir(MODEL_ROOT)
            if os.path.isdir(os.path.join(MODEL_ROOT, d))]

def load_model(model_id: str) -> Model:
    """按需加载模型，缓存满时淘汰最久未使用的模型"""
    # 如果已缓存，移到末尾并返回
    if model_id in model_cache:
        model_cache.move_to_end(model_id)
        return model_cache[model_id]

    # 检查模型是否存在
    model_path = os.path.join(MODEL_ROOT, model_id)
    if not os.path.exists(model_path):
        raise ValueError(f"Model '{model_id}' not found")

    # 淘汰最旧模型（当缓存满时）
    while len(model_cache) >= MAX_CACHED_MODELS:
        oldest_key, _ = model_cache.popitem(last=False)
        print(f"⚠️ 内存清理：卸载模型 '{oldest_key}'")

    # 加载新模型
    print(f"⏳ 正在加载模型 '{model_id}' ...")
    model = Model(model_path)
    model_cache[model_id] = model
    print(f"✅ 模型 '{model_id}' 已加载")
    return model

def convert_audio_to_pcm(audio_bytes: bytes) -> bytes:
    """将任意格式音频转换为 Vosk 所需的 PCM (16kHz, mono, 16-bit)"""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1).set_frame_rate(DEFAULT_SAMPLING_RATE).set_sample_width(2)
    return audio.raw_data

def format_response(text: str, response_format: str = "json") -> dict | str:
    """根据 response_format 返回不同格式"""
    if response_format == "text":
        return text
    elif response_format == "json":
        return {"text": text}
    elif response_format == "verbose_json":
        # 模拟 verbose_json（简化版）
        return {
            "text": text,
            "language": "zh",
            "duration": None,  # 无法准确计算
            "segments": []
        }
    elif response_format == "srt" or response_format == "vtt":
        # 对于字幕格式，简单返回纯文本（或可扩展）
        return text
    else:
        return {"text": text}  # 默认 json

# ---------- OpenAI 兼容端点 ----------

@app.get("/v1/models")
async def list_models():
    """列出所有可用模型（OpenAI 格式）"""
    available = get_available_models()
    data = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "vosk"
        }
        for model_id in available
    ]
    return {
        "object": "list",
        "data": data
    }

@app.post("/v1/audio/transcriptions")
async def transcription(
    file: UploadFile = File(..., description="音频文件"),
    model: str = Form(..., description="模型ID，必须指定"),
    language: Optional[str] = Form(None, description="语言代码（Vosk 忽略）"),
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Form("json"),
    temperature: Optional[float] = Form(None, description="温度（Vosk 忽略）"),
    prompt: Optional[str] = Form(None, description="提示词（Vosk 忽略）")
):
    """
    语音转文字端点，完全兼容 OpenAI API。
    - 必须提供 `model` 参数（对应 models/ 下的文件夹名）
    - 支持多种音频格式（需要 ffmpeg）
    - 支持多种返回格式
    """
    # 1. 检查模型是否存在并加载
    available = get_available_models()
    if model not in available:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model}' not found. Available: {available}",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            }
        )

    try:
        vosk_model = load_model(model)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Failed to load model: {str(e)}",
                    "type": "internal_error",
                    "code": "model_load_failed"
                }
            }
        )

    # 2. 读取并转换音频
    try:
        audio_bytes = await file.read()
        pcm_data = convert_audio_to_pcm(audio_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Audio processing error: {str(e)}. Ensure ffmpeg is installed.",
                    "type": "invalid_request_error",
                    "param": "file",
                    "code": "audio_format_unsupported"
                }
            }
        )

    # 3. 识别
    try:
        rec = KaldiRecognizer(vosk_model, DEFAULT_SAMPLING_RATE)
        if rec.AcceptWaveform(pcm_data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
        else:
            partial = json.loads(rec.PartialResult())
            text = partial.get("partial", "")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Recognition failed: {str(e)}",
                    "type": "internal_error",
                    "code": "recognition_failed"
                }
            }
        )

    # 4. 根据 response_format 返回
    result_data = format_response(text, response_format)
    if response_format == "text":
        return StreamingResponse(io.BytesIO(result_data.encode()), media_type="text/plain")
    else:
        return JSONResponse(content=result_data)

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "cached_models": list(model_cache.keys()),
        "available_models": get_available_models()
    }

# ---------- 启动说明 ----------
if __name__ == "__main__":
    print("请使用 uvicorn 启动：")
    print("uvicorn vosk_openai_api:app --host 0.0.0.0 --port 11404")
