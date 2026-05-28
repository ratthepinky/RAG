"""
LLM 服务 - 模型常驻内存，通过 HTTP API 调用
启动后模型只加载一次，后续请求直接推理
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from threading import Lock

import torch
import torch_mlu  # 必须先导入才能检测 MLU
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# 全局模型
_tokenizer = None
_model = None
_model_lock = Lock()

# 配置
MODEL_PATH = "/AII-heyan/DeepSeek/DeepSeek-R1-Distill-Qwen-7B"

# 自动检测可用设备
def get_device():
    # 尝试 MLU
    if hasattr(torch, 'mlu') and torch.mlu.is_available():
        return torch.device("mlu")
    # 尝试 CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 默认 CPU
    return torch.device("cpu")

DEVICE = get_device()


def load_model():
    """加载模型（只执行一次）"""
    global _tokenizer, _model
    
    with _model_lock:
        if _model is not None:
            LOGGER.info("Model already loaded")
            return
        
        LOGGER.info(f"Loading LLM from {MODEL_PATH}...")
        
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        _model = _model.to(DEVICE)
        _model.eval()
        
        LOGGER.info("LLM loaded successfully!")


def generate(prompt: str, max_new_tokens: int = 512) -> dict:
    """生成回答"""
    global _tokenizer, _model
    
    if _model is None:
        return {"error": "Model not loaded"}
    
    t0 = time.time()
    
    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=_tokenizer.eos_token_id,
    )
    
    response = _tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = response.replace("<|im_end|>", "").replace("<|im_start|>", "")
    
    # 提取 assistant 输出
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1].strip()
    elif "assistant\n" in response:
        response = response.split("assistant\n")[-1].strip()
    
    elapsed = time.time() - t0
    
    return {
        "response": response,
        "elapsed_seconds": round(elapsed, 2),
    }


# ============= Flask API =============
def create_app():
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model_loaded": _model is not None})
    
    @app.route("/generate", methods=["POST"])
    def generate_api():
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 512)
        
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        try:
            result = generate(prompt, max_tokens)
            return jsonify(result)
        except Exception as e:
            LOGGER.error(f"Generation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/memory", methods=["GET"])
    def memory_info():
        """获取显存使用情况"""
        if hasattr(torch.mlu, 'memory_allocated'):
            allocated = torch.mlu.memory_allocated() / 1024**3
            reserved = torch.mlu.memory_reserved() / 1024**3
            return jsonify({
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
            })
        return jsonify({"message": "MLU memory info not available"})
    
    return app


if __name__ == "__main__":
    # 先加载模型
    load_model()
    
    # 启动服务
    app = create_app()
    print("\n" + "=" * 50)
    print("LLM Service started!")
    print("API endpoints:")
    print("  GET  /health     - Health check")
    print("  POST /generate   - Generate text")
    print("  GET  /memory     - Memory usage")
    print("=" * 50 + "\n")
    
    # 0.0.0.0 让局域网也能访问
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
