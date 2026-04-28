from __future__ import annotations

import os
import subprocess

import modal


APP_NAME = "korg-arch1-qwen3-t4-awq-vllm"
MODEL_NAME = "Qwen/Qwen3-8B-AWQ"
GPU_TYPE = "T4"
SERVE_PORT = 8000

MODEL_CACHE_PATH = "/vol/model-cache"
VLLM_CACHE_PATH = "/vol/vllm-cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm==0.10.2",
        "transformers==4.56.1",
        "huggingface_hub[hf_transfer]>=0.34.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODEL_CACHE_PATH,
            "VLLM_CONFIG_ROOT": VLLM_CACHE_PATH,
        }
    )
)

cache_volume = modal.Volume.from_name("korg-arch1-qwen3-t4-awq-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    scaledown_window=300,
    timeout=24 * 60 * 60,
    volumes={"/vol": cache_volume},
)
@modal.web_server(port=SERVE_PORT, startup_timeout=60 * 20)
def serve() -> None:
    os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
    os.makedirs(VLLM_CACHE_PATH, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVE_PORT),
        "--model",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--quantization",
        "awq",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "16",
        "--download-dir",
        MODEL_CACHE_PATH,
    ]

    subprocess.Popen(cmd)


@app.local_entrypoint()
def main() -> None:
    print("Deploy with: modal deploy arch-1/implementation/modal_vllm_qwen3_t4_awq.py")
