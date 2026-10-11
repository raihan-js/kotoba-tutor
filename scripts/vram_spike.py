"""M0 VRAM spike: can vLLM (Qwen3-Swallow-8B AWQ) + faster-whisper int8 share
the RTX 3060 12GB with SBV2JA (separate venv, measured separately)?

Polls nvidia-smi in a background thread for true peak (torch counters miss
vLLM's upfront reservation). Writes results/vram.json.
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

# Forked vLLM workers cannot re-init CUDA after the parent touched it.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def smi_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip().split()[0])


def main() -> None:
    peaks = {"baseline_mb": smi_mb(), "peak_mb": 0}
    stop = False

    def poll() -> None:
        while not stop:
            peaks["peak_mb"] = max(peaks["peak_mb"], smi_mb())
            time.sleep(0.5)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    lat: dict[str, float] = {}
    try:
        # vLLM FIRST: any parent-side CUDA init before the engine starts
        # breaks forked workers (even with spawn, order is safest).
        from vllm import LLM, SamplingParams

        t0 = time.monotonic()
        llm = LLM(model="tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4",
                  quantization="awq", gpu_memory_utilization=0.75,
                  max_model_len=2048, trust_remote_code=True)
        lat["vllm_load_s"] = round(time.monotonic() - t0, 1)
        t0 = time.monotonic()
        out = llm.generate(["日本語で「ありがとう」の使い方を1文で説明して。"],
                           SamplingParams(max_tokens=64, temperature=0.0))
        lat["vllm_64tok_s"] = round(time.monotonic() - t0, 2)
        gen_text: str = out[0].outputs[0].text

        import numpy as np
        from faster_whisper import WhisperModel

        t0 = time.monotonic()
        whisper = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                               device="cuda", compute_type="int8")
        lat["whisper_load_s"] = round(time.monotonic() - t0, 1)
        wav = np.zeros(16000 * 5, dtype=np.float32)
        t0 = time.monotonic()
        list(whisper.transcribe(wav, language="ja", beam_size=1))
        lat["whisper_5s_s"] = round(time.monotonic() - t0, 2)
    finally:
        stop = True
        th.join()
    peaks["peak_mb"] = max(peaks["peak_mb"], smi_mb())
    res = {"gpu": "RTX 3060 12GB", **peaks, "latencies": lat,
           "vllm_sample": gen_text[:200]}
    Path("results").mkdir(exist_ok=True)
    Path("results/vram.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
