"""Kotoba-Tutor M2: LangGraph tutor (listen→transcribe→analyse→accent→
explain→update_store→schedule). All heavy backends injectable so tests run
without GPU. Canonical backends: faster-whisper (kotoba venv), vLLM Swallow
(kotoba venv), akusento scorer + SBV2 TTS (subprocesses in sibling venvs).
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

ROOT = Path(__file__).resolve().parent.parent.parent
KOEBRIDGE_SRC = Path("/home/raihan/Desktop/APPS/research-learn/koe-bridge/src")
AKUSENTO_PY = Path("/home/raihan/Desktop/APPS/research-learn/akusento/.venv/bin/python")
TTS_PY = Path("/home/raihan/Desktop/APPS/research-learn/akusento/.venv-tts/bin/python")


class TState(TypedDict, total=False):
    wav_path: str
    learner_id: str
    transcript: str
    words: list[dict[str, Any]]
    analysis: dict[str, Any]
    accent: dict[str, Any]
    explanation: str
    profile: dict[str, Any]
    exercise: str


State = TState


def transcribe_backend(wav_path: str) -> dict[str, Any]:
    # NOTE: pinned av build lacks open(metadata_errors) which stock
    # faster-whisper needs for file paths — feed float32 numpy instead.
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    from scipy.io import wavfile  # type: ignore[import-untyped]

    sr, y = wavfile.read(wav_path)
    audio = y.astype("float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    peak = float(abs(audio).max()) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly  # type: ignore[import-untyped]

        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype("float32")
    m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                     device="cuda", compute_type="int8")
    segs, _info = m.transcribe(audio, language="ja", beam_size=1,
                               word_timestamps=True)
    words = [{"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2)}
             for s in segs for w in (s.words or [])]
    text = "".join(w["w"] for w in words)
    return {"transcript": text, "words": words}


def make_llm_backend(model: str = "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4",
                     ) -> Callable[[str], str]:
    import os

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, quantization="awq", gpu_memory_utilization=0.75,
              max_model_len=2048, trust_remote_code=True)
    params = SamplingParams(max_tokens=256, temperature=0.0)

    def run(prompt: str) -> str:
        # Chat API (not completion): the RL-instruct model echoes instructions
        # in completion mode (M2 finding). /no_think stops rambling (M0).
        msgs: Any = [{"role": "user", "content": prompt + " /no_think"}]
        try:
            out = llm.chat(
                msgs, params,
                chat_template_kwargs={"enable_thinking": False})
            return out[0].outputs[0].text
        except (TypeError, ValueError):
            out = llm.generate([prompt + " /no_think"], params)
            return out[0].outputs[0].text

    return run


ANALYSE_PROMPT = """あなたは日本語教師です。学習者の文を直し、JSON一行だけで答える。
文: 私は学生です。
{{"has_error": true, "error_type": "particle", "span": "は", "correction": "私が学生です。", "explanation": "ここは「が」を使う。"}}
文: 猫が好き。
{{"has_error": false, "error_type": "none", "span": "", "correction": "猫が好き。", "explanation": ""}}
文: {text}
"""


def analyse_backend(text: str, llm_fn: Callable[[str], str]) -> dict[str, Any]:
    raw = llm_fn(ANALYSE_PROMPT.format(text=text))
    try:
        return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {"has_error": False, "error_type": "none", "span": "",
                "correction": text, "explanation": "", "_raw": raw[:200]}


def accent_backend(wav_path: str, transcript: str,
                   words: list[dict[str, Any]]) -> dict[str, Any]:
    """Score pronunciation via the akusento scorer (subprocess, akusento venv).
    Falls back to unscored (never invents a score)."""
    script = ROOT / "scripts" / "accent_score.py"
    payload = json.dumps({"wav": wav_path, "transcript": transcript,
                          "words": words})
    try:
        r = subprocess.run([str(AKUSENTO_PY), str(script)], check=False, input=payload,
                           capture_output=True, text=True, timeout=600)
        last = r.stdout.strip().splitlines()
        if not last:
            return {"scored": False,
                    "reason": "empty stdout: " + r.stderr.strip()[-200:]}
        return json.loads(last[-1])
    except (json.JSONDecodeError, subprocess.SubprocessError, OSError) as e:
        return {"scored": False, "reason": f"{type(e).__name__}"}


def tts_backend(text: str, out_wav: str) -> dict[str, Any]:
    script = ROOT / "scripts" / "synth.py"
    r = subprocess.run([str(TTS_PY), str(script), text, out_wav], check=False,
                       capture_output=True, text=True, timeout=300)
    ok = Path(out_wav).exists()
    return {"ok": ok, "wav": out_wav if ok else "", "stderr": r.stderr[-200:]}


EXERCISE_TEMPLATES = {
    "particle": "空欄に正しい助詞を入れましょう: {blank}",
    "conjugation": "正しい形に直しましょう: {blank}",
    "politeness": "丁寧語に直しましょう: {blank}",
    "choice": "正しい言葉を選びましょう: {a} / {b}: {blank}",
    "none": "復習: {blank}",
}


def build_graph(transcribe_fn: Callable[[str], dict[str, Any]] = transcribe_backend,
                llm_fn: Callable[[str], str] | None = None,
                accent_fn: Callable[..., dict[str, Any]] = accent_backend,
                tts_fn: Callable[[str, str], dict[str, Any]] = tts_backend,
                store: Any = None, store_path: str = "results/learner_store.json"
                ) -> Any:
    from langgraph.graph import END, StateGraph

    def n_transcribe(s: State) -> dict[str, Any]:
        return transcribe_fn(s["wav_path"])

    def n_analyse(s: State) -> dict[str, Any]:
        assert llm_fn is not None, "llm_fn required"
        return {"analysis": analyse_backend(s["transcript"], llm_fn)}

    def n_accent(s: State) -> dict[str, Any]:
        return {"accent": accent_fn(s["wav_path"], s["transcript"],
                                    s.get("words", []))}

    def n_explain(s: State) -> dict[str, Any]:
        assert llm_fn is not None, "llm_fn required"
        a = s["analysis"]
        if not a.get("has_error"):
            expl = "いいですね。その調子!"
        else:
            p = (f"学習者の文「{s['transcript']}」の誤り「{a.get('span')}」を"
                 f"20字以内の日本語で優しく説明して。 /no_think")
            expl = llm_fn(p).strip().split("\n")[0][:80]
        return {"explanation": expl}

    def n_store(s: State) -> dict[str, Any]:
        st = store if store is not None else {}
        lid = s.get("learner_id", "default")
        if not isinstance(st, dict):  # LangGraph BaseStore
            from langgraph.store.base import BaseStore  # type: ignore[import-untyped]

            assert isinstance(st, BaseStore)
            ns = ("kotoba", lid)
            item = st.get(ns, "profile")
            prof: dict[str, Any] = dict(item.value) if item else {"counts": {}, "n_sessions": 0}
            a = s["analysis"]
            if a.get("has_error"):
                et = a.get("error_type", "?")
                prof["counts"][et] = prof["counts"].get(et, 0) + 1
            prof["n_sessions"] += 1
            st.put(ns, "profile", prof)
            return {"profile": prof}
        a = s["analysis"]
        if a.get("has_error"):
            et = a.get("error_type", "?")
            prof["counts"][et] = prof["counts"].get(et, 0) + 1
        prof["n_sessions"] += 1
        st[lid] = prof
        Path(store_path).parent.mkdir(exist_ok=True)
        Path(store_path).write_text(json.dumps(st, ensure_ascii=False))
        return {"profile": prof}

    def n_schedule(s: State) -> dict[str, Any]:
        counts = s.get("profile", {}).get("counts", {})
        weak = max(counts, key=counts.get) if counts else "particle"
        return {"exercise": EXERCISE_TEMPLATES.get(weak, "{blank}").format(
            blank="(次の文で練習)", a="A", b="B")}

    g: Any = StateGraph(TState)
    g.add_node("transcribe", n_transcribe)
    g.add_node("analyse", n_analyse)
    g.add_node("accent", n_accent)
    g.add_node("explain", n_explain)
    g.add_node("store", n_store)
    g.add_node("schedule", n_schedule)
    g.set_entry_point("transcribe")
    g.add_edge("transcribe", "analyse")
    g.add_edge("analyse", "accent")
    g.add_edge("accent", "explain")
    g.add_edge("explain", "store")
    g.add_edge("store", "schedule")
    g.add_edge("schedule", END)
    return g.compile()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--learner", default="default")
    ap.add_argument("--no-tts", action="store_true")
    args = ap.parse_args()

    llm_fn = make_llm_backend()
    graph = build_graph(llm_fn=llm_fn)
    res = graph.invoke({"wav_path": args.wav, "learner_id": args.learner})
    print("transcript:", res.get("transcript"))
    print("analysis:", json.dumps(res.get("analysis"), ensure_ascii=False))
    print("accent:", json.dumps(res.get("accent"), ensure_ascii=False))
    print("explanation:", res.get("explanation"))
    print("exercise:", res.get("exercise"))
    if not args.no_tts and res.get("explanation"):
        out = f"results/explain_{args.learner}.wav"
        print("tts:", tts_backend(res["explanation"], out))


if __name__ == "__main__":
    sys.exit(main())  # type: ignore[func-returns-value]
