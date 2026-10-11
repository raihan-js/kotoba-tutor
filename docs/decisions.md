# Kotoba-Tutor — decisions (append-only, newest first)

## 2026-10-11 — M0 VRAM spike: GO with SBV2-on-CPU (results/vram.json)
- Weights: tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4 (official org,
  ~5GB, pinned). Faster-whisper-large-v3-turbo int8 (already cached).
- vLLM stood up only with: VLLM_USE_FLASHINFER_SAMPLER=0 +
  VLLM_ATTENTION_BACKEND=FLASH_ATTN (flashinfer JIT fails on sm_86, same as
  FlipGate), WORKER_MULTIPROC_METHOD=spawn, venv bin on PATH (ninja), vLLM
  loaded FIRST (parent CUDA init kills forked workers). gpu_mem_util 0.75,
  ctx 2048 (0.55 refused: KV-cache minimum).
- Peak vLLM+whisper-GPU: 10,873MB. SBV2JA needs +1,777MB on GPU → 12,650MB
  exceeds 12,288MB. SBV2 on CPU synths in 1.43s → verdict GO, TTS on CPU.
- Qwen3-Swallow RL rambles (quiz generation) without thinking control; M2
  must use enable_thinking=False + tight prompts. 64tok in 1.18s.
- LangSmith key confirmed working (~/.config/langsmith/key, sees keiri data).

## 2026-10-11 — M1 corpus survey: Tatoeba+injection now, forms in parallel
- TEC-JL (error-tagged, ideal) and NAIST Lang-8 raw (1.3M JA pairs) are both
  behind Google-Form access requests — owner files those; cLang-8 rejected
  (machine-generated targets, not human gold).
- M1 builds on Tatoeba (CC-BY 2.0 FR, direct download) + rule-injected
  errors (precedent: JST 2014 pseudo-errors): exact gold tags by construction.
  150 correct + 150 injected (particles/conjugation/politeness/word-choice).
  TEC-JL/NAIST upgrade to v1.1 when access granted. Owner-approved.
