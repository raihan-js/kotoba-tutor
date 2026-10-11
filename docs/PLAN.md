# Kotoba-Tutor — design spec (approved 2026-10-11)

Voice-first Japanese tutor agent in LangGraph. Listens to a learner speak
Japanese, corrects grammar/word choice, scores pitch accent with the Akusento
model, remembers recurring mistakes in a cross-session store, schedules
targeted practice. Every correction type evaluated on LangSmith.

Owner decisions (2026-10-11): build order risk-first (M0 VRAM spike first);
dataset from public learner-error corpora + disclosed supplements, no fluent
checker (disclosed, Akusento pattern); LLM = vLLM + Swallow-8B 4-bit;
interface = push-to-talk CLI, web UI deferred.

## M0 — VRAM spike (1 day)

Load vLLM (Swallow-8B-AWQ; exact weight ID pinned here in M0) +
faster-whisper int8 in the kotoba venv and SBV2JA in the akusento tts venv
(separate venvs over a numpy ABI split — one process is impossible, so record
per-venv peaks plus nvidia-smi total) on the RTX 3060 12GB; record to
`results/vram.json`.
Go/no-go: all-GPU, else ASR→CPU fallback (planned, not a failure). No tutor
code. Downloads need approval if >$5 (HF is free; just disk).

## M1 — correction dataset (4 days)

Survey public learner-error corpora for research-compatible licences (eval use
only where permitted). Fill gaps with rule-injected errors on Tatoeba (CC-BY)
as a disclosed supplement arm — never mixed silently. 300 sentences tagged:
particles は/が/を/に, verb conjugation, politeness level, word choice.
Stored as a LangSmith dataset (needs owner LangSmith key). Disclosure: labels
are corpus-derived or script-generated; no fluent reader checked them.

## M2 — tutor graph (6 days)

Nodes: listen (silero VAD, from koe-bridge) → transcribe (faster-whisper) →
analyse (Swallow, structured error tags) → accent (akusento scorer) →
explain (short, level-appropriate) → update store (mistake counts per
type/word) → schedule next exercise. Push-to-talk CLI. Latency budget
measured and reported, never promised. Reuse, don't rebuild: VAD/ASR/TTS
wrappers and the scorer come from sibling repos.

## M3 — evaluation (4 days)

Pre-registered before running. Primary: false-correction rate (correct
Japanese flagged wrong) + detection recall per error type on the 300.
Tutor-with-memory vs memoryless baseline (same LLM, no store) over a
simulated 2-week learner history. Explanation quality by rule-based proxies
only (length cap, restates the correction) — no LLM judge, ever. Never claim
learning outcomes from simulated histories.

## M4 — release (3 days)

GitHub repo + CI, HF dataset (licence permitting; else private/request), demo
recorded by the owner using it himself, EN write-up + JA summary, article
queued last, portfolio card, both resumes. Privacy note: audio processed
locally, deleted by default; no voice data published (research-only voices).

## Non-goals (v1)

Web UI, real-learner study, learning-gain claims, published TTS weights,
grammar explanations beyond the four tagged error types.
