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

## 2026-10-11 — M2 graph loop proven (fake tests 14/14 + live FLEURS + TTS roundtrip)
- LangGraph StateGraph needs a TypedDict schema; plain-dict state silently
  drops input keys between nodes (caught by tests).
- Qwen3-Swallow-RL echoes instructions in vLLM completion mode; chat API +
  1-shot example + /no_think returns valid JSON. Prompt braces must be
  doubled for .format (caught by tests).
- Pinned av build lacks open(metadata_errors) needed by stock
  faster-whisper for file paths — transcribe_backend feeds float32 numpy
  (scipy resample) instead. scipy added (free, small).
- Accent reuse path works: OpenJTalk labels + pyworld F0 + scorer.pkl via
  akusento-venv subprocess; whisper words bound the region, label morae
  split it uniformly (no forced aligner on PC — disclosed approximation;
  TTS audio 16/19 valid, FLEURS 21/54).
- Closed loop on synthetic learner audio: TTS spoke 僕が彼女無しに…,
  whisper transcribed, Swallow flagged が->は with exact gold restoration,
  accent 0.5, explanation spoken back to wav.
- TTS/explain voice is the SBV2 JVNv-F1 research voice (never published).

## 2026-10-11 — M3 eval: tutor brain collapses to "particle" (results/m3.json)
- Frozen-prompt result (prereg kept): false-correction 0.92/0.94,
  recall particle 0.92/0.97, conjugation/politeness/choice 0.0, strict 0.13.
- Confusion (m3_rows.jsonl): predicts particle on 285/300 rows, never the
  other three types. Type classification is collapsed, not just weak.
- Memory line: no detectable effect (0.92 vs 0.94; identical recall shape).
  Valid null outcome per prereg — the store only feeds the scheduler, and
  the profile reminder changes nothing.
- False-correction is an UPPER bound: gold "correct" rows are unverified
  Tatoeba crowdsource; sample complaint (方->人) is debatable, not clear
  model error. Calibration anchor: 4/4 trivial sentences judged 正しい —
  the model can judge, it collapses on real learner-style text.
- Neutral judge-first prompt retest: same all-誤り → not prompt framing.
- Implication (v1.1, not now): analyse needs constrained type output or a
  classifier head, not free JSON. M2 loop itself works end to end.

## 2026-10-12 — corpus access forms filed by owner (TEC-JL + NAIST Lang-8)
- Owner submitted both Google Forms (araihansikder@gmail.com, Independent
  Researcher Tokyo). Waiting on email replies with download links.
- On arrival: TEC-JL -> human-gold v1.1 eval set (replaces/augments Tatoeba
  gold); NAIST raw -> training pool (noisy corrections, never eval gold).
- If NAIST is silent after ~2 weeks, proceed without it (2013 form, may be
  dead). TEC-JL lab active since 2023, expected responsive.

## 2026-10-12 — disk cleanup (~52G freed, 94% -> 88%)
- Deleted: pip cache, HF datasets arrow cache (20G), orphan model blobs
  (ModernBERT/MiniLM/tiny-gpt2/Qwen1.5B/Kokoro/Qwen3.5-2B: 4.15G),
  koe-bridge data/mt + data/tts (4.3G), koe-bridge .venv (7.2G, shipped, no
  lock — rebuildable from pyproject if ever needed), Ollama qwen3.5s (9.3G),
  pycache sweep. Owner-approved scope (safe drops only).
- Kept: Swallow-AWQ + kotoba venv, whisper, wikipedia cache (SAE fuel),
  akusento venvs + SBV2 voice, edit venv (v1.1), fewhumans venv (active),
  FLEURS audio, fedproc-ledger (reviewer round), JSUT/JVS (hard to re-get).

## 2026-10-12 — v1.1 human-gold rerun: confound dead, collapse replicates
- TEC-JL eval (1,702 sents, dual human refs, ERRANT-JA tags) -> 1,288-row
  set: 812 verified-correct + single-type errors (137 part / 15 conj /
  58 polite / 266 choice). Mapping PART->particle, AUX->politeness,
  INFL/TENSE->conjugation, SPELL/OTHER/content->choice; validated on samples.
  LangSmith kotoba-tecjl-gold-v1 (private workspace, research use only).
- Frozen-prompt M3 rerun, same backend: false-correction 0.947/0.954
  CI [0.93, 0.97] (HIGHER than Tatoeba 0.92) — the unverified-gold confound
  is dead; the model genuinely flags correct Japanese ~95% of the time.
- Type collapse replicates exactly: particle predicted ~1,240/1,288;
  conjugation/politeness/choice recall 0.0 (conj CI [0, 0.20], n=15).
  Memory reminder null again. Free-form types leak (structure/character/
  punctuation) — more evidence for constrained output in the brain fix.
- Licence posture: TEC-JL + Lang-8 raw are research-only, NO redistribution.
  Row texts never leave the PC (no HF upload of v11 rows; only rate stats).
  LangSmith dataset is private-workspace research use. Cite LREC 2020 +
  IJCNLP 2011 in any write-up using the data.

## 2026-10-12 — brain-fix verdict: correction is the bottleneck, not typing
- Stage-B single-label LLM typing: never emits exact labels — RL model
  rambles into English chain-of-thought despite /no_think (diag_stageb).
  Output freedom is NOT the cause; stronger constraint can't fix rambling.
- Rule typer (pair-based, CPU): 146/150 on m1 gold pairs, 287/402 on v11
  gold pairs. Typing SOLVED given a good correction.
- Same typer on stage-A corrections: m1 recall 16/37 part, 5/37 conj,
  9/37 pol, 14/39 choice; v11 similar. Degrades to LLM levels because
  stage-A corrections are bad (strict 0.13-0.15).
- Also fixed: 74 TEC-JL rows with edits invisible in plain text (untestable)
  dropped from v11 (1,288 -> 1,214); run_m3 preserves all row keys now.
- Bottom line: the brain fix is the CORRECTOR (detection+correction
  quality), not the typer. Next options: constrained correction (retry +
  self-check loop), bigger/cold model, or supervised correction head.

## 2026-10-12 — fix-loop verdict: MIXED per prereg rule (results/fixloop.json)
- k=3 (temp 0.7, seed 0) + unflag-on-identical + rule-typer majority vote,
  Tatoeba 300: false-correction 0.94 -> 0.693 (DOWN, real gain), strict
  0.13-0.15 -> 0.12 (FLAT, not up). Prereg rule needed both: partial fix.
- Collapse broken: recall spreads (part 0.32, conj 0.08, pol 0.22, choice
  0.44) instead of all-particle. Voting diversifies; generator doesn't improve.
- Conclusion: detection side fixable with loops+voting; correction quality
  needs a better generator or supervised head. Brain-fix v2 = correction
  model, not plumbing. (Side note: a stray ollama vision server held 8GB of
  GPU mid-run; owner-approved kill, uvicorn server left alone.)

## 2026-10-13 — Lang-8 JA training pool extracted (50k pairs, CPU-only)
- scripts/extract_lang8.py streams the 1.2GB .dat from the zip (no
  extraction): Japanese-L2 journals minus TEC-JL/FLUTEC IDs (no eval
  leakage), pairs with a differing correction, cap 50k from 6,098 journals.
  Median learner length 24 chars. data/lang8_ja.jsonl (gitignored,
  research-only). Training pool for brain-fix v2 (supervised correction
  head) when GPU frees. Fixed a None-entry crash mid-run.

## 2026-10-14 — LoRA v2 verdict: FAILED per prereg rule (copies everything)
- LoRA r16 on llm-jp-3-1.8b, 50k Lang-8 pairs, 2 epochs (loss 0.65).
  Frozen TEC-JL 1,214: false-correction 0.954 -> 0.085 (DOWN) but strict
  0.14 -> 0.012 (COLLAPSED); recall ~0 all types (743+208+108+45+10 -> none).
- Diagnosis: SFT collapse to copying. All 50k training completions were
  has_error:true with 1-2 token edits — copying the input minimizes loss
  (~95% tokens right), so the adapter copies almost everything. It never saw
  a correct example either (no negatives).
- Rule needed strict UP + fc DOWN; got the reverse. Next (v3): add correct
  (has_error:false) examples + edit-weighted loss (upweight changed tokens),
  or contrastive objective. Ollama untouched throughout (never reloaded).

## 2026-10-15 — LoRA v3 verdict: PASSES prereg rule, correction still ~zero
- v3 (negatives + 4x novel-token weight, ~1.1 effective epochs): fc 0.166
  (< 0.5 ✓), strict 0.030 (vs v2 0.012, UP 2.5x ✓). Both prereg conditions
  met: PASS — but absolute correction quality remains near-zero (recall ~0
  all types; 102/112 particles -> none).
- Honest read: the operating point moved (conservative, stops flagging),
  correction not learned. Negatives taught restraint, not repair. v2 copied
  everything; v3 flags almost nothing; neither corrects. The rule was
  lenient; reporting the PASS with the absolutes beside it, no goalpost move.
- v3a postscript: the invalid weighted-only run was a useful accident —
  weighting alone (v2 0.085 fc) vs weighting+negatives (0.166 fc) suggests
  negatives cost some restraint while buying strict 0.012 -> 0.030.
