# Kotoba-Tutor — voice-first Japanese tutor agent (local, measured)

Listens to a learner speak Japanese, corrects grammar/word choice with a local
Swallow-8B model, scores pitch accent with the Akusento scorer, remembers
recurring mistakes in a LangGraph store, schedules targeted practice.
Every correction type evaluated on LangSmith against a labelled 300.

Design: `docs/PLAN.md`. Decisions: `docs/decisions.md` (append-only).
Pre-registration: `results/prereg_m3.md` (frozen before the eval run).

## Results

### M0 — VRAM spike: GO (`results/vram.json`)
vLLM (Qwen3-Swallow-8B AWQ, ctx 2048) + faster-whisper int8 peak at
10,873MB of 12,288MB. SBV2JA needs +1,777MB on GPU (overflow) → TTS runs on
CPU (1.43s/synth). Stood up only with flashinfer-sampler off, spawn workers,
ninja, and vLLM loaded first (documented in decisions).

### M1 — correction set: 300, balanced (`results/m1.json`, rows JSONL)
150 correct Tatoeba sentences (CC-BY 2.0 FR, assumed gold) + 150 with one
rule-injected error: particle 37, conjugation 37, politeness 37, choice 39.
Gold tags exact by construction. Live on LangSmith: `kotoba-corrections-300-v1`.
No fluent checker — disclosed; TEC-JL/NAIST access filed in parallel as v1.1.

### M2 — tutor graph: loop proven live
LangGraph nodes: transcribe → analyse (Swallow chat API, 1-shot JSON) →
accent (akusento scorer subprocess) → explain → store → schedule. CLI:
`PYTHONPATH=src python src/kotoba/graph.py --wav X.wav --learner ID`.
Proven on FLEURS audio and on a TTS-spoken error sentence (が flagged,
gold restored exactly, accent 0.5, explanation spoken back). 14 tests.

### M3 — eval: the tutor brain collapses (`results/m3.json`, rows JSONL)
Frozen-prompt, memory vs memoryless, same Swallow backend:

| arm | false-correction | recall part/conj/pol/choice | strict |
|-----|-----------------|-----------------------------|--------|
| memory | 0.92 | 0.92 / 0 / 0 / 0 | 0.13 |
| memoryless | 0.94 | 0.97 / 0 / 0 / 0 | 0.15 |

The model predicts **particle** on 285/300 rows and never the other three
types. Memory reminder: no detectable effect (valid null). False-correction
is an UPPER bound (gold rows are unverified crowdsource; 4/4 trivial
sentences pass). Fix is structural (constrained type output or classifier
head) — queued as v1.1, not claimed.

### v1.1 — human gold kills the confound (`results/v11.json`)
TEC-JL (LREC 2020, research-only, never redistributed): 1,288 rows —
812 human-verified correct + single-type errors (137/15/58/266).
Same frozen prompt, same backend:

| arm | false-correction [95% CI] | recall part/conj/pol/choice |
|-----|--------------------------|-----------------------------|
| memory | 0.947 [0.929, 0.960] | 0.94 / 0 / 0 / 0 |
| memoryless | 0.954 [0.938, 0.967] | 0.96 / 0 / 0 / 0 |

False-correction is HIGHER on verified-correct sentences — the
unverified-gold confound is dead; Swallow flags correct Japanese ~95% of
the time. Collapse replicates (particle ~1,240/1,288; free-form types like
"structure"/"punctuation" leak). Memory null again. Row texts stay on the
PC per licence (only rates published).

### v2 — supervised head collapses the other way (`results/lora_eval.json`)
LoRA r16 on llm-jp-3-1.8B, 50k Lang-8 pairs, 2 epochs: false-correction
0.954 → 0.085 BUT strict 0.14 → 0.012, recall ~0 everywhere. Diagnosis: SFT
collapse to copying — every training completion was has_error:true with
1-2 token edits, so repeating the input minimizes loss; it never saw a
correct example. Fix needs negatives + edit-weighted loss (v3, unstarted).
From trigger-happy base to copy-everything adapter: both failure modes
measured, neither shipped as a tutor.

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/build_m1.py   # needs Tatoeba TSV in data/
```

## Limits
- Analyse node unusable as a tutor as measured (M3); the loop works, the brain needs v1.1.
- Accent timing is a uniform-split approximation (no forced aligner on the PC).
- Simulated histories are not learners; no learning-gain claims.
- TTS voice is research-only (never published); audio processed locally, deleted by default.
