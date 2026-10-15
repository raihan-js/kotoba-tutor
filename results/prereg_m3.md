# M3 pre-registration (written 2026-10-11, BEFORE running)

Dataset: LangSmith `kotoba-corrections-300-v1` (150 correct + 150 injected,
37/37/37/39 particle/conjugation/politeness/choice). Text-only: analyse node
on learner strings (audio loop proven separately in M2).

## Arms
- **memory**: analyse prompt + one profile line from a simulated 2-week
  history ("この学習者は particle の誤りが多い(12回)…に注意して").
  Simulated history is FIXED before the run (script-generated counts:
  particle 12, conjugation 4, politeness 2, choice 1) — same profile for all
  rows, both arms see the same rows.
- **memoryless**: identical prompt without the profile line. Same LLM
  (Qwen3-Swallow-8B AWQ, chat API, /no_think, temp 0), same rows.

## Metrics (primary first)
1. False-correction rate: fraction of the 150 correct rows flagged
   has_error=true. LOWER wins.
2. Detection recall per error type: fraction of injected rows with
   has_error=true AND error_type==gold type. HIGHER wins (reported per type,
   no pooling across types for claims).
3. Strict correction: correction == gold (exact string). Reported, secondary
   (model may produce valid variants; disclosed).
4. Explanation proxies: non-empty when has_error; ≤80 chars (graph truncates).
   Reported as pass rates, never as "quality scores".

## Decision rules
- Memory helps iff it improves (1) or (2) on ANY type without regressing (1).
- "No detectable difference" is a valid outcome; never claim learning gains
  (simulated history, not learners).
- JSON parse failures count as has_error=false (conservative: hurts recall,
  helps false-correction — disclosed direction).
- One run, temp 0, seed fixed by determinism of greedy decoding. No peeking:
  prompt frozen in this doc before the run (profile line template above).

## Amendment 2026-10-12: constrained-decoding arm (brain fix candidate)
Frozen-prompt arms showed type collapse (free-form error_type). New arm keeps
the identical prompt text but enforces a JSON schema via vLLM guided decoding
(error_type enum of the 5 known labels). Question: does the collapse come
from output freedom (fixable by constraint) or from inability (needs a
classifier head)? Decision rule: if constrained recall on non-particle types
stays 0, inability wins and the classifier head is next. Same rows, same
backend, temp 0. Recorded here BEFORE running.

## Amendment 2026-10-12: correction retry loop (the prescribed fix)
Typing is solved given good corrections (rule typer 97%/71% on gold pairs);
stage-A corrections are bad (strict 0.13). New method on the Tatoeba 300:
k=3 candidates (temp 0.7, seed 0) per row; unflag when correction==learner
(fixes incoherent flags); else rule-type each (learner, candidate) pair and
take the majority type with its candidate. Detection = any-flag across the 3.
Same metrics (false-correction, per-type recall, strict). Decision rule: fix
wins iff strict correction rises AND false-correction falls vs frozen-prompt
baseline (0.94 / 0.13-0.15). Recorded here BEFORE running.

## Amendment 2026-10-13: supervised correction head v2 (brain fix v2)
Stage-A correction is the binding constraint (strict 0.13-0.15). New method:
LoRA (r=16, all attention+MLP linears) on llm-jp-3-1.8b (eager attention),
trained on 50k Lang-8 JA pairs (learner->gold, prompt: 直せ + JSON same as
stage A), 2 epochs, batch 8 x accum 4, lr 2e-4, seed 0. Eval FROZEN on
TEC-JL 1,214 (same metrics: false-correction, per-type recall with the
pair-rule typer on (learner, predicted) pairs, strict exact-match) vs the
Swallow frozen-prompt baseline (fc 0.954, strict ~0.14). Decision rule: v2
wins iff strict rises AND false-correction falls. TEC-JL rows never train
(no leakage; Lang-8 extraction excluded TEC-JL/FLUTEC IDs). Recorded here
BEFORE training.

## Amendment 2026-10-15: supervised head v3 (negatives + edit-weighted loss)
v2 collapsed to copying (strict 0.012): all 50k completions were
has_error:true with 1-2 token edits, never a correct example. v3 changes:
(1) add ~10k correct Tatoeba sentences as has_error:false negatives
(correction==input; from the M1 pool, disjoint from eval); (2) edit-weighted
loss: completion tokens absent from the prompt token set get 4x weight
(documented crude novelty proxy at token-id level, no alignment attempted).
Same base (llm-jp-3-1.8b), same LoRA config, 2 epochs, lr 2e-4, seed 0.
Output: models/lora-v3 (v2 adapter kept untouched). Eval FROZEN TEC-JL 1,214
(same metrics, same rule typer) vs v2 (fc 0.085, strict 0.012) and Swallow
baseline (fc 0.954, strict ~0.14). Decision rule: v3 wins iff strict rises
vs v2 AND false-correction stays below 0.5. Recorded here BEFORE training.

## Correction 2026-10-15: v3a run invalid (negatives silently dropped)
The finished "v3" adapter trained with edit-weighted loss ONLY: the epoch
loop iterated len(rows) (50k) while negatives sat in texts/golds unused
(steps 3124 == v2 count; log line confirms 10050 negatives built but
untouched). Adapter preserved as models/lora-v3-weighted-only (useful
ablation: weighting alone). Script fixed (loop over len(texts), per-epoch
seed), relaunching as the true v3. Recorded BEFORE the relaunch.
