"""Upload eval artifacts to HF dataset raihan-js/kotoba-tutor-eval.

Contents: M1 correction rows (Tatoeba-derived, CC-BY 2.0 FR — attribution in
the dataset card), M1/M3 summaries, M3 prereg + per-row outputs. No audio, no
model weights. Re-run any time; files overwritten with newer results.
"""
from pathlib import Path

from huggingface_hub import HfApi

REPO = "kotoba-tutor-eval"
FILES = [
    "results/m1_rows.jsonl",
    "results/m1.json",
    "results/m3.json",
    "results/m3_rows.jsonl",
    "results/prereg_m3.md",
    "results/vram.json",
]

CARD = """---
license: cc-by-2.0
---
# kotoba-tutor-eval

Evaluation artifacts for [kotoba-tutor](https://github.com/raihan-js/kotoba-tutor)
(voice-first Japanese tutor agent in LangGraph, all-local pipeline).

- `m1_rows.jsonl`: 300 correction items — 150 correct + 150 with one
  rule-injected error (particle/conjugation/politeness/choice). Correct
  sentences: Tatoeba `jpn_sentences` (CC-BY 2.0 FR, (c) Tatoeba contributors,
  used under the licence). Gold = Tatoeba original; injection tags exact by
  construction (no fluent checker — disclosed).
- `m1.json`: dataset stats. `m3.json` + `m3_rows.jsonl`: memory-vs-memoryless
  eval (frozen prompt, see `prereg_m3.md`). `vram.json`: M0 VRAM spike.

Headline M3: the Swallow-8B analyse node predicts "particle" on 285/300 rows
and never the other three types; memory reminder has no detectable effect.
"""


def main() -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    for f in FILES:
        api.upload_file(path_or_fileobj=str(root / f), path_in_repo=f,
                        repo_id=f"raihan-js/{REPO}", repo_type="dataset")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=f"raihan-js/{REPO}", repo_type="dataset")
    print(f"uploaded {len(FILES)} files + card to raihan-js/{REPO}")


if __name__ == "__main__":
    main()
