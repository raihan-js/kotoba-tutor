"""Diagnostic 2: trivially correct sentences (model harshness vs data quality)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from diag_prompt import P  # noqa: E402
from kotoba.graph import make_llm_backend  # noqa: E402


def main() -> None:
    llm = make_llm_backend()
    for s in ["猫が好き。", "わたしは学生です。", "今日はいい天気ですね。",
              "彼は毎朝六時に起きます。"]:
        print(repr(s), "->", llm(P + s).strip().splitlines()[0][:60])


if __name__ == "__main__":
    main()
