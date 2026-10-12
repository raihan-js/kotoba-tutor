"""v1.1 analysis: confusion matrix + Wilson CIs + sample complaints."""
import json
import math
from collections import Counter


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - m) / d, 3), round((c + m) / d, 3))


def main() -> None:
    with open("results/v11_rows.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for arm in ["memory", "memoryless"]:
        print(f"== {arm} (n={len(rows)}) ==")
        cm = Counter((r["error_type"], r[arm].get("error_type")) for r in rows)
        for (g, p), n in sorted(cm.items()):
            print(f"  {g:12s} -> {p!s:12s} {n}")
        for et in ["particle", "conjugation", "politeness", "choice"]:
            sub = [r for r in rows if r["error_type"] == et]
            hit = sum(1 for r in sub if r[arm].get("has_error")
                      and r[arm].get("error_type") == et)
            print(f"  recall[{et}]: {hit}/{len(sub)} CI={wilson(hit, len(sub))}")
        neg = [r for r in rows if r["error_type"] == "none"]
        fc = sum(1 for r in neg if r[arm].get("has_error"))
        print(f"  false-correction: {fc}/{len(neg)} CI={wilson(fc, len(neg))}")
    print("== sample human-gold complaints (memoryless, has_error on noop) ==")
    shown = 0
    for r in rows:
        if r["error_type"] == "none" and r["memoryless"].get("has_error") and shown < 3:
            print(" L:", r["learner"][:70])
            print("   span:", r["memoryless"].get("span"),
                  "| corr:", str(r["memoryless"].get("correction"))[:70])
            shown += 1


if __name__ == "__main__":
    main()
