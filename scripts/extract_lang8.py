"""Extract Japanese learner->correction pairs from Lang-8 raw (CPU-only).

Streams the .dat straight from the zip (no 1.2GB extraction). Filters:
Japanese in L2, skips TEC-JL + FLUTEC journal IDs (no eval leakage),
keeps pairs with >=1 correction differing from the learner.
Writes data/lang8_ja.jsonl {learner, gold, journal}. Research-only.
"""
import json
import re
import subprocess
import sys


def remove_tags(s: str) -> str:
    return re.sub(r"\[.*?\]", "", s).strip()


def main() -> None:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    sys.path.insert(0, "/tmp/opencode")
    from make_pairs import fluency_corpus as F
    from make_pairs import tecjl as T
    skip = set(T) | set(F)
    zpath = "data/corpora/lang8-raw.zip"
    inner = "lang-8-20111007-2.0/lang-8-20111007-L1-v2.dat"
    p = subprocess.Popen(["unzip", "-p", zpath, inner], stdout=subprocess.PIPE,
                         text=True, encoding="utf-8", errors="replace")
    assert p.stdout is not None
    n_pairs = n_journals = 0
    out = open("data/lang8_ja.jsonl", "w", encoding="utf-8")
    for line in p.stdout:
        try:
            row = json.loads(line)
            jid, l2 = int(row[0]), row[2]
            if "Japanese" not in l2 or jid in skip:
                continue
            n_journals += 1
            for src, trgs in zip(row[4] or [], row[5] or []):
                if not isinstance(src, str):
                    continue
                src = remove_tags(src.replace("\t", " "))
                for t in trgs or []:
                    if not isinstance(t, str):
                        continue
                    t = remove_tags(t.replace("\t", " "))
                    if t and t != src and 5 <= len(src) <= 300:
                        out.write(json.dumps(
                            {"learner": src, "gold": t, "journal": jid},
                            ensure_ascii=False) + "\n")
                        n_pairs += 1
                        break
                if n_pairs >= cap:
                    break
        except (ValueError, IndexError, TypeError):
            continue
        if n_pairs >= cap:
            break
    out.close()
    p.wait()
    print(f"journals: {n_journals}, pairs: {n_pairs}")


if __name__ == "__main__":
    main()
