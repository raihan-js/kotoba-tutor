"""Live accent scoring for learner audio (runs under the akusento .venv).

Stdin JSON: {wav, transcript, words:[{w,s,e}]}. Method: OpenJTalk expected
H/L per mora (akusento.labels.accent) + F0 (pyworld) + the trained v2
logistic scorer. Timing approximation (documented): whisper words bound the
speech region; label morae split it uniformly (no forced aligner on the PC).
Outputs one JSON line: {scored, mora_match_rate, n_morae, n_valid}.
"""
import json
import sys
from pathlib import Path

AKUSENTO = Path("/home/raihan/Desktop/APPS/research-learn/akusento")
sys.path.insert(0, str(AKUSENTO / "src"))


def main() -> None:
    from akusento.labels.accent import extract
    from akusento.labels.f0 import extract_f0, load_wav_mono
    from akusento.scorer.features import phrase_feature_matrix
    from akusento.scorer.train import load

    payload = json.loads(sys.stdin.read())
    phrases = extract(payload["transcript"])
    morae = [m for p in phrases for m in p.morae]
    words = payload.get("words", [])
    if not morae or not words:
        print(json.dumps({"scored": False, "reason": "no morae or no word times"}))
        return
    y, sr = load_wav_mono(payload["wav"])
    f0res = extract_f0(y, sr)
    f0, t = f0res["f0"], f0res["t"]
    s0, s1 = float(words[0]["s"]), float(words[-1]["e"])
    bounds, idx = [], 0
    for p in phrases:
        n = len(p.morae)
        w = (s1 - s0) / max(len(morae), 1)
        segs = [{"start": s0 + (idx + j) * w, "end": s0 + (idx + j + 1) * w}
                for j in range(n)]
        idx += n
        bounds.append((p, segs))
    sc, clf, _meta = load(AKUSENTO / "models" / "scorer.pkl")
    hits = valid = 0
    for p, segs in bounds:
        X, ok = phrase_feature_matrix(f0, t, segs, len(p.morae))
        if not ok or not any(ok):
            continue
        pred = clf.predict(sc.transform(X))
        for j, (m, v) in enumerate(zip(p.morae, ok)):
            if v:
                valid += 1
                hits += int(bool(pred[j]) == m.high)
    print(json.dumps({"scored": valid > 0,
                      "mora_match_rate": round(hits / valid, 3) if valid else 0.0,
                      "n_morae": len(morae), "n_valid": valid}))


if __name__ == "__main__":
    main()
