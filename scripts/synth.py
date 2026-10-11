"""Synthesize one line with SBV2JA (runs under the akusento .venv-tts).
Usage: synth.py "text" out.wav  (CPU per M0 verdict; GPU would overflow.)
"""
import sys

sys.path.insert(0, "/home/raihan/Desktop/APPS/research-learn/koe-bridge/src")


def main() -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from koebridge.tts import SBV2JA
    from scipy.io import wavfile

    text, out = sys.argv[1], sys.argv[2]
    r = SBV2JA().synth(text)
    wavfile.write(out, r["sr"], r["wav"])
    print(f"wrote {out} sr={r['sr']} synth_s={r['latency_s']:.2f}")


if __name__ == "__main__":
    main()
