"""Rule-injected learner errors on correct Japanese sentences.

Framing (standard GEC eval): source = correct Tatoeba sentence (assumed
intended/gold), learner = source + one injected error. Gold tags are exact by
construction. Limitation: some injections are defensible variants of the
original — gold defines correctness; disclosed, biases against the tutor.
"""
from __future__ import annotations

from typing import Any

_TAGGER: Any = None


def tagger() -> Any:
    global _TAGGER
    if _TAGGER is None:
        import unidic_lite  # type: ignore[import-untyped]
        from fugashi import Tagger  # type: ignore[import-untyped]

        d = unidic_lite.DICDIR
        _TAGGER = Tagger(f"-r {d}/mecabrc -d {d}")
    return _TAGGER


def toks(text: str) -> list[tuple[str, str, str]]:
    """(surface, pos1, pos2) per token."""
    out = []
    for w in tagger()(text):
        out.append((w.surface, w.feature.pos1, w.feature.pos2))
    return out


PARTICLE_SWAP = {"は": "が", "が": "は", "を": "に", "に": "を",
                 "で": "に", "へ": "に", "と": "や"}
PARTICLE_POS = ("格助詞", "係助詞")

CHOICE_PAIRS = [("暑い", "熱い"), ("寒い", "冷たい"), ("見る", "観る"),
                ("聞く", "聴く"), ("借りる", "貸す"), ("上手", "得意")]


def _swap(text: str, old: str, new: str) -> str:
    i = text.find(old)
    return text[:i] + new + text[i + len(old):] if i >= 0 else text


def inject_particle(text: str) -> tuple[str, str] | None:
    for surf, p1, p2 in toks(text):
        if p1 == "助詞" and p2 in PARTICLE_POS and surf in PARTICLE_SWAP:
            return _swap(text, surf, PARTICLE_SWAP[surf]), f"{surf}->{PARTICLE_SWAP[surf]}"
    return None


def inject_conjugation(text: str) -> tuple[str, str] | None:
    """Surface-form auxiliary swaps (unidic-lite splits ました into まし+た,
    so morphology can't see them; order matters: check long forms first)."""
    if "ました" in text:
        return _swap(text, "ました", "ます"), "ました->ます"
    if "ません" in text:
        return None  # polarity change, out of scope for conjugation
    if "ます" in text:
        return _swap(text, "ます", "ました"), "ます->ました"
    if "なかった" in text:
        return _swap(text, "なかった", "ない"), "なかった->ない"
    if any(p1 == "助動詞" and s == "ない" for s, p1, _p2 in toks(text)):
        return _swap(text, "ない", "なかった"), "ない->なかった"
    if "でした" in text:
        return _swap(text, "でした", "です"), "でした->です"
    if "です" in text:
        return _swap(text, "です", "でした"), "です->でした"
    if "ていた" in text:
        return _swap(text, "ていた", "ている"), "ていた->ている"
    if "ている" in text:
        return _swap(text, "ている", "ていた"), "ている->ていた"
    return None


def inject_politeness(text: str) -> tuple[str, str] | None:
    if "です" in text:
        return _swap(text, "です", "だ"), "です->だ"
    if text.rstrip("。！？.!?").endswith("だ"):
        i = text.rfind("だ")
        return text[:i] + "です" + text[i + 1:], "だ->です"
    if "ください" in text:
        return _swap(text, "ください", "くれ"), "ください->くれ"
    if "ます" in text:  # dropped ます: formal sentence left with bare stem
        return _swap(text, "ます", ""), "ます->∅"
    return None


def inject_choice(text: str) -> tuple[str, str] | None:
    for a, b in CHOICE_PAIRS:
        if a in text:
            return _swap(text, a, b), f"{a}->{b}"
    return None


INJECTORS = {"particle": inject_particle, "conjugation": inject_conjugation,
             "politeness": inject_politeness, "choice": inject_choice}
