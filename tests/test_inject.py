"""Injector unit tests: every op fires on fixtures, tags exact, learner != source."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kotoba.inject import (
    INJECTORS,
    inject_choice,
    inject_conjugation,
    inject_particle,
    inject_politeness,
)


def test_particle_fires():
    r = inject_particle("私は学生です。")
    assert r is not None and r[0] != "私は学生です。" and r[1] == "は->が"


def test_particle_skips_non_case():
    assert inject_particle("食べてみる。") is None  # て is 接続助詞, no case particle


def test_conjugation_fires():
    r = inject_conjugation("昨日映画を見ました。")
    assert r is not None and r[1] == "ました->ます" and "見ます" in r[0]


def test_conjugation_none_when_clean():
    assert inject_conjugation("猫が好き。") is None


def test_politeness_desu_da():
    r = inject_politeness("これは本です。")
    assert r is not None and r[0] == "これは本だ。" and r[1] == "です->だ"


def test_politeness_masu_drop():
    r = inject_politeness("朝ごはんを食べます。")
    assert r is not None and r[0] == "朝ごはんを食べ。" and r[1] == "ます->∅"


def test_choice_fires():
    r = inject_choice("今日は暑いですね。")
    assert r is not None and r[0] == "今日は熱いですね。" and r[1] == "暑い->熱い"


def test_choice_none():
    assert inject_choice("静かな夜だ。") is None


def test_all_registered():
    assert set(INJECTORS) == {"particle", "conjugation", "politeness", "choice"}
