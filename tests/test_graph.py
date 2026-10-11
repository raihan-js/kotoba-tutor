"""Graph tests with fake backends (no GPU). Timeline: state flows through all
nodes; store counts mistakes; scheduler picks the weakest type."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kotoba.graph import analyse_backend, build_graph

FAKE_ANALYSIS = {"has_error": True, "error_type": "particle", "span": "は",
                 "correction": "私が学生です。", "explanation": "「は」と「が」の違い"}


def fake_llm(prompt: str) -> str:
    import json

    if "優しく説明" in prompt:
        return "「は」はテーマ、「が」は主語だよ。"
    return json.dumps(FAKE_ANALYSIS, ensure_ascii=False)


def test_analyse_parses_json():
    a = analyse_backend("私は学生です。", fake_llm)
    assert a["error_type"] == "particle" and a["has_error"] is True


def test_analyse_fallback_on_garbage():
    a = analyse_backend("x", lambda _p: "not json at all")
    assert a["has_error"] is False and a["correction"] == "x"


def test_graph_end_to_end_fake():
    store: dict = {}
    g = build_graph(
        transcribe_fn=lambda _w: {"transcript": "私は学生です。",
                                  "words": [{"w": "私", "s": 0.0, "e": 0.3}]},
        llm_fn=fake_llm,
        accent_fn=lambda *a: {"scored": True, "word_match": 0.5},
        tts_fn=lambda t, o: {"ok": True, "wav": o},
        store=store)
    res = g.invoke({"wav_path": "dummy.wav", "learner_id": "u1"})
    assert res["transcript"] == "私は学生です。"
    assert res["analysis"]["error_type"] == "particle"
    assert "は" in res["explanation"] or "が" in res["explanation"]
    assert store["u1"]["counts"] == {"particle": 1}
    assert "助詞" in res["exercise"]


def test_scheduler_no_history():
    g = build_graph(
        transcribe_fn=lambda _w: {"transcript": "猫が好き。", "words": []},
        llm_fn=lambda _p: '{"has_error": false, "error_type": "none"}',
        accent_fn=lambda *a: {"scored": False},
        store={})
    res = g.invoke({"wav_path": "d.wav", "learner_id": "u2"})
    assert res["exercise"]  # defaults to particle template


def test_store_accumulates():
    store: dict = {}
    g = build_graph(
        transcribe_fn=lambda _w: {"transcript": "x", "words": []},
        llm_fn=fake_llm,
        accent_fn=lambda *a: {},
        store=store)
    g.invoke({"wav_path": "d.wav", "learner_id": "u3"})
    g.invoke({"wav_path": "d.wav", "learner_id": "u3"})
    assert store["u3"]["counts"] == {"particle": 2}
    assert store["u3"]["n_sessions"] == 2
