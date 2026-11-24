"""Tests of the demo server logic that need no model."""
from demo import server


def test_reuse_while_typed_text_matches():
    last = {"text": "I would like to ", "ghost": "schedule a meeting"}
    assert server.still_valid(last, "I would like to ") == "schedule a meeting"
    assert server.still_valid(last, "I would like to sch") == "edule a meeting"
    assert server.still_valid(last, "I would like to x") is None            # typed something else
    assert server.still_valid(last, "I would like to schedule a meeting") is None   # nothing left to show
    assert server.still_valid(last, "I would like") is None                 # text was deleted
    assert server.still_valid(None, "abc") is None


def test_trim_keeps_one_word_for_word_level():
    assert server.trim(" movies to", "next_word") == " movies"
    assert server.trim("ies tonight", "partial_word") == "ies"
    assert server.trim(" a good film tonight. ", "sentence") == " a good film tonight."
    assert server.trim("   ", "next_word") == ""


def test_granularity_follows_the_policy():
    assert server.choose_granularity("Hel", "policy") == "partial_word"
    assert server.choose_granularity("He", "policy") is None
    assert server.choose_granularity("Hello ", "policy") == "next_word"
    long_text = "one two three four five six seven eight nine ten "
    assert server.choose_granularity(long_text, "policy") == "sentence"
    assert server.choose_granularity("He", "always") == "partial_word"
    assert server.choose_granularity("", "always") is None


def test_counters_with_a_fake_model(monkeypatch):
    monkeypatch.setattr(server, "generate", lambda text, g: ("lo there" if g == "partial_word" else "world", 10.0))
    server.reset("t")
    out = [server.suggest(t, "t") for t in ["H", "He", "Hel", "Hell", "Hello", "Hellx"]]
    assert [o["source"] for o in out] == ["skip", "skip", "model", "reuse", "reuse", "model"]
    assert out[3]["ghost"] == "o there"
    s = out[-1]["stats"]
    assert (s["key_presses"], s["model_calls"], s["reused"], s["skipped"]) == (6, 2, 2, 2)


def test_always_mode_calls_at_every_key(monkeypatch):
    monkeypatch.setattr(server, "generate", lambda text, g: ("x", 5.0))
    server.reset("a")
    out = [server.suggest(t, "a", "always") for t in ["H", "He", "Hel"]]
    assert [o["source"] for o in out] == ["model", "model", "model"]


def test_always_reuse_mode_reuses_but_never_skips(monkeypatch):
    monkeypatch.setattr(server, "generate", lambda text, g: ("ello", 5.0))
    server.reset("ar")
    out = [server.suggest(t, "ar", "always_reuse") for t in ["H", "He", "Hel", "Hex"]]
    assert [o["source"] for o in out] == ["model", "reuse", "reuse", "model"]
    assert server.choose_granularity("He", "always_reuse") == "partial_word"
