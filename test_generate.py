import json

import pytest

from rag_pipeline import generate as gen


def test_parse_plain_json():
    assert gen.parse_json_answer('{"message": "a", "thoughts": "b"}') == {
        "message": "a", "thoughts": "b"}


def test_parse_fenced_json():
    raw = '```json\n{"message": "a", "thoughts": "b"}\n```'
    assert gen.parse_json_answer(raw)["message"] == "a"


def test_parse_unstructured_text_is_wrapped():
    out = gen.parse_json_answer("just words")
    assert out["message"] == "just words"
    assert set(out) == {"message", "thoughts"}


def test_falls_back_to_looser_format(monkeypatch):
    calls = []

    def fake_post(cfg, payload):
        calls.append(payload.get("response_format", {}).get("type"))
        if len(calls) < 3:
            raise RuntimeError("format not supported")
        return json.dumps({"message": "ok", "thoughts": "t"})

    monkeypatch.setattr(gen, "_post_chat", fake_post)
    out = gen.openai_compatible_chat([{"role": "user", "content": "hi"}])
    assert out["message"] == "ok"
    assert calls == ["json_schema", "json_object", None]


def test_cascade_uses_ollama_when_primary_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(gen, "openai_compatible_chat", boom)
    monkeypatch.setattr(gen, "ollama_chat",
                        lambda messages, temperature=0.3: {"message": "local", "thoughts": "t"})
    assert gen.generate([{"role": "user", "content": "hi"}])["message"] == "local"


def test_all_providers_failing_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(gen, "openai_compatible_chat", boom)
    monkeypatch.setattr(gen, "ollama_chat", boom)
    with pytest.raises(RuntimeError, match="all providers failed"):
        gen.generate([{"role": "user", "content": "hi"}])
