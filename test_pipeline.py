import pytest

from rag_pipeline import pipeline


def _patch(monkeypatch, reply, context="ctx"):
    seen = {}
    monkeypatch.setattr(pipeline, "retrieve", lambda q: context)

    def fake_generate(messages, **kw):
        seen["messages"] = messages
        return reply() if callable(reply) else reply

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    return seen


def test_context_is_added_to_system_prompt(monkeypatch):
    seen = _patch(monkeypatch, {"message": "a", "thoughts": "b"}, context="KB FRAGMENT")
    pipeline.run("how to configure docker")
    assert "KB FRAGMENT" in seen["messages"][0]["content"]


def test_general_intent_is_rejected(monkeypatch):
    _patch(monkeypatch, {"message": "a", "thoughts": "b"})
    with pytest.raises(NotImplementedError):
        pipeline.run("what a lovely day")


def test_invalid_reply_is_retried_then_succeeds(monkeypatch):
    replies = iter([{"message": "", "thoughts": "b"}, {"message": "ok", "thoughts": "b"}])
    _patch(monkeypatch, lambda: next(replies))
    assert pipeline.run("docker problem")["message"] == "ok"


def test_invalid_reply_every_time_raises(monkeypatch):
    _patch(monkeypatch, {"message": "x"})
    with pytest.raises(pipeline.ValidationError):
        pipeline.run("docker problem")


def test_validate_strips_whitespace():
    out = pipeline.validate({"message": " hi ", "thoughts": " t "})
    assert out == {"message": "hi", "thoughts": "t"}
