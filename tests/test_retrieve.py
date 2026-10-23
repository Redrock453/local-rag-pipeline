from rag_pipeline import retrieve


def test_empty_hits_give_empty_context():
    assert retrieve.build_context([]) == ""


def test_hits_without_text_are_ignored():
    assert retrieve.build_context([{"payload": {"source": "x"}}]) == ""


def test_context_has_source_and_is_truncated():
    hit = {"payload": {"text": "a" * 1000, "source": "guide.md"}}
    ctx = retrieve.build_context([hit], snippet_chars=100)
    assert "[source: guide.md]" in ctx
    assert "a" * 100 in ctx and "a" * 101 not in ctx


def test_service_down_degrades_to_empty(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(retrieve.requests, "post", boom)
    assert retrieve.retrieve("anything") == ""
