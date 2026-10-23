from rag_pipeline import indexer


def test_chunks_respect_limit_and_keep_all_text():
    text = "\n\n".join(f"paragraph {i} " + "x" * 40 for i in range(30))
    chunks = indexer.chunk_text(text, max_chars=200)
    assert all(len(c) <= 200 for c in chunks)
    assert "paragraph 29" in chunks[-1]


def test_oversized_paragraph_is_split():
    chunks = indexer.chunk_text("y" * 450, max_chars=200)
    assert [len(c) for c in chunks] == [200, 200, 50]


def test_point_ids_are_deterministic_and_distinct():
    assert indexer.point_id("a.md", 0) == indexer.point_id("a.md", 0)
    assert indexer.point_id("a.md", 0) != indexer.point_id("a.md", 1)
    assert indexer.point_id("a.md", 0) != indexer.point_id("b.md", 0)


def test_secret_chunks_are_not_indexed(tmp_path, monkeypatch):
    (tmp_path / "ok.md").write_text("Safe documentation text.")
    (tmp_path / "bad.md").write_text("password = hunter2hunter2")
    sent = []
    monkeypatch.setattr(indexer, "embed", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(indexer, "ensure_collection", lambda dim: None)

    class Resp:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(indexer.requests, "put",
                        lambda url, **kw: sent.append(kw["json"]) or Resp())
    stats = indexer.index_directory(str(tmp_path))
    assert stats["indexed"] == 1 and stats["skipped_secret"] == 1
    assert all("hunter2" not in p["payload"]["text"]
               for batch in sent for p in batch["points"])


def test_excluded_dirs_are_skipped(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.md").write_text("hidden")
    (tmp_path / "a.md").write_text("visible")
    names = [p.name for p in indexer.iter_files(tmp_path)]
    assert names == ["a.md"]
