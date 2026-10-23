# local-rag-pipeline

A small, local-first RAG pipeline in plain Python: **classify → retrieve → generate → validate**.
It runs against your own Ollama and Qdrant, uses structured outputs so the model's reply always has
the same shape, and degrades gracefully when a component is down.

No framework, one runtime dependency (`requests`), about 500 lines of code.

```
question
   │
   ▼
classify ──► out of scope? ──► rejected (NotImplementedError)
   │ technical / system
   ▼
retrieve:  embed (Ollama, bge-m3) ──► vector search (Qdrant) ──► context block
   │                                   (any failure → empty context, pipeline continues)
   ▼
generate:  OpenAI-compatible API  ──(fails)──►  local Ollama
           json_schema → json_object → plain text + JSON parsing
   │
   ▼
validate:  both fields present and non-empty? ──no──► retry (max_attempts) ──► ValidationError
   │ yes
   ▼
{"message": "...", "thoughts": "..."}
```

## Why it is built this way

- **Graceful degradation.** Embedder, vector store or primary LLM can be unavailable; the pipeline
  still answers (without context, or from the local model) instead of crashing.
- **Structured outputs with fallbacks.** Providers differ in what they support. The generator tries
  strict `json_schema`, then `json_object` with the schema in the prompt, then plain text with JSON
  extraction, so one code path works across OpenRouter, vLLM, llama.cpp server and others.
- **Deterministic rule-based classifier.** Intent routing costs no LLM call and is trivially testable.
- **Secrets never reach the index.** The indexer filters chunks through a conservative regex scan
  before embedding. It is a safety net, not a replacement for gitleaks or trufflehog.
- **Idempotent indexing.** Point ids are `uuid5(path#chunk)`, so re-running the indexer updates
  points instead of duplicating them.

## Quick start

```bash
# 1. Dependencies: Qdrant and Ollama (or run Ollama on the host)
docker compose up -d
docker compose exec ollama ollama pull bge-m3          # embeddings
docker compose exec ollama ollama pull gpt-oss:20b     # local fallback model

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[ru]" for Russian lemma matching, ".[dev]" for tests
cp .env.example .env        # fill in LLM_API_KEY; export the variables or use your own loader

# 3. Index some documents and ask a question
python -m rag_pipeline.cli index examples/docs
python -m rag_pipeline.cli ask "Why does nginx reload fail after I edit the config?"
python -m rag_pipeline.cli eval examples/eval_cases.json
```

Use as a library:

```python
from rag_pipeline import run

answer = run("How do I start my docker compose stack in the background?")
print(answer["message"])
```

## Configuration

All settings are environment variables (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Primary OpenAI-compatible endpoint |
| `LLM_API_KEY` | empty | API key for the primary endpoint |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Primary model |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Embeddings and local fallback |
| `OLLAMA_MODEL` | `gpt-oss:20b` | Local fallback model |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Vector store |
| `RAG_COLLECTION` | `knowledge` | Collection name |
| `RAG_EMBED_MODEL` | `bge-m3` | Embedding model |
| `RAG_TOP_K` | `3` | Fragments added to the prompt |
| `RAG_SCORE_THRESHOLD` | `0.62` | Minimum similarity; tune on your own data |
| `RAG_TRACE_FILE` | unset | Append one JSON line per request (no user text is logged) |

## Project layout

```
src/rag_pipeline/
  classify.py     rule-based intent classifier (English, optional Russian lemmas)
  retrieve.py     embeddings + vector search + context formatting
  generate.py     structured outputs, provider cascade
  pipeline.py     orchestration and output validation
  indexer.py      chunking, secret filter, embedding, upsert
  secret_scan.py  regex filter for credentials
  cli.py          ask / index / eval commands
tests/            36 unit tests, no network needed
examples/         sample documents and eval cases
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The unit tests mock all network calls, so they run without Ollama, Qdrant or an API key.
The `eval` command is the end-to-end check and needs the live services.

## Limitations

- The classifier is keyword-based; extend `TECHNICAL_TERMS` for your domain or swap in a model.
- Chunking is paragraph-based, not token-aware.
- One embedding call per chunk; batch it if you index large corpora.
- Single-turn retrieval: the query is the last user message only.

## License

MIT
