"""Command line interface: ``python -m rag_pipeline.cli ask|index|eval``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .indexer import index_directory
from .pipeline import run


def _ask(args: argparse.Namespace) -> int:
    result = run(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _index(args: argparse.Namespace) -> int:
    print(json.dumps(index_directory(args.path), indent=2))
    return 0


def _eval(args: argparse.Namespace) -> int:
    """Run the cases in a JSON file and report format-contract pass rate."""
    cases = json.loads(Path(args.file).read_text(encoding="utf-8"))
    passed = 0
    for case in cases:
        try:
            run(case["input"])
            ok, note = True, "ok"
        except NotImplementedError as exc:
            ok, note = case.get("expect") == "out_of_scope", f"skipped: {exc}"
        except Exception as exc:  # noqa: BLE001
            ok, note = False, f"failed: {exc}"
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['input'][:50]:50}  {note}")
    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="rag_pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.set_defaults(func=_ask)

    index = sub.add_parser("index", help="index a directory into Qdrant")
    index.add_argument("path")
    index.set_defaults(func=_index)

    ev = sub.add_parser("eval", help="run eval cases from a JSON file")
    ev.add_argument("file")
    ev.set_defaults(func=_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
