import pytest

from rag_pipeline.classify import classify


@pytest.mark.parametrize("text,expected", [
    ("", "unknown"),
    ("/status", "system"),
    ("!reboot now", "system"),
    ("how do I configure nginx behind docker?", "technical"),
    ("restart the worker", "system"),
    ("what a lovely day", "general"),
])
def test_intents(text, expected):
    assert classify(text) == expected


def test_technical_wins_over_system_words():
    assert classify("status of my python script") == "technical"
