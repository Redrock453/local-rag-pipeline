import pytest

from rag_pipeline.secret_scan import is_clean


@pytest.mark.parametrize("text", [
    "plain documentation text",
    "Use environment variables for credentials.",
    "",
])
def test_clean_text_passes(text):
    assert is_clean(text)


@pytest.mark.parametrize("text", [
    "password = hunter2hunter2",
    "api_key=abcdefghijklmnopqrstuvwxyz",
    "-----BEGIN RSA PRIVATE KEY-----",
    "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9abc",
    "ghp_" + "a" * 36,
    "sk-" + "b" * 24,
])
def test_secrets_are_blocked(text):
    assert not is_clean(text)
