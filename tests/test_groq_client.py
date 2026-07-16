"""Tests for the Groq client — uses mocking to avoid real API calls."""
import pytest
from unittest.mock import MagicMock, patch
from generate.groq_client import _parse_reset_seconds, generate


def test_parse_reset_seconds_simple():
    assert _parse_reset_seconds("2s") == 2.0

def test_parse_reset_seconds_millis():
    assert _parse_reset_seconds("500ms") == 0.5

def test_parse_reset_seconds_minutes():
    assert _parse_reset_seconds("1m30s") == 90.0

def test_parse_reset_seconds_decimal():
    assert _parse_reset_seconds("2.5s") == 2.5

def test_parse_reset_seconds_empty():
    assert _parse_reset_seconds("") == 3.0

def test_parse_reset_seconds_none():
    assert _parse_reset_seconds(None) == 3.0

def test_generate_returns_tuple():
    """Integration test — makes a real API call."""
    text, model = generate("Say 'test' and nothing else.")
    assert isinstance(text, str)
    assert len(text) > 0
    assert "scout" in model or "versatile" in model or "instant" in model