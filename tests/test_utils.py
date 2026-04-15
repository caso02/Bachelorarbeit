"""Tests for src/utils.py — .env loading, JSON parsing, Gemini client caching, retry."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils import load_dotenv, parse_llm_json, get_gemini_client, _gemini_clients, call_gemini_with_retry


# ---------------------------------------------------------------------------
# load_dotenv
# ---------------------------------------------------------------------------

class TestLoadDotenv:
    def test_loads_key_value_pairs(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO_TEST_123=bar\nBAZ_TEST_123=qux\n")

        # Ensure keys don't exist yet
        os.environ.pop("FOO_TEST_123", None)
        os.environ.pop("BAZ_TEST_123", None)

        load_dotenv(env_file)

        assert os.environ["FOO_TEST_123"] == "bar"
        assert os.environ["BAZ_TEST_123"] == "qux"

        # Cleanup
        del os.environ["FOO_TEST_123"]
        del os.environ["BAZ_TEST_123"]

    def test_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('QUOTED_TEST_123="hello world"\n')
        os.environ.pop("QUOTED_TEST_123", None)

        load_dotenv(env_file)

        assert os.environ["QUOTED_TEST_123"] == "hello world"
        del os.environ["QUOTED_TEST_123"]

    def test_skips_comments_and_empty(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# this is a comment\n\nVALID_TEST_123=yes\n")
        os.environ.pop("VALID_TEST_123", None)

        load_dotenv(env_file)

        assert os.environ["VALID_TEST_123"] == "yes"
        assert "# this is a comment" not in os.environ
        del os.environ["VALID_TEST_123"]

    def test_does_not_overwrite_existing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_TEST_123=new_value\n")
        os.environ["EXISTING_TEST_123"] = "original"

        load_dotenv(env_file)

        assert os.environ["EXISTING_TEST_123"] == "original"
        del os.environ["EXISTING_TEST_123"]

    def test_missing_file_is_noop(self):
        load_dotenv("/nonexistent/path/.env")  # Should not raise


# ---------------------------------------------------------------------------
# parse_llm_json
# ---------------------------------------------------------------------------

class TestParseLlmJson:
    def test_clean_json(self):
        raw = '{"nudge_text": "Hello", "explanation": "Because"}'
        result = parse_llm_json(raw)
        assert result["nudge_text"] == "Hello"
        assert result["explanation"] == "Because"

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"nudge_text": "Hi", "explanation": "Test"}\n```'
        result = parse_llm_json(raw)
        assert result["nudge_text"] == "Hi"

    def test_json_embedded_in_text(self):
        raw = 'Here is the result:\n{"decision": "APPROVED", "reasoning": "OK"}\nDone.'
        result = parse_llm_json(raw)
        assert result["decision"] == "APPROVED"

    def test_fallback_on_garbage(self):
        raw = "This is not JSON at all"
        result = parse_llm_json(raw)
        assert result["nudge_text"] == "This is not JSON at all"

    def test_custom_fallback_keys(self):
        raw = "broken response"
        result = parse_llm_json(raw, fallback_keys={"decision": "REVISE", "reasoning": ""})
        assert result["decision"] == "REVISE"
        assert result["_raw"] == "broken response"

    def test_whitespace_handling(self):
        raw = '  \n  {"key": "value"}  \n  '
        result = parse_llm_json(raw)
        assert result["key"] == "value"


# ---------------------------------------------------------------------------
# call_gemini_with_retry
# ---------------------------------------------------------------------------

class TestCallGeminiWithRetry:
    def test_success_first_attempt(self):
        """Successful API call returns text immediately."""
        client = MagicMock()
        client.models.generate_content.return_value = MagicMock(text="Hello!")

        result = call_gemini_with_retry(client, "test-model", "prompt")

        assert result == "Hello!"
        assert client.models.generate_content.call_count == 1

    @patch("src.utils.time.sleep")
    def test_retry_on_429(self, mock_sleep):
        """429 error triggers retry and eventually succeeds."""
        client = MagicMock()
        # First call: 429 error, second call: success
        client.models.generate_content.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED. retry in 10s"),
            MagicMock(text="Success after retry"),
        ]

        result = call_gemini_with_retry(client, "test-model", "prompt")

        assert result == "Success after retry"
        assert client.models.generate_content.call_count == 2
        mock_sleep.assert_called_once_with(15)  # 10 + 5 buffer

    @patch("src.utils.time.sleep")
    def test_retry_default_wait(self, mock_sleep):
        """429 without retry-after info uses 65s default."""
        client = MagicMock()
        client.models.generate_content.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED"),
            MagicMock(text="OK"),
        ]

        call_gemini_with_retry(client, "m", "p")

        mock_sleep.assert_called_once_with(65)

    @patch("src.utils.time.sleep")
    def test_max_retries_raises(self, mock_sleep):
        """Exhausting max retries raises RuntimeError."""
        client = MagicMock()
        client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED")

        with pytest.raises(RuntimeError, match="Max retries"):
            call_gemini_with_retry(client, "m", "p", max_retries=2)

        assert client.models.generate_content.call_count == 2

    def test_non_429_error_raised_immediately(self):
        """Non-429 errors are raised without retry."""
        client = MagicMock()
        client.models.generate_content.side_effect = ValueError("Bad input")

        with pytest.raises(ValueError, match="Bad input"):
            call_gemini_with_retry(client, "m", "p")
