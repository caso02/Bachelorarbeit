"""Tests for src/action_agent.py — ActionAgent nudge generation."""

from unittest.mock import patch, MagicMock

import pytest

from src.action_agent import ActionAgent


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """Test the prompt construction, especially revision context."""

    def _make_agent(self):
        """Create ActionAgent with mocked env."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}):
            return ActionAgent(api_key="fake-key")

    def test_basic_prompt_contains_cluster_info(self):
        agent = self._make_agent()
        prompt = agent._build_prompt(
            cluster_data={"topic_label": "health", "subcluster_id": 1, "n_posts": 10},
            eligible_posts=[{"text_clean": "I love running"}],
        )
        assert "health" in prompt
        assert "I love running" in prompt
        assert "VSD-Richtlinien" in prompt

    def test_prompt_with_keywords(self):
        agent = self._make_agent()
        prompt = agent._build_prompt(
            cluster_data={"topic_label": "tech", "keywords": ["python", "AI"]},
            eligible_posts=[{"text_clean": "test post"}],
        )
        assert "python" in prompt
        assert "AI" in prompt

    def test_prompt_with_revision_context(self):
        agent = self._make_agent()
        prompt = agent._build_prompt(
            cluster_data={"topic_label": "health"},
            eligible_posts=[{"text_clean": "test post"}],
            revision_context={
                "previous_attempt": "Join now! Don't miss out!",
                "rejection_reason": "Contains FOMO language.",
            },
        )
        # Revision section should be present
        assert "REVISION" in prompt
        assert "Join now! Don't miss out!" in prompt
        assert "Contains FOMO language" in prompt
        assert "NEUEN Vorschlag" in prompt
        # Standard ending should NOT be present
        assert "Generiere jetzt den Vernetzungsvorschlag" not in prompt

    def test_prompt_without_revision_context(self):
        agent = self._make_agent()
        prompt = agent._build_prompt(
            cluster_data={"topic_label": "health"},
            eligible_posts=[{"text_clean": "test post"}],
            revision_context=None,
        )
        assert "REVISION" not in prompt
        assert "Generiere jetzt den Vernetzungsvorschlag" in prompt


# ---------------------------------------------------------------------------
# regenerate_nudge
# ---------------------------------------------------------------------------

class TestRegenerateNudge:
    """Test the regenerate_nudge method (feedback-loop integration)."""

    def test_regenerate_calls_gemini_with_feedback(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}):
            agent = ActionAgent(api_key="fake-key")

        mock_response = MagicMock()
        mock_response.text = '{"nudge_text": "Improved nudge", "explanation": "Better now"}'

        with patch.object(agent, "_call_gemini", return_value=mock_response.text) as mock_call:
            result = agent.regenerate_nudge(
                cluster_data={"topic_label": "tech", "subcluster_id": 0},
                top_posts=[{"id": "p1", "text_clean": "hello", "readiness_score": 0.9}],
                previous_nudge_text="Bad nudge",
                ethics_feedback="Too pushy.",
            )

        assert result["nudge_text"] == "Improved nudge"
        assert result["explanation"] == "Better now"
        assert result["n_eligible"] == 1
        # Verify the prompt contains the feedback
        call_args = mock_call.call_args[0][0]
        assert "Bad nudge" in call_args
        assert "Too pushy" in call_args

    def test_regenerate_empty_posts(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}):
            agent = ActionAgent(api_key="fake-key")

        result = agent.regenerate_nudge(
            cluster_data={"topic_label": "tech"},
            top_posts=[],
            previous_nudge_text="old text",
            ethics_feedback="rejected",
        )
        assert result["nudge_text"] is None
        assert result["n_eligible"] == 0
