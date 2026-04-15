"""Tests for src/agent_state.py — AgentState + AgentMessage."""

import json
import tempfile
from pathlib import Path

import pytest

from src.agent_state import AgentMessage, AgentState


# ---------------------------------------------------------------------------
# AgentMessage
# ---------------------------------------------------------------------------

class TestAgentMessage:
    def test_to_dict(self):
        msg = AgentMessage(
            sender="A", receiver="B",
            message_type="INFO", payload={"key": "val"},
        )
        d = msg.to_dict()
        assert d["sender"] == "A"
        assert d["receiver"] == "B"
        assert d["message_type"] == "INFO"
        assert d["payload"] == {"key": "val"}
        assert "timestamp" in d

    def test_default_payload_is_empty_dict(self):
        msg = AgentMessage(sender="A", receiver="B", message_type="X")
        assert msg.payload == {}

    def test_timestamp_auto_set(self):
        msg = AgentMessage(sender="A", receiver="B", message_type="X")
        assert msg.timestamp  # non-empty string


# ---------------------------------------------------------------------------
# AgentState — init & defaults
# ---------------------------------------------------------------------------

class TestAgentStateInit:
    def test_defaults(self):
        state = AgentState()
        assert state.query == ""
        assert state.posts == []
        assert state.analysis == {}
        assert state.clusters == []
        assert state.nudges == []
        assert state.final_nudges == []
        assert state.messages == []
        assert state.current_phase == "idle"
        assert state.iteration == 0
        assert state.max_iterations == 3
        assert state.errors == []

    def test_query_param(self):
        state = AgentState(query="health")
        assert state.query == "health"


# ---------------------------------------------------------------------------
# AgentState — Messaging API
# ---------------------------------------------------------------------------

class TestAgentStateMessaging:
    def test_add_message_returns_msg(self):
        state = AgentState()
        msg = state.add_message("EthicsAgent", "ActionAgent", "REVISE_REQUEST", {"x": 1})
        assert isinstance(msg, AgentMessage)
        assert msg.sender == "EthicsAgent"
        assert msg.receiver == "ActionAgent"
        assert msg.message_type == "REVISE_REQUEST"
        assert msg.payload == {"x": 1}
        assert len(state.messages) == 1

    def test_add_message_default_payload(self):
        state = AgentState()
        msg = state.add_message("A", "B", "INFO")
        assert msg.payload == {}

    def test_get_messages_for_receiver(self):
        state = AgentState()
        state.add_message("A", "B", "INFO")
        state.add_message("A", "C", "INFO")
        state.add_message("B", "C", "ERROR")

        msgs_c = state.get_messages_for("C")
        assert len(msgs_c) == 2
        assert all(m.receiver == "C" for m in msgs_c)

    def test_get_messages_for_with_type_filter(self):
        state = AgentState()
        state.add_message("A", "B", "INFO")
        state.add_message("C", "B", "ERROR")
        state.add_message("D", "B", "INFO")

        msgs = state.get_messages_for("B", message_type="INFO")
        assert len(msgs) == 2
        assert all(m.message_type == "INFO" for m in msgs)

    def test_get_messages_between(self):
        state = AgentState()
        state.add_message("A", "B", "INFO")
        state.add_message("B", "A", "RESPONSE")
        state.add_message("A", "B", "ERROR")

        msgs = state.get_messages_between("A", "B")
        assert len(msgs) == 2
        assert all(m.sender == "A" and m.receiver == "B" for m in msgs)


# ---------------------------------------------------------------------------
# AgentState — Snapshot
# ---------------------------------------------------------------------------

class TestAgentStateSnapshot:
    def test_snapshot_keys(self):
        state = AgentState(query="test")
        state.posts = [{"id": "1"}]
        state.clusters = [{"topic": "x"}]
        state.nudges = [{"text": "y"}]
        state.final_nudges = [{"text": "z"}]
        state.add_message("A", "B", "INFO")

        snap = state.snapshot()
        assert snap["query"] == "test"
        assert snap["n_posts"] == 1
        assert snap["n_clusters"] == 1
        assert snap["n_nudges"] == 1
        assert snap["n_final_nudges"] == 1
        assert snap["n_messages"] == 1
        assert snap["n_errors"] == 0
        assert len(snap["messages"]) == 1


# ---------------------------------------------------------------------------
# AgentState — Persist
# ---------------------------------------------------------------------------

class TestAgentStatePersist:
    def test_persist_creates_files(self):
        state = AgentState(query="test")
        state.posts = [{"id": "1", "text": "hello"}]
        state.analysis = {"n_clusters": 2}
        state.nudges = [{"nudge_text": "Join!", "explanation": "test"}]
        state.final_nudges = [
            {"nudge_text": "Join!", "ethics_decision": "APPROVED"},
        ]
        state.add_message("A", "B", "INFO", {"x": 1})

        with tempfile.TemporaryDirectory() as tmpdir:
            state.persist(tmpdir)
            path = Path(tmpdir)

            assert (path / "crew_posts.json").exists()
            assert (path / "crew_analysis.json").exists()
            assert (path / "crew_nudges.json").exists()
            assert (path / "final_nudges.json").exists()
            assert (path / "agent_messages.json").exists()

            # Verify final_nudges structure
            final = json.loads((path / "final_nudges.json").read_text())
            assert final["approved_count"] == 1
            assert final["rejected_count"] == 0
            assert len(final["nudges"]) == 1

    def test_persist_empty_state_creates_final_nudges(self):
        """Even with empty nudges, persist should create final_nudges.json."""
        state = AgentState()

        with tempfile.TemporaryDirectory() as tmpdir:
            state.persist(tmpdir)
            path = Path(tmpdir)

            # No posts → no crew_posts.json
            assert not (path / "crew_posts.json").exists()
            # final_nudges.json is always created
            assert (path / "final_nudges.json").exists()

    def test_persist_agent_messages(self):
        state = AgentState()
        state.add_message("Ethics", "Action", "REVISE_REQUEST", {"feedback": "too pushy"})
        state.add_message("Action", "Ethics", "REVISED_NUDGE", {"text": "new"})

        with tempfile.TemporaryDirectory() as tmpdir:
            state.persist(tmpdir)
            msgs = json.loads((Path(tmpdir) / "agent_messages.json").read_text())
            assert len(msgs) == 2
            assert msgs[0]["sender"] == "Ethics"
            assert msgs[1]["sender"] == "Action"
