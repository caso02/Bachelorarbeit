"""Tests for src/community_crew.py — Agent, Task, Crew, Coordinator abstractions."""

from unittest.mock import patch, MagicMock

import pytest

from src.agent_state import AgentState
from src.community_crew import Agent, Task, Crew, Process, tool, Coordinator


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

class TestToolDecorator:
    def test_sets_tool_name(self):
        @tool("My Tool")
        def my_func(x=""):
            return x

        assert my_func._tool_name == "My Tool"
        assert my_func._is_tool is True

    def test_function_still_callable(self):
        @tool("Echo")
        def echo(x=""):
            return f"echo: {x}"

        assert echo("hello") == "echo: hello"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestAgent:
    def test_select_tool_single(self):
        @tool("Only Tool")
        def only(x=""):
            return "only"

        agent = Agent(role="Test", goal="", backstory="", tools=[only])
        fn = agent.select_tool()
        assert fn is only

    def test_select_tool_by_name(self):
        @tool("Alpha")
        def alpha(x=""):
            return "a"

        @tool("Beta")
        def beta(x=""):
            return "b"

        agent = Agent(role="Test", goal="", backstory="", tools=[alpha, beta])
        assert agent.select_tool("Alpha") is alpha
        assert agent.select_tool("Beta") is beta

    def test_select_tool_fallback(self):
        @tool("First")
        def first(x=""):
            return "1"

        @tool("Second")
        def second(x=""):
            return "2"

        agent = Agent(role="Test", goal="", backstory="", tools=[first, second])
        assert agent.select_tool("NonExistent") is first

    def test_run_tool(self):
        @tool("Greeter")
        def greet(x=""):
            return f"Hello {x}"

        agent = Agent(role="Test", goal="", backstory="", tools=[greet], verbose=False)
        assert agent.run_tool(input_="World") == "Hello World"

    def test_no_tools_raises(self):
        agent = Agent(role="Test", goal="", backstory="", tools=[])
        with pytest.raises(RuntimeError, match="Keine Tools"):
            agent.select_tool()


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class TestTask:
    def test_execute_first_task(self):
        @tool("Echo")
        def echo(x=""):
            return f"got: {x}"

        agent = Agent(role="Test", goal="", backstory="", tools=[echo], verbose=False)
        task = Task(description="test", agent=agent, expected_output="")
        result = task.execute(inputs={"query": "hello"})
        assert result == "got: hello"

    def test_execute_with_context(self):
        @tool("T1")
        def t1(x=""):
            return "step1_output"

        @tool("T2")
        def t2(x=""):
            return f"received: {x[:20]}"

        a1 = Agent(role="Agent1", goal="", backstory="", tools=[t1], verbose=False)
        a2 = Agent(role="Agent2", goal="", backstory="", tools=[t2], verbose=False)

        task1 = Task(description="first", agent=a1, expected_output="")
        task1.execute()

        task2 = Task(description="second", agent=a2, expected_output="", context=[task1])
        result = task2.execute()
        assert "Vorheriger Schritt" in result or "received:" in result

    def test_reset_clears_output(self):
        @tool("T")
        def t(x=""):
            return "output"

        agent = Agent(role="Test", goal="", backstory="", tools=[t], verbose=False)
        task = Task(description="test", agent=agent, expected_output="")
        task.execute()
        assert task._output == "output"
        task.reset()
        assert task._output == ""


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------

class TestCrew:
    def test_sequential_execution(self):
        results = []

        @tool("Step1")
        def step1(x=""):
            results.append(1)
            return "done1"

        @tool("Step2")
        def step2(x=""):
            results.append(2)
            return "done2"

        a1 = Agent(role="A1", goal="", backstory="", tools=[step1], verbose=False)
        a2 = Agent(role="A2", goal="", backstory="", tools=[step2], verbose=False)
        t1 = Task(description="t1", agent=a1, expected_output="")
        t2 = Task(description="t2", agent=a2, expected_output="", context=[t1])

        crew = Crew(agents=[a1, a2], tasks=[t1, t2], verbose=False)
        crew.kickoff()

        assert results == [1, 2]

    def test_error_recovery_stops_pipeline(self):
        call_log = []

        @tool("OK")
        def ok_step(x=""):
            call_log.append("ok")
            return "ok"

        @tool("Fail")
        def fail_step(x=""):
            raise ValueError("boom")

        @tool("After")
        def after_step(x=""):
            call_log.append("after")
            return "after"

        a1 = Agent(role="A1", goal="", backstory="", tools=[ok_step], verbose=False)
        a2 = Agent(role="A2", goal="", backstory="", tools=[fail_step], verbose=False)
        a3 = Agent(role="A3", goal="", backstory="", tools=[after_step], verbose=False)

        t1 = Task(description="t1", agent=a1, expected_output="")
        t2 = Task(description="t2", agent=a2, expected_output="", context=[t1])
        t3 = Task(description="t3", agent=a3, expected_output="", context=[t2])

        crew = Crew(agents=[a1, a2, a3], tasks=[t1, t2, t3], verbose=False)
        result = crew.kickoff()

        assert "ok" in call_log
        assert "after" not in call_log  # Step 3 should not run
        assert "FEHLER" in result

    def test_reset_between_runs(self):
        counter = {"n": 0}

        @tool("Count")
        def count(x=""):
            counter["n"] += 1
            return f"run{counter['n']}"

        a = Agent(role="A", goal="", backstory="", tools=[count], verbose=False)
        t = Task(description="t", agent=a, expected_output="")
        crew = Crew(agents=[a], tasks=[t], verbose=False)

        crew.kickoff()
        assert t._output == "run1"

        crew.kickoff()
        assert t._output == "run2"
        # _output should be from the latest run, not accumulated


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def _make_dummy_state(posts=None, clusters=None, nudges=None):
    """Helper: Create an AgentState pre-filled with test data."""
    state = AgentState(query="test")
    state.posts = posts or [
        {"id": "p1", "text_clean": "hello", "category": "tech"},
    ]
    state.clusters = clusters or [
        {
            "topic_label": "tech", "subcluster_id": 0, "n_posts": 1,
            "avg_readiness": 0.8,
            "top_posts": [{"id": "p1", "text_clean": "hello", "readiness_score": 0.8}],
        },
    ]
    state.nudges = nudges or [
        {
            "nudge_text": "Join the tech group!",
            "explanation": "You share tech interests.",
            "cluster_context": {"topic_label": "tech", "subcluster_id": 0, "n_posts": 1},
            "target_user_ids": ["p1"],
            "n_eligible": 1,
            "_top_posts": [{"id": "p1", "text_clean": "hello", "readiness_score": 0.8}],
        },
    ]
    return state


class TestCoordinator:
    """Tests for the Coordinator — mocking external dependencies."""

    def test_ethics_approved_happy_path(self):
        """When Ethics approves, nudge goes directly to final_nudges."""
        state = _make_dummy_state()
        coord = Coordinator(max_revisions=2, verbose=False)

        mock_ethics = MagicMock()
        mock_ethics.review_nudge.return_value = {
            "decision": "APPROVED",
            "reasoning": "All good.",
            "modified_text": None,
        }

        with patch("src.ethics_agent.EthicsAgent", return_value=mock_ethics):
            state = coord._ethics_with_revision_loop(state)

        assert len(state.final_nudges) == 1
        assert state.final_nudges[0]["ethics_decision"] == "APPROVED"
        assert state.final_nudges[0]["revision_count"] == 0

    def test_ethics_revised_happy_path(self):
        """When Ethics revises, the modified text is stored."""
        state = _make_dummy_state()
        coord = Coordinator(max_revisions=2, verbose=False)

        mock_ethics = MagicMock()
        mock_ethics.review_nudge.return_value = {
            "decision": "REVISE",
            "reasoning": "Minor tweak needed.",
            "modified_text": "Join the tech community!",
        }

        with patch("src.ethics_agent.EthicsAgent", return_value=mock_ethics):
            state = coord._ethics_with_revision_loop(state)

        assert len(state.final_nudges) == 1
        assert state.final_nudges[0]["ethics_decision"] == "REVISED"
        assert state.final_nudges[0]["nudge_text"] == "Join the tech community!"
        assert state.final_nudges[0]["original_nudge_text"] == "Join the tech group!"

    def test_revision_loop_approved_after_retry(self):
        """REJECTED → regenerate → APPROVED after 1 revision."""
        state = _make_dummy_state()
        coord = Coordinator(max_revisions=3, verbose=False)

        mock_ethics = MagicMock()
        # First call: REJECTED; second call (after revision): APPROVED
        mock_ethics.review_nudge.side_effect = [
            {"decision": "REJECTED", "reasoning": "Too pushy.", "modified_text": None},
            {"decision": "APPROVED", "reasoning": "Now it's fine.", "modified_text": None},
        ]

        mock_action = MagicMock()
        mock_action.regenerate_nudge.return_value = {
            "nudge_text": "Improved nudge text.",
            "explanation": "Better explanation.",
            "cluster_context": {"topic_label": "tech", "subcluster_id": 0, "n_posts": 1},
            "target_user_ids": ["p1"],
            "n_eligible": 1,
        }

        with patch("src.ethics_agent.EthicsAgent", return_value=mock_ethics), \
             patch("src.action_agent.ActionAgent", return_value=mock_action):
            state = coord._ethics_with_revision_loop(state)

        assert len(state.final_nudges) == 1
        assert state.final_nudges[0]["ethics_decision"] == "APPROVED"
        assert state.final_nudges[0]["revision_count"] == 1
        mock_action.regenerate_nudge.assert_called_once()

    def test_revision_loop_max_reached(self):
        """REJECTED stays REJECTED after max_revisions → no final nudge."""
        state = _make_dummy_state()
        coord = Coordinator(max_revisions=2, verbose=False)

        mock_ethics = MagicMock()
        # Always rejects
        mock_ethics.review_nudge.return_value = {
            "decision": "REJECTED",
            "reasoning": "Still bad.",
            "modified_text": None,
        }

        mock_action = MagicMock()
        mock_action.regenerate_nudge.return_value = {
            "nudge_text": "Still bad nudge.",
            "explanation": "Tried again.",
            "cluster_context": {"topic_label": "tech", "subcluster_id": 0},
            "target_user_ids": ["p1"],
            "n_eligible": 1,
        }

        with patch("src.ethics_agent.EthicsAgent", return_value=mock_ethics), \
             patch("src.action_agent.ActionAgent", return_value=mock_action):
            state = coord._ethics_with_revision_loop(state)

        # No final nudges — rejected after max attempts
        assert len(state.final_nudges) == 0
        # Action agent called max_revisions times
        assert mock_action.regenerate_nudge.call_count == 2
        # MAX_REVISIONS message logged
        max_msgs = [m for m in state.messages if m.message_type == "MAX_REVISIONS_REACHED"]
        assert len(max_msgs) == 1

    def test_skips_empty_nudge_text(self):
        """Nudges without nudge_text are skipped (not sent to Ethics)."""
        state = _make_dummy_state(nudges=[
            {"nudge_text": None, "cluster_context": {}, "target_user_ids": []},
        ])
        coord = Coordinator(verbose=False)

        mock_ethics = MagicMock()

        with patch("src.ethics_agent.EthicsAgent", return_value=mock_ethics):
            state = coord._ethics_with_revision_loop(state)

        mock_ethics.review_nudge.assert_not_called()
        assert len(state.final_nudges) == 0
        skipped_msgs = [m for m in state.messages if m.message_type == "SKIPPED"]
        assert len(skipped_msgs) == 1

    def test_messages_audit_trail(self):
        """The revision loop creates proper audit-trail messages."""
        state = _make_dummy_state()
        coord = Coordinator(max_revisions=3, verbose=False)

        mock_ethics = MagicMock()
        mock_ethics.review_nudge.side_effect = [
            {"decision": "REJECTED", "reasoning": "Too pushy.", "modified_text": None},
            {"decision": "APPROVED", "reasoning": "OK now.", "modified_text": None},
        ]

        mock_action = MagicMock()
        mock_action.regenerate_nudge.return_value = {
            "nudge_text": "Better text.",
            "explanation": "Improved.",
            "cluster_context": {"topic_label": "tech", "subcluster_id": 0},
            "target_user_ids": ["p1"],
            "n_eligible": 1,
        }

        with patch("src.ethics_agent.EthicsAgent", return_value=mock_ethics), \
             patch("src.action_agent.ActionAgent", return_value=mock_action):
            state = coord._ethics_with_revision_loop(state)

        # Should have: REVISE_REQUEST, REVISED_NUDGE, APPROVED_AFTER_REVISION
        msg_types = [m.message_type for m in state.messages]
        assert "REVISE_REQUEST" in msg_types
        assert "REVISED_NUDGE" in msg_types
        assert "APPROVED_AFTER_REVISION" in msg_types

    def test_on_phase_callback(self):
        """The on_phase callback is called for each pipeline phase."""
        state = AgentState(query="test")
        coord = Coordinator(verbose=False)
        phases = []

        def on_phase(phase, detail=""):
            phases.append(phase)

        # Mock all pipeline steps to avoid real processing
        with patch("src.community_crew._collect_stateful") as mock_collect, \
             patch("src.community_crew._analyse_stateful") as mock_analyse, \
             patch("src.community_crew._generate_nudges_stateful") as mock_nudge:

            def fake_collect(s):
                s.posts = [{"id": "1"}]
                return s
            def fake_analyse(s, output_dir=None):
                s.clusters = [{"topic": "x"}]
                return s
            def fake_nudge(s):
                return s

            mock_collect.side_effect = fake_collect
            mock_analyse.side_effect = fake_analyse
            mock_nudge.side_effect = fake_nudge

            with patch("src.ethics_agent.EthicsAgent"):
                state = coord.run(state, on_phase=on_phase)

        assert "collect" in phases
        assert "analyse" in phases
        assert "nudge" in phases
        assert "ethics" in phases
        assert "persist" in phases

    def test_max_nudges_parameter(self):
        """AgentState.max_nudges controls how many clusters get nudges."""
        from src.community_crew import _generate_nudges_stateful

        state = AgentState(query="test", max_nudges=2)
        state.clusters = [
            {"topic_label": "a", "subcluster_id": 0, "n_posts": 5,
             "avg_readiness": 0.5,
             "top_posts": [{"id": "1", "text_clean": "post a", "readiness_score": 0.5}]},
            {"topic_label": "b", "subcluster_id": 0, "n_posts": 3,
             "avg_readiness": 0.45,
             "top_posts": [{"id": "2", "text_clean": "post b", "readiness_score": 0.45}]},
            {"topic_label": "c", "subcluster_id": 0, "n_posts": 2,
             "avg_readiness": 0.42,
             "top_posts": [{"id": "3", "text_clean": "post c", "readiness_score": 0.42}]},
        ]

        mock_action = MagicMock()
        mock_action.generate_community_nudge.return_value = {
            "nudge_text": "Test nudge",
            "explanation": "Test",
            "cluster_context": {},
            "target_user_ids": ["1"],
            "n_eligible": 1,
        }

        with patch("src.action_agent.ActionAgent", return_value=mock_action):
            state = _generate_nudges_stateful(state)

        # max_nudges=2 → nur 2 Nudges, obwohl 3 Cluster verfügbar
        assert len(state.nudges) == 2
        assert mock_action.generate_community_nudge.call_count == 2

    def test_max_nudges_5_generates_more(self):
        """max_nudges=5 generates up to 5 nudges if enough clusters exist."""
        from src.community_crew import _generate_nudges_stateful

        state = AgentState(query="test", max_nudges=5)
        state.clusters = [
            {"topic_label": f"topic{i}", "subcluster_id": 0, "n_posts": 3,
             "avg_readiness": 0.5 - i * 0.02,
             "top_posts": [{"id": str(i), "text_clean": f"post {i}", "readiness_score": 0.5 - i * 0.02}]}
            for i in range(6)
        ]

        mock_action = MagicMock()
        mock_action.generate_community_nudge.return_value = {
            "nudge_text": "Nudge", "explanation": "Exp",
            "cluster_context": {}, "target_user_ids": ["1"], "n_eligible": 1,
        }

        with patch("src.action_agent.ActionAgent", return_value=mock_action):
            state = _generate_nudges_stateful(state)

        assert len(state.nudges) == 5
        assert mock_action.generate_community_nudge.call_count == 5
