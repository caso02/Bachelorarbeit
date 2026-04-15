"""Tests for src/universal_collector.py — text cleaning, anonymization, Kaggle loader, regex safety."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.universal_collector import UniversalCollector


@pytest.fixture
def collector():
    return UniversalCollector()


# ---------------------------------------------------------------------------
# anonymize_author
# ---------------------------------------------------------------------------

class TestAnonymizeAuthor:
    def test_deterministic(self, collector):
        """Same input always produces same hash."""
        h1 = collector.anonymize_author("testuser")
        h2 = collector.anonymize_author("testuser")
        assert h1 == h2

    def test_length_16(self, collector):
        result = collector.anonymize_author("someuser")
        assert len(result) == 16

    def test_hex_chars_only(self, collector):
        result = collector.anonymize_author("user123")
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_users_different_hashes(self, collector):
        h1 = collector.anonymize_author("alice")
        h2 = collector.anonymize_author("bob")
        assert h1 != h2

    def test_deleted_returns_empty(self, collector):
        assert collector.anonymize_author("[deleted]") == ""

    def test_automoderator_returns_empty(self, collector):
        assert collector.anonymize_author("AutoModerator") == ""

    def test_empty_returns_empty(self, collector):
        assert collector.anonymize_author("") == ""

    def test_none_returns_empty(self, collector):
        assert collector.anonymize_author("None") == ""


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_removes_urls(self, collector):
        text = "Check this out https://example.com and www.test.org done"
        result = collector._clean_text(text)
        assert "https://" not in result
        assert "www." not in result
        assert "Check this out" in result

    def test_normalizes_whitespace(self, collector):
        text = "  too   many    spaces  "
        result = collector._clean_text(text)
        assert result == "too many spaces"

    def test_removes_control_characters(self, collector):
        text = "hello\x00world\x1f!"
        result = collector._clean_text(text)
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_preserves_normal_text(self, collector):
        text = "This is a perfectly normal sentence about health."
        result = collector._clean_text(text)
        assert result == text


# ---------------------------------------------------------------------------
# Regex injection safety (M1 fix)
# ---------------------------------------------------------------------------

class TestRegexSafety:
    def test_keyword_with_regex_metachar_does_not_crash(self, collector):
        """str.contains with regex=False should handle metacharacters safely."""
        # This would crash with regex=True: unbalanced bracket
        posts = collector._fallback_mock("[invalid regex")
        assert isinstance(posts, list)

    def test_keyword_with_parentheses(self, collector):
        posts = collector._fallback_mock("test()")
        assert isinstance(posts, list)


# ---------------------------------------------------------------------------
# Kaggle Multi-CSV Loader
# ---------------------------------------------------------------------------

def _make_csv(path: Path, data: dict) -> Path:
    """Helper: Write a dict-of-lists to a CSV file."""
    pd.DataFrame(data).to_csv(path, index=False)
    return path


class TestKaggleMultiCSV:
    """Tests for _try_kaggle with multi-CSV support."""

    def test_single_csv_basic_columns(self, collector):
        """Single CSV with body/subreddit/score/id columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "test.csv", {
                "id": ["1", "2", "3"],
                "body": ["Hello world", "Tech is great", "Health tips"],
                "subreddit": ["general", "tech", "health"],
                "score": [10, 20, 30],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100)

            assert posts is not None
            assert len(posts) == 3
            assert posts[0]["text_clean"] == "Hello world"
            assert posts[1]["category"] == "tech"
            assert posts[2]["score"] == 30

    def test_multiple_csvs_concatenated(self, collector):
        """Multiple CSVs are loaded and concatenated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "askreddit.csv", {
                "id": ["a1", "a2"],
                "body": ["Question one", "Question two"],
                "subreddit": ["askreddit", "askreddit"],
                "score": [100, 200],
            })
            _make_csv(kaggle_dir / "fitness.csv", {
                "id": ["f1", "f2"],
                "body": ["Running tips", "Lifting heavy"],
                "subreddit": ["fitness", "fitness"],
                "score": [50, 60],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100)

            assert posts is not None
            assert len(posts) == 4
            categories = {p["category"] for p in posts}
            assert "askreddit" in categories
            assert "fitness" in categories

    def test_keyword_filter(self, collector):
        """Query parameter filters posts by text content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "data.csv", {
                "id": ["1", "2", "3"],
                "body": ["I love running", "Python programming", "Running marathon"],
                "subreddit": ["fitness", "tech", "fitness"],
                "score": [10, 20, 30],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("running", limit=100)

            assert posts is not None
            assert len(posts) == 2
            assert all("running" in p["text_clean"].lower() for p in posts)

    def test_nsfw_and_bot_filter(self, collector):
        """Posts with is_nsfw=True or is_bot=True are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "data.csv", {
                "id": ["1", "2", "3", "4"],
                "body": ["Normal post", "NSFW post", "Bot post", "Clean post"],
                "subreddit": ["test"] * 4,
                "score": [10, 20, 30, 40],
                "is_nsfw": [False, True, False, False],
                "is_bot": [False, False, True, False],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100)

            assert posts is not None
            assert len(posts) == 2
            texts = {p["text_clean"] for p in posts}
            assert "Normal post" in texts
            assert "Clean post" in texts
            assert "NSFW post" not in texts
            assert "Bot post" not in texts

    def test_limit_sampling(self, collector):
        """When more posts than limit, sample down."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "data.csv", {
                "id": [str(i) for i in range(100)],
                "body": [f"Post number {i}" for i in range(100)],
                "subreddit": ["test"] * 100,
                "score": list(range(100)),
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=10)

            assert posts is not None
            assert len(posts) == 10

    def test_title_body_merge(self, collector):
        """When text_col is body and title is also present, merge them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "data.csv", {
                "id": ["1"],
                "title": ["My Title"],
                "body": ["My body text"],
                "subreddit": ["test"],
                "score": [10],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100)

            assert posts is not None
            assert "My Title" in posts[0]["text_clean"]
            assert "My body text" in posts[0]["text_clean"]

    def test_empty_body_excluded(self, collector):
        """Posts with empty body are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "data.csv", {
                "id": ["1", "2", "3"],
                "body": ["Good post", "", "  "],
                "subreddit": ["test"] * 3,
                "score": [10, 20, 30],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100)

            assert posts is not None
            assert len(posts) == 1
            assert posts[0]["text_clean"] == "Good post"

    def test_no_csv_files_returns_none(self, collector):
        """Empty kaggle directory returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100)

            assert posts is None

    def test_missing_dir_returns_none(self, collector):
        """Non-existent kaggle directory returns None."""
        with patch("src.universal_collector.KAGGLE_DIR", Path("/nonexistent/path")):
            posts = collector._try_kaggle("", limit=100)

        assert posts is None

    def test_author_anonymization(self, collector):
        """Authors are anonymized with SHA-256."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "data.csv", {
                "id": ["1", "2"],
                "body": ["Post A", "Post B"],
                "subreddit": ["test"] * 2,
                "author": ["alice", "[deleted]"],
                "score": [10, 20],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100)

            assert posts is not None
            assert len(posts[0]["author_hash"]) == 16  # SHA-256 hex
            assert posts[1]["author_hash"] == ""  # [deleted]


# ---------------------------------------------------------------------------
# Subreddit-Filter
# ---------------------------------------------------------------------------

class TestSubredditFilter:
    """Tests for subreddit filtering in _try_kaggle."""

    def test_filter_single_subreddit_by_filename(self, collector):
        """Only loads CSVs matching the requested subreddit name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "tech.csv", {
                "id": ["t1", "t2"],
                "body": ["Tech post 1", "Tech post 2"],
                "subreddit": ["tech"] * 2,
                "score": [10, 20],
            })
            _make_csv(kaggle_dir / "art.csv", {
                "id": ["a1"],
                "body": ["Art post"],
                "subreddit": ["art"],
                "score": [30],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100, subreddits=["tech"])

            assert posts is not None
            assert len(posts) == 2
            assert all(p["category"] == "tech" for p in posts)

    def test_filter_multiple_subreddits(self, collector):
        """Multiple subreddits loads matching CSVs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "tech.csv", {
                "id": ["t1"],
                "body": ["Tech post"],
                "subreddit": ["tech"],
                "score": [10],
            })
            _make_csv(kaggle_dir / "art.csv", {
                "id": ["a1"],
                "body": ["Art post"],
                "subreddit": ["art"],
                "score": [20],
            })
            _make_csv(kaggle_dir / "health.csv", {
                "id": ["h1"],
                "body": ["Health post"],
                "subreddit": ["health"],
                "score": [30],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100, subreddits=["tech", "art"])

            assert posts is not None
            assert len(posts) == 2
            cats = {p["category"] for p in posts}
            assert cats == {"tech", "art"}

    def test_filter_case_insensitive(self, collector):
        """Subreddit filter is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "tech.csv", {
                "id": ["t1"],
                "body": ["Tech post"],
                "subreddit": ["Tech"],
                "score": [10],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100, subreddits=["TECH"])

            assert posts is not None
            assert len(posts) == 1

    def test_filter_no_match_returns_none(self, collector):
        """Non-existent subreddit filter returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "tech.csv", {
                "id": ["t1"],
                "body": ["Tech post"],
                "subreddit": ["tech"],
                "score": [10],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100, subreddits=["nonexistent"])

            assert posts is None

    def test_collect_passes_subreddits(self, collector):
        """The collect() method passes subreddits to _try_kaggle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "art.csv", {
                "id": ["a1", "a2"],
                "body": ["Art piece 1", "Art piece 2"],
                "subreddit": ["art"] * 2,
                "score": [10, 20],
            })
            _make_csv(kaggle_dir / "tech.csv", {
                "id": ["t1"],
                "body": ["Tech post"],
                "subreddit": ["tech"],
                "score": [30],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector.collect(query="", limit=100, subreddits=["art"])

            assert len(posts) == 2
            assert all(p["category"] == "art" for p in posts)

    def test_empty_body_uses_title_fallback(self, collector):
        """When body column is empty but title exists, title is used as text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "art.csv", {
                "id": ["a1", "a2", "a3"],
                "title": ["Oil painting, me, 2024", "Pixel art castle", "Sunset watercolor"],
                "body": ["", "", ""],
                "subreddit": ["art"] * 3,
                "score": [100, 50, 200],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100, subreddits=["art"])

            assert posts is not None
            assert len(posts) == 3
            assert "Oil painting" in posts[0]["text_clean"]
            assert all(p["category"] == "art" for p in posts)

    def test_all_columns_empty_returns_none(self, collector):
        """When both body and title are empty, returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            _make_csv(kaggle_dir / "empty.csv", {
                "id": ["1", "2"],
                "title": ["", ""],
                "body": ["", ""],
                "subreddit": ["test"] * 2,
                "score": [10, 20],
            })

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector._try_kaggle("", limit=100)

            assert posts is None

    def test_collect_no_mock_fallback_with_subreddits(self, collector):
        """When subreddits are explicitly set and kaggle fails, no mock fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle_dir = Path(tmpdir)
            # No matching CSV for 'nonexistent'

            with patch("src.universal_collector.KAGGLE_DIR", kaggle_dir):
                posts = collector.collect(query="", limit=100, subreddits=["nonexistent"])

            assert posts == []  # Empty list, NOT mock data
