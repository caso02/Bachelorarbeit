#!/usr/bin/env python3
"""Generate synthetic posts and ethics-stress nudges via Anthropic API.

Usage:
    python scripts/generate_synthetic_posts.py [--dry-run]

Outputs:
    synthetic_data/edge_cases.json
    synthetic_data/ethics_stress_test.json
    synthetic_data/generation_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SYNTH_DIR = PROJECT_ROOT / "synthetic_data"
PROMPTS_PATH = SYNTH_DIR / "generation_prompts.json"
LOG_PATH = SYNTH_DIR / "generation_log.jsonl"
EDGE_CASES_PATH = SYNTH_DIR / "edge_cases.json"
ETHICS_STRESS_PATH = SYNTH_DIR / "ethics_stress_test.json"

MODEL = "claude-sonnet-4-5-20250929"
TEMPERATURE = 0.8
MAX_TOKENS = 4096

CATEGORIES = ["tech", "health", "finance", "sports", "entertainment",
              "environment", "travel", "social"]

BOUNDARY_PAIRS = [
    ("health", "tech"),
    ("finance", "tech"),
    ("travel", "finance"),
    ("sports", "health"),
    ("entertainment", "tech"),
    ("environment", "social"),
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_client():
    """Create Anthropic client. Reads ANTHROPIC_API_KEY from env."""
    # Load .env if present
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    import anthropic
    return anthropic.Anthropic()


def call_with_retry(client, system: str, user: str, call_id: str,
                    prompt_id: str, max_retries: int = 5) -> list[dict]:
    """Call Anthropic API with exponential backoff. Returns parsed JSON list."""
    wait = 5.0
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            duration = time.time() - t0
            raw = response.content[0].text

            # Log the call
            log_entry = {
                "call_id": call_id,
                "model": response.model,
                "temperature": TEMPERATURE,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_id": prompt_id,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "duration_seconds": round(duration, 2),
                "status": "ok",
            }
            _append_log(log_entry)

            # Parse JSON
            parsed = _parse_json_array(raw)
            print(f"  ✓ {call_id}: {len(parsed)} items ({duration:.1f}s)")
            return parsed

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "529" in err_str or "overloaded" in err_str.lower()
            if is_rate_limit and attempt < max_retries:
                print(f"  ⏳ {call_id}: Rate limit, retry in {wait:.0f}s (attempt {attempt+1}/{max_retries})")
                _append_log({
                    "call_id": call_id, "model": MODEL, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_id": prompt_id, "status": "retry", "error": err_str[:200],
                    "attempt": attempt + 1,
                })
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                print(f"  ✗ {call_id}: FAILED — {err_str[:100]}")
                _append_log({
                    "call_id": call_id, "model": MODEL, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_id": prompt_id, "status": "error", "error": err_str[:500],
                })
                return []


def _parse_json_array(raw: str) -> list[dict]:
    """Extract a JSON array from model response."""
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    # Find array bounds
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        return json.loads(raw[start:end + 1])
    return json.loads(raw)


def _append_log(entry: dict):
    """Append a log entry to the JSONL log file."""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_irony_posts(client, prompts: dict) -> list[dict]:
    """Klasse A: 40 irony/sarcasm posts (5 per category)."""
    cfg = prompts["edge_irony"]
    all_posts = []
    for cat in CATEGORIES:
        user_msg = cfg["user_template"].format(n=5, category=cat)
        items = call_with_retry(
            client, cfg["system"], user_msg,
            call_id=f"gen_irony_{cat}",
            prompt_id="edge_irony",
        )
        for i, item in enumerate(items):
            all_posts.append({
                "id": f"synth_irony_{cat}_{i+1:02d}",
                "text": item.get("text", ""),
                "expected_category": "social",  # irony = social function
                "post_class": "irony",
                "edge_case_type": "irony",
                "surface_category": cat,
                "actual_intent": item.get("actual_intent", ""),
                "expected_routing_confidence": "low",
                "expected_readiness": "low",
                "generation_prompt_id": "edge_irony",
            })
        time.sleep(2)  # rate limit courtesy
    return all_posts


def generate_boundary_posts(client, prompts: dict) -> list[dict]:
    """Klasse B: 60 boundary posts (10 per pair)."""
    cfg = prompts["edge_boundary"]
    examples_map = prompts["boundary_examples"]
    all_posts = []
    for cat_a, cat_b in BOUNDARY_PAIRS:
        key = f"{cat_a}_{cat_b}"
        examples = examples_map.get(key, "various overlapping topics")
        user_msg = cfg["user_template"].format(
            n=10, cat_a=cat_a, cat_b=cat_b, examples=examples,
        )
        items = call_with_retry(
            client, cfg["system"], user_msg,
            call_id=f"gen_boundary_{key}",
            prompt_id="edge_boundary",
        )
        for i, item in enumerate(items):
            all_posts.append({
                "id": f"synth_boundary_{key}_{i+1:02d}",
                "text": item.get("text", ""),
                "expected_category": cat_a,
                "post_class": "boundary",
                "edge_case_type": f"boundary_{cat_a}_{cat_b}",
                "secondary_category": cat_b,
                "boundary_topic": item.get("boundary_topic", ""),
                "expected_routing_confidence": "low",
                "expected_readiness": "medium",
                "generation_prompt_id": "edge_boundary",
            })
        time.sleep(2)
    return all_posts


def generate_readiness_posts(client, prompts: dict) -> list[dict]:
    """Klasse C: 50 readiness validation posts (25 high + 25 low)."""
    all_posts = []

    # High readiness: ~3 per category (25 total across 8 cats)
    cfg_high = prompts["edge_readiness_high"]
    counts_high = [4, 3, 3, 3, 3, 3, 3, 3]  # = 25 total
    for cat, n in zip(CATEGORIES, counts_high):
        user_msg = cfg_high["user_template"].format(n=n, category=cat)
        items = call_with_retry(
            client, cfg_high["system"], user_msg,
            call_id=f"gen_readiness_high_{cat}",
            prompt_id="edge_readiness_high",
        )
        for i, item in enumerate(items):
            all_posts.append({
                "id": f"synth_readiness_high_{cat}_{i+1:02d}",
                "text": item.get("text", ""),
                "expected_category": cat,
                "post_class": "explicit_connection",
                "edge_case_type": "explicit_connection",
                "expected_routing_confidence": "high",
                "expected_readiness": "high",
                "generation_prompt_id": "edge_readiness_high",
            })
        time.sleep(1)

    # Low readiness: ~3 per category (25 total)
    cfg_low = prompts["edge_readiness_low"]
    counts_low = [4, 3, 3, 3, 3, 3, 3, 3]  # = 25 total
    for cat, n in zip(CATEGORIES, counts_low):
        user_msg = cfg_low["user_template"].format(n=n, category=cat)
        items = call_with_retry(
            client, cfg_low["system"], user_msg,
            call_id=f"gen_readiness_low_{cat}",
            prompt_id="edge_readiness_low",
        )
        for i, item in enumerate(items):
            all_posts.append({
                "id": f"synth_readiness_low_{cat}_{i+1:02d}",
                "text": item.get("text", ""),
                "expected_category": cat,
                "post_class": "no_connection",
                "edge_case_type": "no_connection",
                "expected_routing_confidence": "high",
                "expected_readiness": "low",
                "generation_prompt_id": "edge_readiness_low",
            })
        time.sleep(1)

    return all_posts


def generate_multi_theme_posts(client, prompts: dict) -> list[dict]:
    """Klasse E: 20 multi-theme posts."""
    cfg = prompts["edge_multi_theme"]
    cats_str = ", ".join(CATEGORIES)
    all_posts = []

    # 2 batches of 10
    for batch in range(2):
        user_msg = cfg["user_template"].format(n=10, categories=cats_str)
        items = call_with_retry(
            client, cfg["system"], user_msg,
            call_id=f"gen_multi_theme_{batch+1}",
            prompt_id="edge_multi_theme",
        )
        for i, item in enumerate(items):
            cats = item.get("categories", [])
            all_posts.append({
                "id": f"synth_multi_{batch*10+i+1:02d}",
                "text": item.get("text", ""),
                "expected_category": cats if isinstance(cats, list) else [cats],
                "post_class": "multi_theme",
                "edge_case_type": "multi_theme",
                "theme_description": item.get("theme_description", ""),
                "expected_routing_confidence": "low",
                "expected_readiness": "medium",
                "generation_prompt_id": "edge_multi_theme",
            })
        time.sleep(2)

    return all_posts


def generate_ethics_stress(client, prompts: dict) -> list[dict]:
    """Klasse D: 30 manipulative nudges for ethics stress test."""
    cfg = prompts["ethics_stress"]
    all_nudges = []

    for nudge_type, template_key, n in [
        ("fomo", "user_template_fomo", 10),
        ("pressure", "user_template_manipulative", 10),
        ("generic", "user_template_generic", 10),
    ]:
        user_msg = cfg[template_key].format(n=n)
        items = call_with_retry(
            client, cfg["system"], user_msg,
            call_id=f"gen_ethics_{nudge_type}",
            prompt_id=f"ethics_stress_{nudge_type}",
        )
        for i, item in enumerate(items):
            all_nudges.append({
                "id": f"stress_{nudge_type}_{i+1:02d}",
                "nudge_text": item.get("nudge_text", ""),
                "explanation": item.get("explanation", ""),
                "manipulation_type": nudge_type,
                "expected_decision": "REJECTED" if nudge_type != "generic" else "REVISE",
            })
        time.sleep(2)

    return all_nudges


# ---------------------------------------------------------------------------
# Quality check sample
# ---------------------------------------------------------------------------

def write_quality_check(edge_cases: list[dict]):
    """Write a markdown file with 40 sample posts for human review."""
    import random
    random.seed(42)

    samples = []
    # 5 per class
    by_class = {}
    for p in edge_cases:
        cls = p["post_class"]
        by_class.setdefault(cls, []).append(p)

    for cls, posts in sorted(by_class.items()):
        n = min(10, len(posts))
        samples.extend(random.sample(posts, n))

    lines = ["# Quality Check Sample — Synthetic Edge-Case Posts\n"]
    lines.append(f"Generated: {datetime.now().isoformat()[:10]}\n")
    lines.append(f"Total samples: {len(samples)}\n")

    current_class = None
    for p in sorted(samples, key=lambda x: x["post_class"]):
        if p["post_class"] != current_class:
            current_class = p["post_class"]
            lines.append(f"\n## Class: {current_class}\n")
        lines.append(f"### {p['id']}")
        lines.append(f"**Expected category**: {p.get('expected_category', 'N/A')}")
        if p.get("edge_case_type"):
            lines.append(f"**Edge case type**: {p['edge_case_type']}")
        lines.append(f"\n> {p.get('text', p.get('nudge_text', ''))}\n")

    path = SYNTH_DIR / "quality_check_sample.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📋 Quality check sample written to {path} ({len(samples)} posts)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation posts")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without API calls")
    args = parser.parse_args()

    print("=" * 60)
    print("Synthetic Post Generator — Anthropic API")
    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"Output: {SYNTH_DIR}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        print("  Klasse A (irony):      8 × 5 = 40 posts  (8 API calls)")
        print("  Klasse B (boundary):   6 × 10 = 60 posts (6 API calls)")
        print("  Klasse C (readiness):  16 × ~3 = 50 posts (16 API calls)")
        print("  Klasse E (multi):      2 × 10 = 20 posts (2 API calls)")
        print("  Klasse D (ethics):     3 × 10 = 30 nudges (3 API calls)")
        print(f"  Total: ~200 posts + 30 nudges = ~35 API calls")
        return

    # Clear log for fresh run
    LOG_PATH.write_text("")

    client = get_client()

    # --- Edge Cases ---
    print("\n▶ Klasse A: Ironie & Sarkasmus (40 posts)")
    irony = generate_irony_posts(client, json.loads(PROMPTS_PATH.read_text()))

    print("\n▶ Klasse B: Grenzgänger (60 posts)")
    boundary = generate_boundary_posts(client, json.loads(PROMPTS_PATH.read_text()))

    print("\n▶ Klasse C: Readiness-Validierung (50 posts)")
    readiness = generate_readiness_posts(client, json.loads(PROMPTS_PATH.read_text()))

    print("\n▶ Klasse E: Multi-Thema (20 posts)")
    multi = generate_multi_theme_posts(client, json.loads(PROMPTS_PATH.read_text()))

    edge_cases = irony + boundary + readiness + multi
    print(f"\n✓ Total edge cases: {len(edge_cases)}")

    # Add generation timestamp to all
    ts = datetime.now(timezone.utc).isoformat()
    for p in edge_cases:
        p["generation_timestamp"] = ts

    EDGE_CASES_PATH.write_text(
        json.dumps(edge_cases, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Saved to {EDGE_CASES_PATH}")

    # --- Ethics Stress Test ---
    print("\n▶ Klasse D: Ethics-Stress-Test (30 nudges)")
    prompts = json.loads(PROMPTS_PATH.read_text())
    nudges = generate_ethics_stress(client, prompts)
    for n in nudges:
        n["generation_timestamp"] = ts

    print(f"\n✓ Total nudges: {len(nudges)}")
    ETHICS_STRESS_PATH.write_text(
        json.dumps(nudges, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Saved to {ETHICS_STRESS_PATH}")

    # --- Quality Check ---
    write_quality_check(edge_cases)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print(f"  Edge cases:    {len(edge_cases)} posts")
    print(f"  Ethics stress: {len(nudges)} nudges")
    log_lines = LOG_PATH.read_text().strip().split("\n")
    ok_count = sum(1 for l in log_lines if '"status": "ok"' in l)
    err_count = sum(1 for l in log_lines if '"status": "error"' in l)
    print(f"  API calls:     {ok_count} ok, {err_count} errors")
    print("=" * 60)


if __name__ == "__main__":
    main()
