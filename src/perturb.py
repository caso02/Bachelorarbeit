"""Mild, realistic text perturbations for stability testing."""

from __future__ import annotations

import random
import re


_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)

_PUNCT_SWAPS = {
    "!": ".",
    ".": "!",
    ",": " ",
    ";": ",",
    ":": " –",
    "–": ":",
}


def perturb_text(text: str, prob: float = 0.15, rng: random.Random | None = None) -> str:
    """Apply mild, non-destructive perturbations to a social-media text.

    Each perturbation fires independently with probability ``prob``:
        - Emoji removal or duplication
        - Minor punctuation swap
        - Extra/removed whitespace

    The text meaning is preserved; no synonym replacement is used.

    Args:
        text: Original cleaned text.
        prob: Per-perturbation trigger probability.
        rng: Optional seeded Random instance for reproducibility.
    """
    if rng is None:
        rng = random.Random()

    if not text:
        return text

    # 1) Emoji: remove or duplicate
    emojis = _EMOJI_PATTERN.findall(text)
    if emojis and rng.random() < prob:
        emoji = rng.choice(emojis)
        if rng.random() < 0.5:
            text = text.replace(emoji, "", 1)
        else:
            text = text.replace(emoji, emoji + emoji, 1)

    # 2) Punctuation swap (one random occurrence)
    if rng.random() < prob:
        positions = [(i, ch) for i, ch in enumerate(text) if ch in _PUNCT_SWAPS]
        if positions:
            idx, ch = rng.choice(positions)
            text = text[:idx] + _PUNCT_SWAPS[ch] + text[idx + 1:]

    # 3) Whitespace: insert double space or collapse one
    if rng.random() < prob:
        space_positions = [i for i, ch in enumerate(text) if ch == " "]
        if space_positions:
            idx = rng.choice(space_positions)
            if rng.random() < 0.5:
                text = text[:idx] + "  " + text[idx + 1:]
            else:
                text = text[:idx] + text[idx + 1:]

    return text
