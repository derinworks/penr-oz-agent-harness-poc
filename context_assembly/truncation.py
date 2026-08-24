"""Non-destructive elision, and the report types for what got cut.

Two distinct operations happen under budget overflow (see
`ContextAssembler.assemble`):

- A RETRIEVABLE item can be dropped entirely. This is not considered
  "destroying meaning" in the sense the design guards against: by
  definition a retrievable item is something the harness can fetch again on
  a later turn (that's the whole point of "retrieved on demand" rather than
  "always kept"). Only its presence in *this* turn's context is cut, and
  that fact is recorded in a `DroppedItem` so the caller can see it
  happened and why.
- A KEPT item is never dropped. If it must shrink to fit the budget, it is
  *elided*: a contiguous middle section is replaced with a visible marker
  that states how much was cut, while the head and tail survive. This is
  the "summarize/elide with a visible marker" strategy -- lossy, but never
  silent. It is recorded in an `ElidedItem`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .tokens import TokenCounter


@dataclass(frozen=True)
class DroppedItem:
    """Record of a retrievable item cut entirely from this turn's context."""

    item_id: str
    kind: str
    priority: int
    tokens: int
    reason: str


@dataclass(frozen=True)
class ElidedItem:
    """Record of a kept item shortened (with a visible marker) to fit budget."""

    item_id: str
    kind: str
    original_tokens: int
    kept_tokens: int
    reason: str


def elide_text(
    text: str, target_tokens: int, token_counter: TokenCounter
) -> Tuple[str, bool]:
    """Shrink `text` to at most `target_tokens`, if needed, with a visible marker.

    Keeps a prefix and a suffix of `text` and replaces the middle with a
    marker naming how many characters were omitted, e.g.:
    ``"...head text ...[elided 812 chars]... tail text..."``.

    Returns `(result_text, was_elided)`. If `text` already fits within
    `target_tokens`, returns `(text, False)` unchanged -- elision only ever
    happens when it is actually needed.

    Uses a binary search over how many characters to keep (rather than a
    fixed char/token ratio) so this works with any `token_counter`,
    including a real tokenizer where the chars-per-token rate varies.
    """
    if token_counter(text) <= target_tokens:
        return text, False
    if target_tokens <= 0:
        return _render_elided(text, 0), True

    lo, hi = 0, len(text)
    best_keep = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _render_elided(text, mid)
        if token_counter(candidate) <= target_tokens:
            best_keep = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return _render_elided(text, best_keep), True


def _render_elided(text: str, keep_len: int) -> str:
    """Render `text` keeping `keep_len` characters split across head/tail."""
    if keep_len >= len(text):
        return text
    keep_len = max(0, keep_len)
    head_len = (keep_len * 2) // 3
    tail_len = keep_len - head_len
    omitted = len(text) - head_len - tail_len
    marker = f" …[elided {omitted} chars]… "
    head = text[:head_len]
    tail = text[len(text) - tail_len :] if tail_len else ""
    return f"{head}{marker}{tail}"
