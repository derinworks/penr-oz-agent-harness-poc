"""Token counting.

A real harness would count tokens with the target model's actual tokenizer
(e.g. tiktoken, a HuggingFace `AutoTokenizer`). Wiring up a specific
tokenizer is explicitly out of scope for this PoC -- what matters for the
context-assembly policy is that *some* monotonic, swappable size measure
exists. `approximate_token_count` is a stand-in; pass a real one via
`ContextAssembler(token_counter=...)` when a tokenizer is available.
"""
from __future__ import annotations

from typing import Callable

TokenCounter = Callable[[str], int]
"""Anything that maps text to an integer size estimate. Must be
non-decreasing in string length for the truncation search in
`context_assembly.truncation.elide_text` to behave sensibly -- true of a
real tokenizer's token count, and true of the approximation below."""

_CHARS_PER_TOKEN = 4


def approximate_token_count(text: str) -> int:
    """Estimate a token count at roughly 4 characters per token.

    This is a rough, language-model-agnostic heuristic (English prose
    averages a bit under 4 characters per GPT-style token), not a real
    tokenizer. It exists so the rest of this module has *something* to
    measure a budget against without taking a dependency on any one
    tokenizer library.
    """
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)
