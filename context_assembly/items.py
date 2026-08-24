"""Data types shared across the context-assembly module.

A `ContextItem` is a candidate for inclusion in a model's context window: it
carries just enough metadata (a `kind`, a `priority`, and an optional
explicit `retention` pin) for a `RetentionPolicy` to classify it and for a
`ContextAssembler` to decide what survives a budget. See
`context_assembly.policy` and `context_assembly.assembler`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class Retention(str, Enum):
    """How a context item is treated when context is assembled.

    KEPT items are always included in the assembled context (subject only to
    non-destructive elision under extreme budget pressure -- see
    `context_assembly.truncation`). RETRIEVABLE items are only included when
    a retriever decides they are relevant, and are the first thing dropped
    when the assembled context does not fit the budget.
    """

    KEPT = "kept"
    RETRIEVABLE = "retrievable"


@dataclass(frozen=True)
class ContextItem:
    """A single candidate piece of context, before classification.

    Attributes:
        id: Stable, unique identifier for this item (used for reporting
            what was dropped/elided).
        kind: A short tag such as "system_prompt", "task_state", "history",
            or "docs". `RetentionPolicy` classifies items primarily by kind.
        text: The actual content.
        priority: Higher means more important. Used to decide which
            retrievable items are dropped first, and which kept items are
            elided first, under overflow. Default 0.
        retention: Optional explicit override. When set, it wins over any
            kind-based or rule-based classification -- a call site can pin
            a specific item as always-kept or always-retrievable regardless
            of its kind.
        metadata: Free-form, unused by this module; a convenience for
            callers (e.g. source path, timestamp) and for custom
            `RetentionPolicy` rules to inspect.
    """

    id: str
    kind: str
    text: str
    priority: int = 0
    retention: Optional[Retention] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedItem:
    """A `ContextItem` after classification and (possibly) elision.

    This is what actually ends up in an `AssemblyResult`. `text` may be a
    shortened, marker-bearing version of the original `ContextItem.text` if
    `elided` is True; it is never silently shortened without `elided` being
    set and without a corresponding `ElidedItem` report entry.
    """

    id: str
    kind: str
    retention: Retention
    text: str
    tokens: int
    priority: int
    elided: bool = False
