"""The keep-vs-retrieve classification policy.

This is the "explicit policy for what's kept in-context vs. retrieved on
demand" called for by the design: a small, inspectable set of rules rather
than a judgment call buried in a prompt or made ad hoc at call sites.

Classification precedence, highest to lowest:

1. An explicit `ContextItem.retention` pin always wins.
2. Custom `rules`, tried in order; the first one to return a non-None
   `Retention` wins.
3. `kind` membership in `always_keep_kinds` / `retrievable_kinds`.
4. `default_retention` (conservatively RETRIEVABLE -- see below).

Why the default is RETRIEVABLE, not KEPT: always-kept context is a standing
tax on every single turn's budget, forever. Treating an unrecognized kind as
"only shown when relevant" is the safer failure mode -- it costs a possible
retrieval miss, not a permanently bloated context window. Anything that
truly must always be present (system prompt, active task state) should be
tagged with one of `always_keep_kinds`, or pinned explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Iterable, List, Optional, Tuple

from .items import ContextItem, Retention

RetentionRule = Callable[[ContextItem], Optional[Retention]]
"""A custom classification rule: given an item, return a `Retention` to
claim it, or None to defer to the next rule / the kind-based default."""

#: Kinds that represent the model's standing orientation for the turn:
#: who it is, what it is doing, and what state that task is currently in.
#: Losing any of these mid-turn would change the meaning of the turn, not
#: just its length -- so they are always kept.
DEFAULT_ALWAYS_KEEP_KINDS: FrozenSet[str] = frozenset(
    {"system_prompt", "task_state", "instructions", "goal"}
)

#: Kinds that are useful but reconstructable: they can be fetched again on
#: a later turn, so cutting them from *this* turn's context loses nothing
#: permanently.
DEFAULT_RETRIEVABLE_KINDS: FrozenSet[str] = frozenset(
    {"history", "long_history", "docs", "reference", "memory", "tool_output"}
)


@dataclass
class RetentionPolicy:
    """Classifies `ContextItem`s as `Retention.KEPT` or `Retention.RETRIEVABLE`."""

    always_keep_kinds: FrozenSet[str] = field(
        default_factory=lambda: DEFAULT_ALWAYS_KEEP_KINDS
    )
    retrievable_kinds: FrozenSet[str] = field(
        default_factory=lambda: DEFAULT_RETRIEVABLE_KINDS
    )
    rules: List[RetentionRule] = field(default_factory=list)
    default_retention: Retention = Retention.RETRIEVABLE

    def classify(self, item: ContextItem) -> Retention:
        """Return the `Retention` for a single item, per the precedence above."""
        if item.retention is not None:
            return item.retention

        for rule in self.rules:
            result = rule(item)
            if result is not None:
                return result

        if item.kind in self.always_keep_kinds:
            return Retention.KEPT
        if item.kind in self.retrievable_kinds:
            return Retention.RETRIEVABLE

        return self.default_retention

    def classify_many(
        self, items: Iterable[ContextItem]
    ) -> Tuple[List[ContextItem], List[ContextItem]]:
        """Split `items` into (kept, retrievable), preserving relative order."""
        kept: List[ContextItem] = []
        retrievable: List[ContextItem] = []
        for item in items:
            if self.classify(item) is Retention.KEPT:
                kept.append(item)
            else:
                retrievable.append(item)
        return kept, retrievable
