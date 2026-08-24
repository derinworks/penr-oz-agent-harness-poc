"""`ContextAssembler`: turns candidate context items into a context that fits
a token budget, per an explicit `RetentionPolicy` and a non-destructive
truncation policy.

This module owns the *policy*, not retrieval or real token counting:

- Retrieval backend (embeddings, BM25, whatever) is out of scope. A
  `Retriever` is just a callable the assembler calls to decide which
  retrievable candidates are actually relevant to this turn; the default
  passes everything through unfiltered.
- Real tokenizers are out of scope. `token_counter` defaults to
  `approximate_token_count` and can be swapped for a real one.

See `docs/context-assembly.md` for the design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .items import ContextItem, RenderedItem, Retention
from .policy import RetentionPolicy
from .tokens import TokenCounter, approximate_token_count
from .truncation import DroppedItem, ElidedItem, elide_text

Retriever = Callable[[str, List[ContextItem]], List[ContextItem]]
"""Given a query string and the candidates classified as RETRIEVABLE,
return the subset (in any order) that should actually be fetched for this
turn. A real implementation would rank by relevance against an index; this
is deliberately just an interface -- see `default_retriever`."""


def default_retriever(query: str, candidates: List[ContextItem]) -> List[ContextItem]:
    """Pass every retrievable candidate through unfiltered.

    A stand-in for an actual retrieval backend (semantic search, BM25,
    ...), which is out of scope for this PoC. This default lets the
    keep-vs-retrieve *policy* and the truncation policy be exercised and
    tested without depending on one. Replace it by passing
    `ContextAssembler(retriever=...)`.
    """
    del query  # unused by the stub; kept for interface parity
    return list(candidates)


@dataclass(frozen=True)
class AssemblyResult:
    """The outcome of one `ContextAssembler.assemble` call.

    `included` holds every item that made it into the final context (kept
    items first, in their original order, then surviving retrieved items,
    in their original order) -- some possibly elided. `dropped` and
    `elided` make the truncation policy's decisions observable: nothing is
    cut without a corresponding entry here explaining what and why.
    """

    included: List[RenderedItem]
    dropped: List[DroppedItem]
    elided: List[ElidedItem]
    budget: int
    pre_truncation_tokens: int
    total_tokens: int
    truncated: bool
    unresolved_overflow: bool

    def render_text(self, separator: str = "\n\n---\n\n") -> str:
        """Join the included items' text into one prompt-ready string."""
        return separator.join(item.text for item in self.included)

    def render_report(self) -> str:
        """A human-readable summary of what was included/dropped/elided and why."""
        lines = [
            f"budget={self.budget} tokens "
            f"used={self.total_tokens} tokens "
            f"pre_truncation={self.pre_truncation_tokens} tokens"
        ]
        if self.dropped:
            lines.append("dropped (retrievable, cut before any kept item):")
            lines.extend(
                f"  - {d.item_id} (kind={d.kind}, priority={d.priority}, "
                f"{d.tokens} tok): {d.reason}"
                for d in self.dropped
            )
        if self.elided:
            lines.append("elided (kept, shortened with a visible marker):")
            lines.extend(
                f"  - {e.item_id} (kind={e.kind}): "
                f"{e.original_tokens} -> {e.kept_tokens} tok: {e.reason}"
                for e in self.elided
            )
        if self.unresolved_overflow:
            lines.append(
                "WARNING: budget still exceeded after dropping every retrievable "
                "item and eliding every kept item to its minimum."
            )
        return "\n".join(lines)


class ContextAssembler:
    """Assembles a token-budgeted context from candidate items.

    Pipeline for `assemble`:

    1. Classify each candidate as KEPT or RETRIEVABLE via `policy`.
    2. Call `retriever(query, retrievable_candidates)` to decide which
       retrievable items are actually relevant to this turn.
    3. If everything fits `budget`, return it all, untouched.
    4. Otherwise, drop RETRIEVABLE items -- lowest `priority` first, ties
       broken by original order -- until it fits, or none are left.
    5. If it still does not fit, elide KEPT items -- again lowest
       `priority` first -- shrinking each toward the budget with a visible
       marker. KEPT items are never dropped, only shrunk.

    Every drop and every elision is recorded on the returned
    `AssemblyResult` so the decision is inspectable, not silent.
    """

    def __init__(
        self,
        policy: Optional[RetentionPolicy] = None,
        token_counter: TokenCounter = approximate_token_count,
        retriever: Retriever = default_retriever,
    ) -> None:
        self.policy = policy or RetentionPolicy()
        self.token_counter = token_counter
        self.retriever = retriever

    def assemble(
        self, candidates: List[ContextItem], budget: int, query: str = ""
    ) -> AssemblyResult:
        """Assemble `candidates` into a context that fits within `budget` tokens."""
        if budget < 0:
            raise ValueError("budget must be >= 0")
        self._check_unique_ids(candidates)

        kept_candidates, retrievable_candidates = self.policy.classify_many(candidates)
        retrieved_candidates = self.retriever(query, retrievable_candidates)

        kept = [self._render(c, Retention.KEPT) for c in kept_candidates]
        retrieved = [self._render(c, Retention.RETRIEVABLE) for c in retrieved_candidates]

        pre_total = self._sum(kept) + self._sum(retrieved)

        dropped: List[DroppedItem] = []
        elided: List[ElidedItem] = []

        retrieved = self._drop_retrievable(kept, retrieved, budget, dropped)
        kept = self._elide_kept(kept, retrieved, budget, elided)

        final_total = self._sum(kept) + self._sum(retrieved)

        return AssemblyResult(
            included=kept + retrieved,
            dropped=dropped,
            elided=elided,
            budget=budget,
            pre_truncation_tokens=pre_total,
            total_tokens=final_total,
            truncated=bool(dropped or elided),
            unresolved_overflow=final_total > budget,
        )

    def _drop_retrievable(
        self,
        kept: List[RenderedItem],
        retrieved: List[RenderedItem],
        budget: int,
        dropped: List[DroppedItem],
    ) -> List[RenderedItem]:
        """Phase 1: cut lowest-priority retrievable items until it fits.

        Never inspects or touches `kept`. Sorting is stable, so among items
        of equal priority the ones that came first are cut first.
        """
        kept_tokens = self._sum(kept)
        by_priority = sorted(retrieved, key=lambda item: item.priority)

        running_total = self._sum(retrieved)
        cut_ids = set()
        for item in by_priority:
            if kept_tokens + running_total <= budget:
                break
            dropped.append(
                DroppedItem(
                    item_id=item.id,
                    kind=item.kind,
                    priority=item.priority,
                    tokens=item.tokens,
                    reason=(
                        "dropped to satisfy the token budget; retrievable items "
                        "are cut, lowest priority first, before any kept item "
                        "is touched"
                    ),
                )
            )
            cut_ids.add(item.id)
            running_total -= item.tokens

        return [item for item in retrieved if item.id not in cut_ids]

    def _elide_kept(
        self,
        kept: List[RenderedItem],
        retrieved: List[RenderedItem],
        budget: int,
        elided: List[ElidedItem],
    ) -> List[RenderedItem]:
        """Phase 2: shrink lowest-priority kept items, with a marker, if still over.

        Only runs once every retrievable item has already been dropped and
        the budget is still exceeded. Kept items are shrunk, never removed.
        """
        current_total = self._sum(kept) + self._sum(retrieved)
        over = current_total - budget
        if over <= 0 or not kept:
            return kept

        by_id = {item.id: item for item in kept}
        for item in sorted(kept, key=lambda item: item.priority):
            if over <= 0:
                break
            target_tokens = max(0, item.tokens - over)
            new_text, was_elided = elide_text(item.text, target_tokens, self.token_counter)
            if not was_elided:
                continue
            new_tokens = self.token_counter(new_text)
            elided.append(
                ElidedItem(
                    item_id=item.id,
                    kind=item.kind,
                    original_tokens=item.tokens,
                    kept_tokens=new_tokens,
                    reason=(
                        "elided to satisfy the token budget after every "
                        "retrievable item was already dropped; kept items are "
                        "shrunk with a visible marker, never removed"
                    ),
                )
            )
            over -= item.tokens - new_tokens
            by_id[item.id] = RenderedItem(
                id=item.id,
                kind=item.kind,
                retention=item.retention,
                text=new_text,
                tokens=new_tokens,
                priority=item.priority,
                elided=True,
            )

        return [by_id[item.id] for item in kept]

    def _render(self, item: ContextItem, retention: Retention) -> RenderedItem:
        return RenderedItem(
            id=item.id,
            kind=item.kind,
            retention=retention,
            text=item.text,
            tokens=self.token_counter(item.text),
            priority=item.priority,
            elided=False,
        )

    @staticmethod
    def _sum(items: List[RenderedItem]) -> int:
        return sum(item.tokens for item in items)

    @staticmethod
    def _check_unique_ids(candidates: List[ContextItem]) -> None:
        seen = set()
        for item in candidates:
            if item.id in seen:
                raise ValueError(f"duplicate context item id: {item.id!r}")
            seen.add(item.id)
