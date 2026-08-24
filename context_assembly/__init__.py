"""Context assembly: what stays in context, what's retrieved on demand, and
how overflow is truncated without destroying meaning.

Public API:

- `ContextItem`, `Retention`, `RenderedItem` -- the data types (`items.py`).
- `RetentionPolicy` -- the keep-vs-retrieve classification policy (`policy.py`).
- `ContextAssembler`, `AssemblyResult`, `Retriever`, `default_retriever` --
  the assembly pipeline and its result (`assembler.py`).
- `approximate_token_count`, `TokenCounter` -- the swappable size measure
  (`tokens.py`).
- `DroppedItem`, `ElidedItem`, `elide_text` -- the truncation policy and its
  report types (`truncation.py`).

See `docs/context-assembly.md` for the design write-up.
"""
from .assembler import AssemblyResult, ContextAssembler, Retriever, default_retriever
from .items import ContextItem, RenderedItem, Retention
from .policy import RetentionPolicy, RetentionRule
from .tokens import TokenCounter, approximate_token_count
from .truncation import DroppedItem, ElidedItem, elide_text

__all__ = [
    "AssemblyResult",
    "ContextAssembler",
    "Retriever",
    "default_retriever",
    "ContextItem",
    "RenderedItem",
    "Retention",
    "RetentionPolicy",
    "RetentionRule",
    "TokenCounter",
    "approximate_token_count",
    "DroppedItem",
    "ElidedItem",
    "elide_text",
]
