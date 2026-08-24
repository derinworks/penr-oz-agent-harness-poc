# Context assembly

Implements [issue #7](https://github.com/derinworks/penr-oz-agent-harness-poc/issues/7),
the first piece of the **Context and state** component described in the README: "an
explicit policy for what stays in context versus what is retrieved on demand, and how
to truncate on overflow without destroying meaning."

Code: [`context_assembly/`](../context_assembly/). Tests: [`tests/`](../tests/).

## The problem

A model call only sees what the harness puts in its context window. Two decisions
follow from that, and both are usually made implicitly (a growing prompt string, an
`if` buried in a loop) instead of being named:

1. **What is always there**, regardless of the current turn's content — the system
   prompt, the active task's state. Losing these mid-turn doesn't just shorten the
   context, it changes what the turn *means*.
2. **What is fetched only when relevant** — long history, reference docs, prior tool
   output. These are useful, but they are also *reconstructable*: the source still
   exists, and a future turn can go get it again.

And a third decision that harnesses get wrong constantly: **what happens when the
assembled context doesn't fit the budget.** The naive answer — chop the string at N
characters — is a silent, meaning-blind cut. It can sever a sentence mid-word, discard
the one instruction that mattered, and leave no trace that anything was lost. The
README calls this out by name as a harness failure: "silently truncates its context."

This module is the explicit policy for both decisions, as code, not prose.

## Keep vs. retrieve: `RetentionPolicy`

`RetentionPolicy.classify(item)` (in [`policy.py`](../context_assembly/policy.py))
returns `Retention.KEPT` or `Retention.RETRIEVABLE` for a `ContextItem`, by this
precedence:

1. An explicit `ContextItem.retention` pin, if the caller set one.
2. Custom `rules` (`Callable[[ContextItem], Optional[Retention]]`), tried in order —
   the first to return non-`None` wins. This is the escape hatch for anything kind
   alone can't decide (e.g. "keep this doc if it's tagged `pinned` in metadata").
3. `item.kind` membership in `always_keep_kinds` (default: `system_prompt`,
   `task_state`, `instructions`, `goal`) or `retrievable_kinds` (default: `history`,
   `long_history`, `docs`, `reference`, `memory`, `tool_output`).
4. `default_retention`, which defaults to `RETRIEVABLE`.

The default-retention choice is deliberate: an unrecognized kind is treated as
"fetch when relevant" rather than "always present." Always-kept context is a standing
tax paid on every future turn's budget; if a new kind of item genuinely needs to be
permanent, that should be a conscious addition to `always_keep_kinds` (or an explicit
per-item pin), not an accident of an unclassified default.

`classify_many` is the batch form other harness components actually call — it splits a
candidate list into `(kept, retrievable)`, preserving relative order.

## On-demand retrieval: a stub, on purpose

`ContextAssembler` calls a `Retriever` — `Callable[[str, list[ContextItem]],
list[ContextItem]]` — to decide which retrievable candidates are actually relevant to
the current turn (`assembler.py`). This is explicitly an interface, not an
implementation: **the retrieval backend (embeddings, BM25, recency ranking, whatever)
is out of scope for this issue.** `default_retriever` is a pass-through stub that
returns every retrievable candidate unfiltered, which is enough to exercise the
keep-vs-retrieve and truncation policies in isolation. A real retriever is a drop-in
replacement passed as `ContextAssembler(retriever=...)`; no other code changes.

## Truncation: non-destructive by construction

`ContextAssembler.assemble(candidates, budget, query="")` runs:

1. Classify candidates (`RetentionPolicy.classify_many`).
2. Retrieve (`retriever(query, retrievable_candidates)`).
3. If everything fits `budget`, return it as-is. No truncation logic runs at all.
4. Otherwise, **drop RETRIEVABLE items, lowest `priority` first** (ties broken by
   original order), until it fits or none are left.
5. If it *still* doesn't fit — every retrievable item is already gone — **elide KEPT
   items, lowest `priority` first**, shrinking each with a visible marker
   (`" …[elided N chars]… "`) that keeps a head and a tail fragment. Kept items are
   never removed, only shrunk.

Every cut is recorded, not just performed:

- A dropped retrievable item produces a `DroppedItem(item_id, kind, priority, tokens,
  reason)`.
- An elided kept item produces an `ElidedItem(item_id, kind, original_tokens,
  kept_tokens, reason)`.
- `AssemblyResult.truncated` says whether anything happened at all;
  `unresolved_overflow` says whether the budget still couldn't be met even after
  dropping everything droppable and eliding everything kept (i.e. the budget is
  smaller than the irreducible minimum — a marker costs tokens too).
- `AssemblyResult.render_report()` renders all of the above as text, for a trace log
  or a debugging session.

### Why this counts as non-destructive

- **Dropping a retrievable item does not destroy the information.** By definition,
  something classified `RETRIEVABLE` is available to fetch again on a later turn — that
  is the entire premise of "retrieved on demand" rather than "always kept." What's cut
  is this turn's *inclusion* of it, not the underlying source. And that cut is visible
  in `dropped`, so the caller (and anyone reading the trace) knows precisely what was
  left out and why, rather than discovering a gap by surprise.
- **A kept item is never silently shortened.** It only shrinks through `elide_text`
  (`truncation.py`), which always leaves a marker stating how many characters were
  removed, and always preserves a head and a tail rather than an arbitrary cut point.
  The marker is part of the text itself, so even a caller who never inspects
  `AssemblyResult.elided` still sees, in the prompt, that something was cut.
- **Priority order is respected in both directions.** Lower-priority retrievable items
  go before higher-priority ones; lower-priority kept items get elided before
  higher-priority ones. A caller who cares about ordering controls it with one field.
- **Nothing is touched unless the budget actually requires it.** Step 3 above is a
  fast exit — if it fits, the pipeline doesn't run truncation logic at all, so
  `pre_truncation_tokens == total_tokens` and both report lists are empty.

## Edge cases handled (see `tests/test_assembler.py`)

- Empty candidate list, budget zero: a no-op, not an error.
- A single retrievable item larger than the whole budget: dropped entirely (it's
  retrievable — nothing is lost that can't be fetched again), reported once.
- A single kept item larger than the whole budget: elided down to fit, still present,
  reported once.
- A budget smaller than even the irreducible marker overhead: the kept item is
  shrunk as far as it can go and `unresolved_overflow` is `True` — the caller is told
  the budget is unsatisfiable rather than being handed a result that silently doesn't
  fit.
- Duplicate item IDs: rejected with `ValueError` at the start of `assemble`, since IDs
  are how drops/elisions are reported and reconciled.

## Explicitly out of scope

- **A real retrieval backend.** `default_retriever` is a pass-through stub. Semantic
  search, BM25, recency weighting, and reranking are all future work behind the same
  `Retriever` interface.
- **A real tokenizer.** `approximate_token_count` (`tokens.py`) estimates ~4 characters
  per token — a common rough heuristic, not tied to any model's actual vocabulary.
  `ContextAssembler(token_counter=...)` accepts any `Callable[[str], int]`, so wiring in
  `tiktoken` or a HuggingFace tokenizer is a constructor argument, not a rewrite. The
  only contract truncation relies on is that the counter is non-decreasing in string
  length.
- **Summarization by another model call.** Elision here is mechanical (head/tail with a
  marker), not an LLM-generated summary. A future harness component could summarize a
  dropped/elided item's content and feed that back in as a new candidate — this module
  just needs to keep reporting drops/elisions in enough detail for that to be possible
  later.
- **Memory persistence, the execution loop, and verification gates** — separate,
  later issues per the README's component list. This module is a pure function of
  `(candidates, budget, query) -> AssemblyResult`; it holds no state across calls and
  makes no I/O calls of its own.
