import pytest

from context_assembly import (
    ContextAssembler,
    ContextItem,
    Retention,
    RetentionPolicy,
    approximate_token_count,
)


def item(id_, kind, text="x", priority=0, retention=None):
    return ContextItem(id=id_, kind=kind, text=text, priority=priority, retention=retention)


def test_empty_candidates_with_zero_budget_is_a_no_op():
    assembler = ContextAssembler()
    result = assembler.assemble([], budget=0)

    assert result.included == []
    assert result.dropped == []
    assert result.elided == []
    assert result.total_tokens == 0
    assert result.pre_truncation_tokens == 0
    assert result.truncated is False
    assert result.unresolved_overflow is False


def test_everything_fits_nothing_is_touched():
    assembler = ContextAssembler()
    candidates = [
        item("sys", "system_prompt", text="You are an agent."),
        item("h1", "history", text="earlier turn"),
    ]
    result = assembler.assemble(candidates, budget=10_000)

    assert [i.id for i in result.included] == ["sys", "h1"]
    assert all(not i.elided for i in result.included)
    assert result.dropped == []
    assert result.elided == []
    assert result.truncated is False
    assert result.unresolved_overflow is False
    assert result.total_tokens == result.pre_truncation_tokens


def test_kept_items_come_before_retrieved_items_in_output_order():
    assembler = ContextAssembler()
    candidates = [
        item("h1", "history", text="a"),
        item("sys", "system_prompt", text="b"),
        item("h2", "history", text="c"),
        item("task", "task_state", text="d"),
    ]
    result = assembler.assemble(candidates, budget=10_000)
    ids = [i.id for i in result.included]
    # both kept items appear before both retrieved items, each group in
    # its original relative order
    assert ids.index("sys") < ids.index("h1")
    assert ids.index("task") < ids.index("h2")
    assert ids.index("h1") < ids.index("h2")


def test_overflow_drops_lowest_priority_retrievable_item_first():
    assembler = ContextAssembler()
    long_text = "word " * 200  # large enough to force overflow
    candidates = [
        item("sys", "system_prompt", text="system", priority=100),
        item("low", "history", text=long_text, priority=0),
        item("high", "history", text="short and important", priority=10),
    ]
    budget = approximate_token_count("system") + approximate_token_count(
        "short and important"
    ) + 2

    result = assembler.assemble(candidates, budget=budget)

    dropped_ids = [d.item_id for d in result.dropped]
    assert "low" in dropped_ids
    assert "high" not in dropped_ids
    included_ids = [i.id for i in result.included]
    assert "sys" in included_ids
    assert "high" in included_ids
    assert "low" not in included_ids
    # the kept item's text must be completely untouched
    sys_item = next(i for i in result.included if i.id == "sys")
    assert sys_item.text == "system"
    assert sys_item.elided is False
    assert result.truncated is True


def test_kept_items_are_never_dropped_only_elided_under_extreme_overflow():
    assembler = ContextAssembler()
    huge_system_prompt = "critical instructions " * 500
    candidates = [item("sys", "system_prompt", text=huge_system_prompt, priority=0)]

    result = assembler.assemble(candidates, budget=20)

    assert len(result.included) == 1
    assert result.included[0].id == "sys"
    assert result.included[0].elided is True
    assert "elided" in result.included[0].text
    assert len(result.dropped) == 0
    assert len(result.elided) == 1
    assert result.elided[0].item_id == "sys"


def test_single_retrievable_item_exceeding_budget_alone_is_dropped_not_elided():
    assembler = ContextAssembler()
    huge_doc = "reference material " * 500
    candidates = [item("doc", "docs", text=huge_doc, priority=0)]

    result = assembler.assemble(candidates, budget=10)

    assert result.included == []
    assert len(result.dropped) == 1
    assert result.dropped[0].item_id == "doc"
    assert result.elided == []
    assert result.unresolved_overflow is False


def test_single_kept_item_exceeding_budget_alone_is_elided_and_present():
    assembler = ContextAssembler()
    huge_prompt = "you must follow these rules exactly: " * 100
    candidates = [item("sys", "system_prompt", text=huge_prompt)]

    result = assembler.assemble(candidates, budget=50)

    assert len(result.included) == 1
    assert result.included[0].id == "sys"
    assert result.included[0].elided is True
    assert result.included[0].tokens <= 60  # roughly within budget + overhead
    assert result.unresolved_overflow is False


def test_budget_too_small_even_for_elided_kept_item_reports_unresolved_overflow():
    assembler = ContextAssembler()
    candidates = [item("sys", "system_prompt", text="x" * 500)]

    result = assembler.assemble(candidates, budget=0)

    # still present (never dropped), just shrunk as far as possible
    assert len(result.included) == 1
    assert result.included[0].id == "sys"
    assert result.included[0].elided is True
    # the marker itself costs tokens, so a budget of 0 cannot truly be met
    assert result.unresolved_overflow is True


def test_retrievable_items_dropped_in_ascending_priority_then_original_order():
    assembler = ContextAssembler()
    candidates = [
        item("r1", "history", text="a" * 40, priority=1),
        item("r2", "history", text="a" * 40, priority=1),
        item("r3", "history", text="a" * 40, priority=5),
    ]
    # budget for exactly one of the three items
    one_item_tokens = approximate_token_count("a" * 40)
    result = assembler.assemble(candidates, budget=one_item_tokens)

    # r3 (higher priority) must survive; between the priority-1 ties,
    # r1 (came first) is dropped before r2
    included_ids = [i.id for i in result.included]
    assert included_ids == ["r3"]
    dropped_ids = [d.item_id for d in result.dropped]
    assert dropped_ids == ["r1", "r2"]


def test_custom_retriever_filters_which_retrievable_candidates_are_considered():
    seen = {}

    def only_docs(query, candidates):
        seen["query"] = query
        seen["candidates"] = [c.id for c in candidates]
        return [c for c in candidates if c.kind == "docs"]

    assembler = ContextAssembler(retriever=only_docs)
    candidates = [
        item("h1", "history", text="a"),
        item("d1", "docs", text="b"),
    ]
    result = assembler.assemble(candidates, budget=10_000, query="what docs matter")

    assert seen["query"] == "what docs matter"
    assert sorted(seen["candidates"]) == ["d1", "h1"]
    included_ids = [i.id for i in result.included]
    assert "d1" in included_ids
    assert "h1" not in included_ids  # filtered out by the retriever, not by truncation
    assert result.dropped == []  # never even considered, so not "dropped"


def test_default_retriever_passes_everything_through():
    assembler = ContextAssembler()
    candidates = [item("d1", "docs", text="a"), item("d2", "reference", text="b")]
    result = assembler.assemble(candidates, budget=10_000)
    included_ids = {i.id for i in result.included}
    assert included_ids == {"d1", "d2"}


def test_duplicate_ids_raise_value_error():
    assembler = ContextAssembler()
    candidates = [item("dup", "docs", text="a"), item("dup", "history", text="b")]
    with pytest.raises(ValueError):
        assembler.assemble(candidates, budget=1000)


def test_negative_budget_raises_value_error():
    assembler = ContextAssembler()
    with pytest.raises(ValueError):
        assembler.assemble([], budget=-1)


def test_explicit_retention_pin_is_honored_end_to_end():
    policy = RetentionPolicy()
    assembler = ContextAssembler(policy=policy)
    # a "docs" item explicitly pinned KEPT should survive even when it
    # would otherwise be the first thing dropped
    candidates = [
        item("pinned_doc", "docs", text="a" * 40, priority=0, retention=Retention.KEPT),
        item("other", "history", text="a" * 40, priority=100),
    ]
    one_item_tokens = approximate_token_count("a" * 40)
    result = assembler.assemble(candidates, budget=one_item_tokens)

    included_ids = [i.id for i in result.included]
    assert "pinned_doc" in included_ids
    # the pinned item is treated as kept, so it is never a drop candidate
    assert all(d.item_id != "pinned_doc" for d in result.dropped)


def test_render_text_joins_included_items():
    assembler = ContextAssembler()
    candidates = [item("sys", "system_prompt", text="A"), item("h1", "history", text="B")]
    result = assembler.assemble(candidates, budget=10_000)
    assert result.render_text(separator="|") == "A|B"


def test_render_report_mentions_dropped_and_elided_reasons():
    assembler = ContextAssembler()
    candidates = [
        item("sys", "system_prompt", text="x" * 500),
        item("doc", "docs", text="y" * 500, priority=0),
    ]
    result = assembler.assemble(candidates, budget=5)
    report = assembler.assemble(candidates, budget=5).render_report()
    assert "dropped" in report
    assert "doc" in report
    assert result.truncated is True
