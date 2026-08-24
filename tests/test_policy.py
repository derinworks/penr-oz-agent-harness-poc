from context_assembly import ContextItem, Retention, RetentionPolicy


def item(id_, kind, priority=0, retention=None, text="x"):
    return ContextItem(id=id_, kind=kind, text=text, priority=priority, retention=retention)


def test_known_always_keep_kinds_are_kept():
    policy = RetentionPolicy()
    for kind in ("system_prompt", "task_state", "instructions", "goal"):
        assert policy.classify(item("a", kind)) is Retention.KEPT


def test_known_retrievable_kinds_are_retrievable():
    policy = RetentionPolicy()
    for kind in ("history", "long_history", "docs", "reference", "memory", "tool_output"):
        assert policy.classify(item("a", kind)) is Retention.RETRIEVABLE


def test_unknown_kind_defaults_to_retrievable():
    policy = RetentionPolicy()
    assert policy.classify(item("a", "some_new_kind")) is Retention.RETRIEVABLE


def test_default_retention_is_configurable():
    policy = RetentionPolicy(default_retention=Retention.KEPT)
    assert policy.classify(item("a", "unrecognized")) is Retention.KEPT


def test_explicit_item_retention_overrides_kind():
    policy = RetentionPolicy()
    pinned = item("a", "docs", retention=Retention.KEPT)
    assert policy.classify(pinned) is Retention.KEPT

    pinned_other_way = item("b", "system_prompt", retention=Retention.RETRIEVABLE)
    assert policy.classify(pinned_other_way) is Retention.RETRIEVABLE


def test_custom_rule_takes_precedence_over_kind_sets():
    def pin_urgent(candidate):
        if candidate.metadata.get("urgent"):
            return Retention.KEPT
        return None

    policy = RetentionPolicy(rules=[pin_urgent])
    urgent_doc = ContextItem(id="a", kind="docs", text="x", metadata={"urgent": True})
    normal_doc = ContextItem(id="b", kind="docs", text="x")

    assert policy.classify(urgent_doc) is Retention.KEPT
    assert policy.classify(normal_doc) is Retention.RETRIEVABLE


def test_explicit_item_retention_beats_custom_rule():
    def always_retrievable(_candidate):
        return Retention.RETRIEVABLE

    policy = RetentionPolicy(rules=[always_retrievable])
    pinned = item("a", "docs", retention=Retention.KEPT)
    assert policy.classify(pinned) is Retention.KEPT


def test_rules_tried_in_order_first_match_wins():
    def never_matches(_candidate):
        return None

    def matches_kept(_candidate):
        return Retention.KEPT

    def matches_retrievable(_candidate):
        return Retention.RETRIEVABLE

    policy = RetentionPolicy(rules=[never_matches, matches_kept, matches_retrievable])
    assert policy.classify(item("a", "docs")) is Retention.KEPT


def test_classify_many_splits_and_preserves_order():
    policy = RetentionPolicy()
    items = [
        item("sys", "system_prompt"),
        item("h1", "history"),
        item("task", "task_state"),
        item("h2", "history"),
    ]
    kept, retrievable = policy.classify_many(items)
    assert [i.id for i in kept] == ["sys", "task"]
    assert [i.id for i in retrievable] == ["h1", "h2"]


def test_classify_many_handles_empty_input():
    policy = RetentionPolicy()
    kept, retrievable = policy.classify_many([])
    assert kept == []
    assert retrievable == []
