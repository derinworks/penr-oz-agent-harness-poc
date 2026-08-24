from context_assembly import approximate_token_count, elide_text


def test_text_within_budget_is_returned_unchanged():
    text = "short text"
    result, was_elided = elide_text(text, target_tokens=1000, token_counter=approximate_token_count)
    assert result == text
    assert was_elided is False


def test_text_over_budget_is_shortened_with_visible_marker():
    text = "HEAD" * 50 + "MIDDLE" * 200 + "TAIL" * 50
    target = 20
    result, was_elided = elide_text(text, target_tokens=target, token_counter=approximate_token_count)

    assert was_elided is True
    assert result != text
    assert len(result) < len(text)
    assert "elided" in result
    assert approximate_token_count(result) <= target + 5  # marker overhead tolerance


def test_elided_text_preserves_head_and_tail_fragments():
    text = "BEGINNING-MARKER" + ("filler " * 500) + "ENDING-MARKER"
    result, was_elided = elide_text(text, target_tokens=30, token_counter=approximate_token_count)

    assert was_elided is True
    assert result.startswith("BEGINNING-MARKER") or "BEGINNING-MARKER" in result[:40]
    assert "ENDING-MARKER" in result[-40:] or result.endswith("ENDING-MARKER")


def test_target_tokens_zero_still_returns_a_marker_stub_not_empty():
    text = "some content that must be entirely elided"
    result, was_elided = elide_text(text, target_tokens=0, token_counter=approximate_token_count)

    assert was_elided is True
    assert "elided" in result


def test_empty_text_is_never_elided():
    result, was_elided = elide_text("", target_tokens=0, token_counter=approximate_token_count)
    assert result == ""
    assert was_elided is False
