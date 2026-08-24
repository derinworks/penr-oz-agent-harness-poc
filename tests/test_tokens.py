from context_assembly import approximate_token_count


def test_empty_string_is_zero_tokens():
    assert approximate_token_count("") == 0


def test_short_nonempty_string_is_at_least_one_token():
    assert approximate_token_count("hi") >= 1


def test_monotonic_non_decreasing_with_length():
    lengths = [approximate_token_count("x" * n) for n in range(0, 200, 7)]
    assert lengths == sorted(lengths)


def test_roughly_four_chars_per_token():
    text = "x" * 400
    tokens = approximate_token_count(text)
    assert 90 <= tokens <= 110
