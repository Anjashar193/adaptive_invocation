try:
    from src.policy import PolicyParams, call_rate, decide, letters_in_word, replay, words_typed
except ImportError:  # run as a plain script from inside tests/
    from policy import PolicyParams, call_rate, decide, letters_in_word, replay, words_typed


def test_empty_text_never_calls():
    assert decide("") == []


def test_word_start_calls_next_word():
    assert decide("hello ") == ["next_word"]


def test_mid_word_waits_for_min_letters():
    assert decide("hello wo") == []          # 2 letters
    assert decide("hello wor") == ["partial_word"]  # 3 letters


def test_sentence_only_after_enough_words():
    ten_words = " ".join(["w"] * 9) + " "     # 9 words + space -> spaces + 1 = 10
    eleven = " ".join(["w"] * 10) + " "       # spaces + 1 = 11
    assert decide(ten_words) == ["next_word"]
    assert decide(eleven) == ["next_word", "sentence"]


def test_words_typed_counts_spaces_plus_one():
    assert words_typed("") == 1
    assert words_typed("a b ") == 3
    assert letters_in_word("a bcd") == 3
    assert letters_in_word("a ") == 0


def test_punctuation_and_digits_do_not_call_mid_word():
    assert decide("hello wor,") == []
    assert decide("room 101") == []


def test_params_change_thresholds():
    p = PolicyParams(min_letters=2, sentence_min_words=3)
    assert decide("a wo", p) == ["partial_word"]
    assert decide("a b ", p) == ["next_word", "sentence"]


def test_call_rate_is_between_zero_and_one_and_below_always_invoke():
    text = "Could you please recommend some good movies to watch tonight after work"
    r = call_rate(text)
    assert 0.0 < r < 1.0
    assert len(replay(text)) == len(text)