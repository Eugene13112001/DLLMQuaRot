from blockkv.run_longproc import answer_limit, main_score


def test_answer_limits_follow_longproc():
    assert answer_limit("path_traversal_8k") == 9216
    assert answer_limit("html_to_tsv_2k") == 3072
    assert answer_limit("countdown_0.5k") == 1024


def test_main_score_prefers_f1_then_partial_then_accuracy():
    assert main_score({"f1": 0.5, "accuracy": 1.0}) == 0.5
    assert main_score({"accuracy": 0.0, "partial_accuracy": 0.4}) == 0.4
    assert main_score({"accuracy": 1.0}) == 1.0
