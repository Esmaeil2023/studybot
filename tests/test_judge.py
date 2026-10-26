import csv
import json

import pytest

from studybot import judge as j


def rec(qid, gold_answerable=True, answerable=True, text="Vier Fachsemester.", **kw):
    return {
        "qid": qid, "lang": qid.rsplit("-", 1)[1], "question": "Wie lange?",
        "gold_answerable": gold_answerable, "gold_answer": "Vier Fachsemester.",
        "answerable": answerable, "text": text, **kw,
    }  # fmt: skip


def llm(verdict="correct"):
    return lambda system, user: json.dumps({"verdict": verdict, "reason": "ok"})


def test_parse_verdict_accepts_json_and_rejects_noise():
    assert j.parse_verdict('{"verdict": "Partial", "reason": "x"}') == ("partial", "x")
    assert j.parse_verdict('text {"verdict": "correct", "reason": "y"} text')[0] == "correct"
    assert j.parse_verdict("")[0] is None
    assert j.parse_verdict('{"verdict": "great"}')[0] is None


def test_answered_question_is_graded_by_the_llm():
    out = j.judge_one(rec("f01-de"), llm("partial"))
    assert (out["verdict"], out["judged_by"]) == ("partial", "llm")


def test_prompt_contains_reference_and_answer_but_not_citations_rules_are_in_system():
    seen = []
    j.judge_one(rec("f01-de"), lambda s, u: seen.append((s, u)) or '{"verdict": "correct"}')
    system, user = seen[0]
    assert "Reference answer: Vier Fachsemester." in user and "System answer:" in user
    assert "correct" in system and "partial" in system


def test_rule_based_cases_make_no_llm_call():
    def boom(system, user):
        raise AssertionError("no call expected")

    assert j.judge_one(rec("f01-de", answerable=False), boom)["verdict"] == "wrong"
    assert (
        j.judge_one(rec("u01-de", gold_answerable=False, answerable=False), boom)["verdict"]
        == "correct"
    )
    assert (
        j.judge_one(rec("u11-de", gold_answerable=False, answerable=True), boom)["verdict"]
        == "review"
    )


def test_unreadable_judge_reply_is_retried_once_then_marked_unreadable():
    replies = iter(["", '{"verdict": "wrong", "reason": "r"}'])
    assert j.judge_one(rec("f01-de"), lambda s, u: next(replies))["verdict"] == "wrong"
    assert j.judge_one(rec("f01-de"), lambda s, u: "")["verdict"] == "unreadable"


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    items = [(f"p{i}", float(i % 2)) for i in range(40)]
    mean, low, high = j.bootstrap_ci(items)
    assert mean == 0.5 and low < mean < high
    assert j.bootstrap_ci(items) == (mean, low, high)


def test_bootstrap_resamples_pairs_not_single_questions():
    # two twins always agree: pairs are all-1 or all-0, so resampling can reach 0 and 1 widely
    twins = [(f"p{i}", float(i % 2)) for i in range(10) for _ in range(2)]
    singles = [(f"s{k}", float((k // 2) % 2)) for k in range(20)]
    _, low_t, high_t = j.bootstrap_ci(twins)
    _, low_s, high_s = j.bootstrap_ci(singles)
    assert high_t - low_t >= high_s - low_s


def test_bootstrap_of_nothing_is_nan():
    assert all(x != x for x in j.bootstrap_ci([]))


def test_cohen_kappa():
    same = ["correct", "wrong", "correct", "wrong"]
    assert j.cohen_kappa(same, same) == 1.0
    assert j.cohen_kappa(same, ["wrong", "correct", "wrong", "correct"]) == -1.0
    # always saying 'correct' agrees by chance only: kappa 0 despite 75% raw agreement
    human = ["correct", "correct", "correct", "wrong"]
    assert j.cohen_kappa(human, ["correct"] * 4) == 0.0
    assert j.cohen_kappa([], []) != j.cohen_kappa([], [])  # nan


def test_agreement_report_lists_disagreements():
    report = j.agreement_report(
        [("correct", "correct"), ("partial", "correct"), ("wrong", "wrong")]
    )
    assert report["n"] == 3 and report["agreement"] == pytest.approx(2 / 3)
    assert report["agreement_binary"] == pytest.approx(2 / 3)  # partial vs correct differ
    assert report["disagreements"] == [{"human": "partial", "judge": "correct"}]


def test_summarise_counts_only_correct_as_correct_and_splits_languages():
    judged = [
        {**rec("f01-de"), "verdict": "correct"},
        {**rec("f01-en"), "verdict": "partial"},
        {**rec("f02-de"), "verdict": "wrong"},
        {**rec("f02-en"), "verdict": "correct"},
        {**rec("u01-de", gold_answerable=False, answerable=False), "verdict": "correct"},
        {**rec("u11-de", gold_answerable=False), "verdict": "review"},
    ]
    s = j.summarise(judged)
    assert s["answerable_correct"]["mean"] == 0.5 and s["answerable_correct"]["n"] == 4
    assert s["answerable_correct_de"]["mean"] == 0.5
    assert s["answerable_correct_en"]["mean"] == 0.5
    assert s["unanswerable_handled"]["mean"] == 0.5
    assert s["verdicts"]["review"] == 1


def test_sheet_hides_the_judge_verdict_and_labels_roundtrip(tmp_path):
    rows = [{**rec("f01-de"), "verdict": "correct", "judged_by": "llm", "reason": "SECRET"},
            {**rec("f02-de"), "verdict": "wrong", "judged_by": "rule", "reason": "r"}]  # fmt: skip
    sample = j.pick_sample(rows, 5)
    assert [r["qid"] for r in sample] == ["f01-de"]  # rule-based rows are not sampled
    path = tmp_path / "sheet.csv"
    j.write_sheet(sample, path)
    text = path.read_text(encoding="utf-8")
    assert "SECRET" not in text and "correct" not in text.split("\n", 1)[1]
    with open(path, newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    table[0]["human_label"] = "C"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=j.SHEET_FIELDS)
        w.writeheader()
        w.writerows(table)
    assert j.read_labels(path) == {"f01-de": "correct"}


def test_read_labels_rejects_unknown_label(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("qid,question,reference_answer,system_answer,human_label\nf1,q,r,a,x\n")
    with pytest.raises(ValueError):
        j.read_labels(path)


def test_test_split_is_locked(capsys):
    assert j.main(["--split", "test"]) == 2
    assert "locked" in capsys.readouterr().out
