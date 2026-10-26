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


def test_interactive_labelling_saves_skips_and_quits(tmp_path):
    rows = [{**rec(f"f0{i}-de"), "verdict": "correct", "judged_by": "llm"} for i in range(1, 5)]
    path = tmp_path / "sheet.csv"
    j.write_sheet(rows, path)
    answers = iter(["x", "c", "s", "w", "q"])  # 'x' is ignored and asked again
    j.label_interactively(path, ask=lambda prompt: next(answers))
    assert j.read_labels(path) == {"f01-de": "correct", "f03-de": "wrong"}
    # a second session only asks for what is still blank (f02 was skipped, f04 not reached)
    asked = []
    again = iter(["p", "p"])
    j.label_interactively(path, ask=lambda prompt: asked.append(prompt) or next(again))
    assert len(asked) == 2 and len(j.read_labels(path)) == 4


def test_relabel_asks_again_only_for_the_named_qids(tmp_path):
    rows = [{**rec(f"f0{i}-de"), "verdict": "correct", "judged_by": "llm"} for i in range(1, 4)]
    path = tmp_path / "sheet.csv"
    j.write_sheet(rows, path)
    first = iter(["c", "c", "c"])
    j.label_interactively(path, ask=lambda prompt: next(first))
    second = iter(["p"])
    j.label_interactively(path, ask=lambda prompt: next(second), relabel=["f02-de"])
    assert j.read_labels(path) == {"f01-de": "correct", "f02-de": "partial", "f03-de": "correct"}


def test_swap_number_changes_a_key_number_but_not_citations_or_paragraphs():
    assert j.swap_number("Mindestens 60 ECTS-Punkte [C1].", "Mindestens 60 ECTS-Punkte.") == (
        "Mindestens 70 ECTS-Punkte [C1]."
    )
    assert j.swap_number("At least IELTS 6.5 [C1]", "IELTS 6.5 or TOEFL 90") == (
        "At least IELTS 7.5 [C1]"
    )
    assert j.swap_number("Nach § 25 Abs. 1 gilt 5 Jahre.", "Fünf Jahre, 5") == (
        "Nach § 25 Abs. 1 gilt 7 Jahre."
    )
    assert j.swap_number("Es gilt [C2] ohne Zahl.", "Nichts") is None
    assert j.swap_number("It lasts 20 minutes.", "It lasts 5 minutes.") is None  # not a key fact


def test_swap_number_word_keeps_capitalisation_and_needs_the_reference_to_agree():
    assert j.swap_number_word("Vier Fachsemester [C1].", "Vier Fachsemester.") == (
        "Sechs Fachsemester [C1]."
    )
    assert j.swap_number_word("It takes four semesters.", "Four semesters.") == (
        "It takes six semesters."
    )
    assert j.swap_number_word("It takes four semesters.", "Some semesters.") is None


def test_swap_name_replaces_only_a_name_the_reference_has():
    answer = "Verantwortlich ist Prof. Dr. Stefan Ultes [C1]."
    assert FAKE in j.swap_name(answer, "Prof. Dr. Stefan Ultes.")
    assert j.swap_name(answer, "Niemand.") is None


FAKE = j.FAKE_NAME


def test_stress_cases_only_come_from_answers_the_judge_called_correct():
    def row(qid, text, ref, verdict="correct", by="llm"):
        return {**rec(qid, text=text), "gold_answer": ref, "verdict": verdict, "judged_by": by}

    judged = [
        row("f01-de", "Vier Fachsemester [C1].", "Vier Fachsemester."),
        row("f02-de", "Mindestens 60 ECTS [C1].", "Mindestens 60 ECTS."),
        row("f03-de", "Falsch.", "Richtig.", verdict="wrong"),
        row("f04-de", "Regel.", "Regel.", by="rule"),
    ]
    cases = j.make_stress_cases(judged)
    assert {c["qid"] for c in cases} == {"f01-de", "f02-de"}
    kinds = {(c["qid"], c["kind"]) for c in cases}
    assert ("f01-de", "changed_number_word") in kinds and ("f02-de", "changed_number") in kinds
    assert ("f01-de", "not_stated") in kinds and ("f02-de", "other_answer") in kinds
    other = next(c for c in cases if c["qid"] == "f01-de" and c["kind"] == "other_answer")
    assert other["answer"] == "Mindestens 60 ECTS [C1]."  # taken from a different pair


def test_stress_summary_counts_caught_and_lists_misses():
    cases = [
        {"qid": "a", "kind": "changed_number", "question": "q", "reference": "r", "answer": "x"},
        {"qid": "b", "kind": "changed_number", "question": "q", "reference": "r", "answer": "y"},
        {"qid": "c", "kind": "not_stated", "question": "q", "reference": "r", "answer": "z"},
    ]
    replies = iter(['{"verdict": "wrong"}', '{"verdict": "correct"}', '{"verdict": "partial"}'])
    summary = j.summarise_stress(j.run_stress(cases, lambda s, u: next(replies)))
    assert summary["kinds"]["changed_number"] == {"n": 2, "caught": 1, "rate": 0.5}
    assert summary["caught"] == 2 and summary["n"] == 3
    assert [m["qid"] for m in summary["missed"]] == ["b"]


def test_swap_number_changes_every_occurrence_of_the_key_number():
    out = j.swap_number("It lasts 90 minutes; the handbook also says 90 minutes.", "90 minutes.")
    assert out == "It lasts 100 minutes; the handbook also says 100 minutes."
