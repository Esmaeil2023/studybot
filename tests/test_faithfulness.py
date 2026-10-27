import csv
import json

import pytest

from studybot import faithfulness as f
from studybot.ingest import Chunk


def chunk(cid, text):
    return Chunk(
        chunk_id=cid, doc_id="reg", doc_title="Reg", version="1", section="§ 1",
        section_title="T", page_start=1, page_end=1, header="[Reg | § 1]", text=text,
        n_words=len(text.split()),
    )  # fmt: skip


BY_ID = {
    "reg:a:0": chunk("reg:a:0", "Die Regelstudienzeit beträgt vier Fachsemester."),
    "reg:b:0": chunk("reg:b:0", "Die Masterarbeit hat 30 ECTS-Punkte."),
}


def record(qid="f01-de", text="Es sind vier Fachsemester [C1].", cites=("reg:a:0",), ans=True):
    return {"qid": qid, "answerable": ans, "text": text, "citations": list(cites)}


def reply(*verdicts):
    return json.dumps(
        {"claims": [{"claim": f"c{i}", "verdict": v} for i, v in enumerate(verdicts)]}
    )


def test_parse_claims_accepts_good_and_rejects_bad_replies():
    assert f.parse_claims(reply("supported", "unsupported"))[1]["verdict"] == "unsupported"
    assert f.parse_claims("text " + reply("supported") + " text") is not None
    assert f.parse_claims("") is None
    assert f.parse_claims('{"claims": []}') is None
    assert f.parse_claims('{"claims": [{"claim": "x", "verdict": "maybe"}]}') is None
    assert f.parse_claims('{"claims": "nope"}') is None


def test_prompt_contains_only_the_cited_excerpts_and_no_citation_markers():
    seen = []
    f.faith_one(record(), BY_ID, lambda s, u: seen.append(u) or reply("supported"))
    assert "vier Fachsemester" in seen[0] and "ECTS" not in seen[0]  # reg:b:0 not cited
    assert "[C1]" not in seen[0] and "Answer: Es sind vier Fachsemester" in seen[0]


def test_fully_supported_answer_is_faithful():
    out = f.faith_one(record(), BY_ID, lambda s, u: reply("supported", "supported"))
    assert out["faithful"] and out["n_supported"] == out["n_claims"] == 2
    assert out["status"] == "graded"


def test_one_unsupported_claim_makes_the_answer_unfaithful():
    out = f.faith_one(record(), BY_ID, lambda s, u: reply("supported", "unsupported"))
    assert not out["faithful"] and (out["n_supported"], out["n_claims"]) == (1, 2)


def test_refusals_are_not_graded():
    assert f.faith_one(record(ans=False), BY_ID, lambda s, u: pytest.fail("no call")) is None


def test_answer_without_a_usable_citation_cannot_be_checked():
    out = f.faith_one(record(cites=()), BY_ID, lambda s, u: pytest.fail("no call"))
    assert out["status"] == "no_citation" and not out["faithful"]
    gone = f.faith_one(record(cites=("zzz:0",)), BY_ID, lambda s, u: pytest.fail("no call"))
    assert gone["status"] == "no_citation"


def test_unreadable_reply_is_retried_once_then_marked():
    replies = iter(["", reply("supported")])
    assert f.faith_one(record(), BY_ID, lambda s, u: next(replies))["faithful"]
    out = f.faith_one(record(), BY_ID, lambda s, u: "")
    assert out["status"] == "unreadable" and not out["faithful"]


def test_summary_counts_claims_and_answers_and_lists_problems():
    results = [
        {**record("f01-de"), "faithful": True, "n_claims": 2, "n_supported": 2, "status": "graded"},
        {**record("f01-en"), "faithful": True, "n_claims": 1, "n_supported": 1, "status": "graded"},
        {
            **record("f02-de"),
            "faithful": False,
            "n_claims": 2,
            "n_supported": 1,
            "status": "graded",
        },
        {
            **record("f02-en"),
            "faithful": False,
            "n_claims": 0,
            "n_supported": 0,
            "status": "no_citation",
        },  # fmt: skip
    ]
    s = f.summarise(results)
    assert s["n_answers"] == 4 and s["n_claims"] == 5
    assert s["claim_support"] == pytest.approx(4 / 5)
    assert (
        s["fully_faithful"] == 0.5 and s["fully_faithful_ci"][0] <= 0.5 <= s["fully_faithful_ci"][1]
    )
    assert s["not_graded"] == ["f02-en"] and s["unfaithful"] == ["f02-de"]
    assert "0.50" in f.format_summary(s)


def test_summary_of_nothing():
    assert f.summarise([]) == {"n_answers": 0}


def test_stress_cases_are_built_only_from_faithful_answers_and_detected():
    graded = [
        {**record("f01-de", text="Es sind 30 ECTS [C1]."), "faithful": True},
        {**record("f02-de"), "faithful": False},
        None,
    ]
    cases = f.make_stress_cases(graded)
    kinds = {(c["qid"], c["kind"]) for c in cases}
    assert kinds == {("f01-de", "invented_sentence"), ("f01-de", "changed_number")}
    assert any(f.UNSUPPORTED_SENTENCE in c["text"] for c in cases)
    assert any("40 ECTS" in c["text"] for c in cases)
    # a judge that flags everything catches everything; one that flags nothing misses all
    flag = f.summarise_stress(f.run_stress(cases, BY_ID, lambda s, u: reply("unsupported")))
    miss = f.summarise_stress(f.run_stress(cases, BY_ID, lambda s, u: reply("supported")))
    assert flag["caught"] == flag["n"] == 2 and not flag["missed"]
    assert miss["caught"] == 0 and len(miss["missed"]) == 2


def test_sheet_hides_the_judge_and_labels_roundtrip(tmp_path):
    results = [{**record("f01-de"), "status": "graded", "faithful": True, "claims": ["SECRET"]},
               {**record("f02-de"), "status": "no_citation", "faithful": False}]  # fmt: skip
    sample = f.pick_sample(results, 5)
    assert [r["qid"] for r in sample] == ["f01-de"]
    path = tmp_path / "s.csv"
    f.write_sheet(sample, BY_ID, path)
    text = path.read_text(encoding="utf-8")
    assert "SECRET" not in text and "vier Fachsemester" in text and "[C1]" not in text
    answers = iter(["x", "y"])
    f.label_interactively(path, ask=lambda p: next(answers))
    assert f.read_labels(path) == {"f01-de": "y"}
    again = iter(["n"])
    f.label_interactively(path, ask=lambda p: next(again), relabel=["f01-de"])
    assert f.read_labels(path) == {"f01-de": "n"}


def test_read_labels_rejects_other_letters(tmp_path):
    path = tmp_path / "s.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=f.SHEET_FIELDS)
        w.writeheader()
        w.writerow({"qid": "q", "answer": "a", "excerpts": "e", "human_label": "c"})
    with pytest.raises(ValueError):
        f.read_labels(path)


def test_agreement_and_lock():
    r = f.agreement([("y", "y"), ("y", "n"), ("n", "n"), ("y", "y")])
    assert r["agreement"] == 0.75 and r["disagreements"] == [{"human": "y", "judge": "n"}]
    assert f.main(["--split", "test"]) == 2


def test_label_without_a_sheet_explains_instead_of_crashing(tmp_path, capsys):
    assert f.main(["--label", "--sheet", str(tmp_path / "none.csv")]) == 2
    assert "not found" in capsys.readouterr().out
