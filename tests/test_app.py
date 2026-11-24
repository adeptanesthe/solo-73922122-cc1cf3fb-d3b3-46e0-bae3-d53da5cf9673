import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app
from app.engine import (
    MAX_ALIGN_GAP_MS,
    MAX_SPEAKERS,
    MAX_WORDS,
    TranscriptError,
    align_pair,
    build_context,
    build_result,
    parse_transcript,
    solve_pairing,
)

SAMPLES = ROOT / "samples"


def word(text, time_ms, speaker):
    return {"word": text, "time_ms": time_ms, "speaker": speaker}


def load_sample(name):
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def transcript(items, label="稿", require_nonempty=False):
    return parse_transcript(items, label=label, require_nonempty=require_nonempty)


@pytest.fixture()
def client():
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    with app.test_client() as test_client:
        yield test_client


def upload(client, ref_items, hyp_items):
    data = {
        "ref_file": (
            __import__("io").BytesIO(
                json.dumps(ref_items, ensure_ascii=False).encode("utf-8")
            ),
            "ref.json",
        ),
        "hyp_file": (
            __import__("io").BytesIO(
                json.dumps(hyp_items, ensure_ascii=False).encode("utf-8")
            ),
            "hyp.json",
        ),
    }
    return client.post("/upload", data=data, content_type="multipart/form-data")


# ---------------------------------------------------------------- validation

def test_reference_must_not_be_empty():
    with pytest.raises(TranscriptError, match="人工稿不能为空"):
        parse_transcript([], label="人工稿", require_nonempty=True)


def test_recognized_may_be_empty():
    parsed = parse_transcript([], label="识别稿")
    assert parsed.word_count == 0
    assert parsed.speakers == ()


def test_word_limit_is_6000():
    ok = [word("词", index * 600, "S1") for index in range(MAX_WORDS)]
    assert transcript(ok).word_count == MAX_WORDS
    too_many = [word("词", index * 600, "S1") for index in range(MAX_WORDS + 1)]
    with pytest.raises(TranscriptError, match="超过单稿上限"):
        transcript(too_many)


def test_speaker_limit_is_8():
    ok = [word("词", index * 600, f"S{index}") for index in range(MAX_SPEAKERS)]
    transcript(ok)
    too_many = [
        word("词", index * 600, f"S{index}") for index in range(MAX_SPEAKERS + 1)
    ]
    with pytest.raises(TranscriptError, match="超过单稿上限"):
        transcript(too_many)


@pytest.mark.parametrize(
    "bad, message",
    [
        ([{"word": "好", "speaker": "S1"}], "缺少毫秒时间"),
        ([{"word": "好", "time_ms": -1, "speaker": "S1"}], "不能为负数"),
        ([{"word": "好", "time_ms": 1.5, "speaker": "S1"}], "整数毫秒"),
        ([{"word": "好", "time_ms": "100", "speaker": "S1"}], "必须是数字"),
        ([{"word": "好", "time_ms": 10**16, "speaker": "S1"}], "时间过大"),
        ([{"word": "好", "time_ms": True, "speaker": "S1"}], "必须是数字"),
        ([{"word": "  ", "time_ms": 100, "speaker": "S1"}], "word 文本"),
        ([{"word": "好", "time_ms": 100}], "说话人编号"),
        ([{"word": "好", "time_ms": 100, "speaker": ""}], "不能为空"),
        ("not json", "必须是 JSON 数组"),
        ({"no_words": []}, "words"),
    ],
)
def test_bad_payloads_are_rejected(bad, message):
    with pytest.raises(TranscriptError, match=message):
        transcript(bad)


def test_words_are_normalised_into_time_order():
    parsed = transcript(
        [word("后", 2000, "S1"), word("前", 1000, "S1")], label="人工稿"
    )
    assert [w.text for w in parsed.words] == ["前", "后"]


# ------------------------------------------------------------------ pairing

def test_unpaired_speakers_count_every_word():
    ref = transcript([word("甲", 1000, "S1"), word("乙", 2000, "S1")],
                     label="人工稿", require_nonempty=True)
    hyp = transcript([word("甲", 1000, "T1"), word("丙", 2000, "T2")])
    result = build_result(ref, hyp)
    # Only one pair is possible; the third hyp word is an insertion and the
    # second ref word may be deleted (or matched -- pairing S1->T2 costs 1 too).
    assert result.total_errors == 2
    assert result.ref_word_count == 2
    assert result.wer == 1.0
    assert result.insertion_count + result.deletion_count + result.substitution_count == 2


def test_pairing_is_one_to_one_and_minimal():
    ref = transcript(
        [word("甲", 1000, "S1"), word("乙", 5000, "S2")],
        label="人工稿", require_nonempty=True,
    )
    hyp = transcript(
        [word("乙", 5000, "T1"), word("甲", 1000, "T2")], label="识别稿"
    )
    solution = solve_pairing(ref, hyp)
    assert solution.pairs == {"S1": "T2", "S2": "T1"}
    assert solution.total_errors == 0


def test_global_assignment_beats_greedy():
    # S1 has a perfect 1-word match with T2; pairing it with T1 costs 1 but
    # would make S2 unable to use T2.
    ref = transcript(
        [
            word("甲", 1000, "S1"),
            word("乙", 2000, "S2"),
            word("乙", 3000, "S2"),
        ],
        label="人工稿", require_nonempty=True,
    )
    hyp = transcript(
        [
            word("甲", 1000, "T1"),
            word("乙", 2000, "T2"),
            word("乙", 3000, "T2"),
        ],
        label="识别稿",
    )
    result = build_result(ref, hyp)
    assert result.total_errors == 0
    assert result.pairs[0].ref_speaker == "S1"
    assert result.pairs[0].hyp_speaker == "T1"


# ---------------------------------------------------------------- alignment

def test_500ms_is_inclusive_and_501ms_is_not():
    ref = transcript([word("好", 1000, "S1")], label="人工稿",
                     require_nonempty=True)
    exact = transcript([word("好", 1000 + MAX_ALIGN_GAP_MS, "T1")])
    assert build_result(ref, exact).total_errors == 0

    beyond = transcript([word("好", 1000 + MAX_ALIGN_GAP_MS + 1, "T1")])
    result = build_result(ref, beyond)
    assert result.total_errors == 2
    assert result.deletion_count == 1
    assert result.insertion_count == 1


def test_substitution_delete_insert_counted_once_each():
    ref = transcript(
        [word("a", 1000, "S1"), word("b", 2000, "S1")],
        label="人工稿", require_nonempty=True,
    )
    hyp = transcript([word("a", 1000, "T1"), word("c", 2000, "T1")])
    result = build_result(ref, hyp)
    assert result.total_errors == 1
    assert result.substitution_count == 1
    assert result.wer == 0.5

    hyp_missing = transcript([word("a", 1000, "T1")])
    deleted = build_result(ref, hyp_missing)
    assert deleted.deletion_count == 1

    hyp_extra = transcript(
        [word("a", 1000, "T1"), word("b", 2000, "T1"), word("d", 2500, "T1")]
    )
    inserted = build_result(ref, hyp_extra)
    assert inserted.insertion_count == 1


def test_alignment_prefers_nearer_timestamp_when_word_repeats():
    ref = transcript([word("好", 2000, "S1")], label="人工稿",
                     require_nonempty=True)
    hyp = transcript(
        [word("好", 1700, "T1"), word("好", 2480, "T1")], label="识别稿"
    )
    result = build_result(ref, hyp)
    assert result.total_errors == 1
    kept = next(e for e in result.errors if e.kind == "insertion")
    # 2000 ms is 300 ms from 1700 and 480 ms from 2480, so the latter is extra.
    assert kept.hyp_word.time_ms == 2480


def test_large_alignment_memory_stays_linear():
    """Peak extra memory must scale ~linearly, not as n*m."""
    import gc
    import tracemalloc

    def measure(n):
        ref_items = []
        hyp_items = []
        time_ms = 0
        for index in range(n):
            time_ms += 200
            ref_items.append(word(f"w{index % 97}", time_ms, "S1"))
            hyp_items.append(
                word(f"w{index % 97}" if index % 11 else "x",
                     time_ms + (index % 121) - 60, "T1")
            )
        ref = transcript(ref_items, label="人工稿", require_nonempty=True)
        hyp = transcript(hyp_items, label="识别稿")
        gc.collect()
        tracemalloc.start()
        build_result(ref, hyp)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    small = measure(750)
    large = measure(3000)
    # A full n*m matrix of int64 would need ~144 MB for the larger case; the
    # Hirschberg implementation should stay under a few MB and grow linearly.
    assert large < 8 * 1024 * 1024
    assert large < 6 * small


def test_traceback_matches_cost_on_sample():
    ref = transcript(load_sample("meeting_ref.json"), label="人工稿",
                     require_nonempty=True)
    hyp = transcript(load_sample("meeting_hyp.json"), label="识别稿")
    context = build_context(ref, hyp)
    for (ref_speaker, hyp_speaker), cost in context.pair_costs.items():
        ops = align_pair(ref.speaker_words[ref_speaker],
                         hyp.speaker_words[hyp_speaker])
        assert sum(1 for op, _, _ in ops if op != "match") == cost


# ------------------------------------------------------------------- samples

def test_meeting_sample_demonstrates_pairing_and_all_error_kinds():
    ref = transcript(load_sample("meeting_ref.json"), label="人工稿",
                     require_nonempty=True)
    hyp = transcript(load_sample("meeting_hyp.json"), label="识别稿")
    result = build_result(ref, hyp)
    assert result.total_errors == 8
    assert result.substitution_count == 3
    assert result.deletion_count == 1
    assert result.insertion_count == 4
    assert result.wer == 8 / 53
    pairs = {(row.ref_speaker, row.hyp_speaker) for row in result.pairs}
    assert ("S1", "T1") in pairs
    assert ("S2", "T3") in pairs
    assert ("S3", "T2") in pairs
    assert ("—", "T4") in pairs
    assert result.unpaired_hyp == ("T4",)


def test_sparse_sample_shows_unpaired_and_gap_limit():
    ref = transcript(load_sample("sparse_ref.json"), label="人工稿",
                     require_nonempty=True)
    hyp = transcript(load_sample("sparse_hyp.json"), label="识别稿")
    result = build_result(ref, hyp)
    assert result.total_errors == 6
    assert result.unpaired_ref == ("S3",)
    assert result.deletion_count == 4
    assert result.insertion_count == 2


# --------------------------------------------------------------------- locks

def test_locks_are_compulsory():
    ref = transcript(load_sample("meeting_ref.json"), label="人工稿",
                     require_nonempty=True)
    hyp = transcript(load_sample("meeting_hyp.json"), label="识别稿")
    locked = build_result(ref, hyp, {"S2": "T1"})
    by_ref = {row.ref_speaker: row.hyp_speaker for row in locked.pairs}
    assert by_ref["S2"] == "T1"
    assert locked.lock_signature == (("S2", "T1"),)
    # Forcing a bad pair cannot score better than the free optimum.
    assert locked.total_errors > 8


def test_duplicate_lock_is_rejected():
    ref = transcript(load_sample("meeting_ref.json"), label="人工稿",
                     require_nonempty=True)
    hyp = transcript(load_sample("meeting_hyp.json"), label="识别稿")
    with pytest.raises(ValueError, match="一对一"):
        solve_pairing(ref, hyp, {"S1": "T1", "S2": "T1"})


def test_lock_to_unknown_speaker_is_rejected():
    ref = transcript([word("甲", 1000, "S1")], label="人工稿",
                     require_nonempty=True)
    hyp = transcript([word("甲", 1000, "T1")])
    with pytest.raises(ValueError, match="不存在"):
        solve_pairing(ref, hyp, {"S1": "T9"})


# ----------------------------------------------------------------------- web

def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "转写验收" in response.get_data(as_text=True)


def test_upload_and_display(client):
    response = upload(client, load_sample("meeting_ref.json"),
                      load_sample("meeting_hyp.json"))
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "8" in body
    assert "T4" in body
    api = client.get("/api/result").get_json()
    assert api["total_errors"] == 8
    assert api["revision"] == 1
    assert len(api["errors"]) == 8
    assert api["errors"][0]["kind"] == "deletion"


def test_upload_rejects_empty_reference(client):
    response = upload(client, [], load_sample("meeting_hyp.json"))
    assert response.status_code == 400
    assert "人工稿不能为空" in response.get_data(as_text=True)


def test_upload_rejects_invalid_json_text(client):
    response = client.post(
        "/upload",
        data={
            "ref_file": (__import__("io").BytesIO(b"{oops"), "ref.json"),
            "hyp_file": (__import__("io").BytesIO(b"[]"), "hyp.json"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "合法 JSON" in response.get_data(as_text=True)


def test_lock_then_unlock_via_web(client):
    upload(client, load_sample("meeting_ref.json"), load_sample("meeting_hyp.json"))
    response = client.post("/locks", data={"lock_S2": "T1"})
    assert response.status_code == 200
    api = client.get("/api/result").get_json()
    assert {"ref_speaker": "S2", "hyp_speaker": "T1"} in api["locks"]
    assert api["revision"] == 2
    locked_errors = api["total_errors"]
    assert locked_errors > 8

    response = client.post("/locks", data={"lock_S2": ""})
    assert response.status_code == 200
    api = client.get("/api/result").get_json()
    assert api["locks"] == []
    assert api["total_errors"] == 8
    assert api["revision"] == 3


def test_conflicting_lock_keeps_original_lock(client):
    upload(client, load_sample("meeting_ref.json"), load_sample("meeting_hyp.json"))
    client.post("/locks", data={"lock_S1": "T1"})
    response = client.post("/locks", data={"lock_S2": "T1"})
    assert response.status_code == 409
    body = response.get_data(as_text=True)
    assert "已锁定给人工稿说话人" in body
    assert "原锁定未被改动" in body
    api = client.get("/api/result").get_json()
    assert api["locks"] == [{"ref_speaker": "S1", "hyp_speaker": "T1"}]
    # The score shown in the conflict page must still be the S1->T1 score.
    assert "评分与错词已按最新有效锁定重算" not in body


def test_lock_wrong_hypothesis_for_speaker_is_still_a_forced_pair(client):
    # S2's words are spoken by T3; locking S2->T2 must be honoured even though
    # it is a bad match, and the displayed score must be the locked score.
    upload(client, load_sample("meeting_ref.json"), load_sample("meeting_hyp.json"))
    baseline = client.get("/api/result").get_json()["total_errors"]
    response = client.post("/locks", data={"lock_S2": "T2"})
    assert response.status_code == 200
    api = client.get("/api/result").get_json()
    assert api["locks"] == [{"ref_speaker": "S2", "hyp_speaker": "T2"}]
    assert api["total_errors"] > baseline
    body = response.get_data(as_text=True)
    assert str(api["total_errors"]) in body
    # The page never shows the stale baseline score as the headline.
    assert "已锁定" in body


def test_manual_pair_and_unlock_forms(client):
    upload(client, load_sample("meeting_ref.json"), load_sample("meeting_hyp.json"))
    response = client.post(
        "/locks",
        data={"manual_mode": "pair", "manual_ref": "S3", "manual_hyp": "T1"},
    )
    assert response.status_code == 200
    api = client.get("/api/result").get_json()
    assert api["locks"] == [{"ref_speaker": "S3", "hyp_speaker": "T1"}]

    response = client.post(
        "/locks", data={"manual_mode": "unlock", "manual_unlock": "S3"}
    )
    assert response.status_code == 200
    assert client.get("/api/result").get_json()["locks"] == []


def test_api_requires_upload(client):
    response = client.get("/api/result")
    assert response.status_code == 404


def test_reset_clears_session(client):
    upload(client, load_sample("meeting_ref.json"), load_sample("meeting_hyp.json"))
    client.post("/reset")
    assert client.get("/api/result").status_code == 404
    assert client.get("/").status_code == 200


# ------------------------------------------------------------- sample routes

def test_load_sample_via_form(client):
    response = client.post("/sample", data={"sample": "sparse"})
    assert response.status_code == 200
    api = client.get("/api/result").get_json()
    assert api["total_errors"] == 6
    assert api["unpaired_ref"] == ["S3"]


def test_unknown_sample_is_404(client):
    assert client.post("/sample", data={"sample": "nope"}).status_code == 404


def test_sample_download_is_whitelisted(client):
    ok = client.get("/samples/meeting_ref.json")
    assert ok.status_code == 200
    assert ok.mimetype == "application/json"
    assert client.get("/samples/../app/web.py").status_code == 404
    assert client.get("/samples/unknown.json").status_code == 404
