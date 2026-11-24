"""Delivery regression checks for lazy scoring and the 8x750 sample.

This file is additive and deliberately separately named so it cannot conflict
with updates to the original ``tests/test_app.py`` suite.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app
from app import web as web_module
from app.engine import (
    build_context,
    build_result,
    parse_transcript,
    solve_pairing,
    solve_pairing_certified,
)

SAMPLES = ROOT / "samples"


def load_json(name: str):
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


@pytest.fixture()
def client():
    app = create_app({"TESTING": True, "SECRET_KEY": "delivery-check"})
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture(scope="module")
def scale_pair():
    ref = parse_transcript(
        load_json("scale8_ref.json"), label="人工稿", require_nonempty=True
    )
    hyp = parse_transcript(load_json("scale8_hyp.json"), label="识别稿")
    return ref, hyp


# ------------------------------------------------------------- sample shape

def test_scale_sample_shape_and_shared_grid(scale_pair):
    ref, hyp = scale_pair
    assert len(ref.speakers) == len(hyp.speakers) == 8
    assert ref.word_count == hyp.word_count == 6000

    grids = set()
    vocabulary = {}
    for transcript in (ref, hyp):
        for speaker in transcript.speakers:
            words = transcript.speaker_words[speaker]
            times = [w.time_ms for w in words]
            assert len(words) == 750
            assert all(b > a for a, b in itertools.pairwise(times))
            grids.add(tuple(times))
            vocabulary[speaker] = {w.text for w in words}
    assert len(grids) == 1  # one shared increasing grid for every speaker

    for i in range(1, 9):
        rw = ref.speaker_words[f"S{i}"]
        hw = hyp.speaker_words[f"T{i}"]
        differing = [k for k, (a, b) in enumerate(zip(rw, hw))
                     if a.text != b.text]
        assert len(differing) == 1
        k = differing[0]
        assert rw[k].time_ms == hw[k].time_ms
        assert len(vocabulary[f"S{i}"] & vocabulary[f"T{i}"]) == 749

    # Non-corresponding speakers never share vocabulary.
    names = list(vocabulary)
    for x, left_name in enumerate(names):
        li = int(left_name[1:])
        for right_name in names[x + 1:]:
            ri = int(right_name[1:])
            corresponding = left_name[0] != right_name[0] and li == ri
            if not corresponding:
                assert vocabulary[left_name].isdisjoint(
                    vocabulary[right_name]
                ), (left_name, right_name)


def test_scale_sample_button_rendered(client):
    body = client.get("/").get_data(as_text=True)
    assert 'value="scale8"' in body
    assert "规模样稿" in body


# ------------------------------------------------------- engine laziness

def test_first_review_runs_dp_for_exactly_eight_pairs(scale_pair):
    ref, hyp = scale_pair
    context = build_context(ref, hyp)
    result = build_result(ref, hyp, {}, context)

    assert context.scored_pairs == {
        (f"S{i}", f"T{i}") for i in range(1, 9)
    }
    assert result.total_errors == 8
    assert result.substitution_count == 8
    assert result.wer == pytest.approx(8 / 6000)
    by_ref = {p.ref_speaker: p.hyp_speaker for p in result.pairs}
    assert by_ref == {f"S{i}": f"T{i}" for i in range(1, 9)}
    assert all(p.errors == 1 and p.matches == 749
               for p in result.pairs if p.ref_speaker != "—")


@pytest.mark.parametrize("sample", ["meeting", "sparse"])
def test_certified_matches_full_matrix_on_shipped_samples(sample):
    ref = parse_transcript(
        load_json(f"{sample}_ref.json"), label="r", require_nonempty=True
    )
    hyp = parse_transcript(load_json(f"{sample}_hyp.json"), label="h")
    context = build_context(ref, hyp)

    for edge, lower in context.lower_bounds.items():
        assert lower <= context.pair_costs.exact_cost(*edge)

    locks = {"S2": "T1"} if sample == "meeting" else {}
    certified = solve_pairing_certified(ref, hyp, locks, context)
    for edge in list(context.pair_costs):
        context.pair_costs.exact_cost(*edge)
    eager = solve_pairing(ref, hyp, locks, context)
    assert certified.pairs == eager.pairs
    assert certified.total_errors == eager.total_errors
    assert certified.unpaired_ref == eager.unpaired_ref
    assert certified.unpaired_hyp == eager.unpaired_hyp

    lazy_result = build_result(ref, hyp, locks)
    eager_result = build_result(ref, hyp, locks, context)
    assert lazy_result.total_errors == eager_result.total_errors
    assert [(e.kind, e.sort_ms, e.ref_word, e.hyp_word)
            for e in lazy_result.errors] == [
        (e.kind, e.sort_ms, e.ref_word, e.hyp_word)
        for e in eager_result.errors
    ]


# ----------------------------------------------------------- web workflow

def test_web_scale_review_lock_undo_reuse_and_rollback(client, monkeypatch):
    engine_module = sys.modules["app.engine"]
    calls = []
    real_pair_cost = engine_module.pair_cost

    def counting(*args, **kwargs):
        calls.append(1)
        return real_pair_cost(*args, **kwargs)

    monkeypatch.setattr(engine_module, "pair_cost", counting)

    response = client.post("/sample", data={"sample": "scale8"})
    assert response.status_code == 200
    assert len(calls) == 8

    api = client.get("/api/result").get_json()
    assert api["total_errors"] == 8
    assert api["wer"] == pytest.approx(8 / 6000)
    assert len(api["errors"]) == 8
    assert {e["kind"] for e in api["errors"]} == {"substitution"}
    assert '<span class="score-value">8</span>' in response.get_data(as_text=True)

    # Locking a skipped off-diagonal pair computes that pair plus the single
    # displaced partner needed for the new global optimum.
    calls.clear()
    response = client.post("/locks", data={"lock_S1": "T2"})
    assert response.status_code == 200
    assert len(calls) == 2
    locked = client.get("/api/result").get_json()
    assert {"ref_speaker": "S1", "hyp_speaker": "T2"} in locked["locks"]
    assert locked["total_errors"] == 1506
    assert locked["revision"] == 2
    assert "1506" in response.get_data(as_text=True)

    # Undo returns to the certified optimum without new word-level scoring.
    calls.clear()
    response = client.post("/locks", data={"lock_S1": ""})
    assert response.status_code == 200
    assert calls == []
    undone = client.get("/api/result").get_json()
    assert undone["locks"] == []
    assert undone["total_errors"] == 8
    assert undone["revision"] == 3

    # Re-locking reuses the exact costs already computed for this transcript.
    calls.clear()
    response = client.post("/locks", data={"lock_S1": "T2"})
    assert response.status_code == 200
    assert calls == []
    assert client.get("/api/result").get_json()["total_errors"] == 1506
    last_good = client.get("/api/result").get_json()

    # A failing recompute must preserve the previous locks and full snapshot.
    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated recompute failure")

    with monkeypatch.context() as patch:
        patch.setattr(web_module, "build_result", boom)
        failed = client.post("/locks", data={"lock_S2": "T5"})
        assert failed.status_code == 500
        assert "已保留" in failed.get_data(as_text=True)

    retained = client.get("/api/result").get_json()
    assert retained["locks"] == last_good["locks"]
    assert retained["total_errors"] == last_good["total_errors"]
    assert retained["revision"] == last_good["revision"]
    assert len(retained["errors"]) == len(last_good["errors"])

    # Session remains usable after the failed attempt.
    assert client.post("/locks", data={"lock_S2": "T5"}).status_code == 200
    recovered = client.get("/api/result").get_json()
    assert {"ref_speaker": "S2", "hyp_speaker": "T5"} in recovered["locks"]


def test_new_sample_resets_revision_and_aliases(client):
    assert client.post("/sample", data={"sample": "large"}).status_code == 200
    assert client.get("/api/result").get_json()["total_errors"] == 8

    response = client.post("/sample", data={"sample": "meeting"})
    assert response.status_code == 200
    meeting = client.get("/api/result").get_json()
    assert meeting["revision"] == 1
    assert meeting["total_errors"] == 8

    assert client.get("/samples/large8_ref.json").status_code == 200
    assert client.get("/samples/unknown.json").status_code == 404
