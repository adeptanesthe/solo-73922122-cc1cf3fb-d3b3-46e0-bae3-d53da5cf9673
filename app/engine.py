"""Core transcript validation and word-alignment engine.

The engine is deliberately independent from Flask so that it can be unit-tested
and reasoned about in isolation.  The important properties are:

* a transcript is parsed into a flat, time-ordered list of words;
* two speaker transcripts are aligned with unit-cost edit distance, where a
  diagonal (match/substitution) is only allowed if the two words are at most
  ``MAX_ALIGN_GAP_MS`` milliseconds apart;
* speaker pairing is a one-to-one assignment.  Unpaired reference speakers
  contribute one deletion per word, unpaired hypothesis speakers one insertion
  per word;
* scoring uses a rolling row of ``m + 1`` integers and traceback uses
  Hirschberg's divide-and-conquer algorithm, so extra memory is O(n + m),
  not O(n * m).
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

MAX_WORDS = 6000
MAX_SPEAKERS = 8
MAX_ALIGN_GAP_MS = 500
# Far beyond any real recording (about 31,700 years) but well inside int64, so
# the numeric DP can never overflow on hostile input.
MAX_TIME_MS = 10**15

# The dynamic program stores ``_TIME_WEIGHT * distance + |time gap|`` so that,
# among equally-cheap alignments, the one with the smallest total timing
# deviation wins.  ``_TIME_WEIGHT`` is larger than any reachable gap total
# (6000 words x 500 ms), so edit count always dominates.  int64 is used
# throughout to keep all intermediate sums exact.
_MAX_WORD_GAP_MS = 500
_TIME_WEIGHT = MAX_WORDS * _MAX_WORD_GAP_MS + 1
# A forbidden transition.  Distance is at most 12000 and the gap term at most
# 6000 * 500, so the largest legitimate value is ~1.2e10; 1e15 is far above it
# and adding a few multiples of _TIME_WEIGHT still cannot wrap int64.
_INF = 10**15

_OPS_MATCH = "match"
_OPS_SUB = "sub"
_OPS_DEL = "del"
_OPS_INS = "ins"

_DELETION = "deletion"
_INSERTION = "insertion"
_SUBSTITUTION = "substitution"


class TranscriptError(ValueError):
    """Raised when an uploaded transcript does not satisfy the contract."""


@dataclass(frozen=True)
class Word:
    text: str
    time_ms: int
    speaker: str
    order: int


@dataclass(frozen=True)
class Transcript:
    label: str
    words: tuple[Word, ...]
    speaker_words: Mapping[str, tuple[Word, ...]]
    speakers: tuple[str, ...]

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def speaker_count(self) -> int:
        return len(self.speakers)

    @property
    def duration_ms(self) -> int:
        if not self.words:
            return 0
        return max(w.time_ms for w in self.words)


@dataclass(frozen=True)
class PairingContext:
    """Precomputed material shared by every recomputation of one upload.

    Pair costs do not depend on the locks.  They are nevertheless *not* all
    computed up front: :class:`LazyPairCosts` starts from cheap, provably safe
    lower bounds and only runs the word-level DP for a speaker combination
    when the certifying solver asks for its exact error count.  Exact costs are
    cached on this context and therefore reused by every lock change made on
    the same two transcripts.
    """

    ref: Transcript
    hyp: Transcript
    pair_costs: Mapping[tuple[str, str], int]
    lower_bounds: Mapping[tuple[str, str], int] | None = None
    exact_costs: dict[tuple[str, str], int] = field(default_factory=dict)
    scored_pairs: set[tuple[str, str]] = field(default_factory=set)


@dataclass(frozen=True)
class PairingSolution:
    pairs: Mapping[str, str]
    unpaired_ref: tuple[str, ...]
    unpaired_hyp: tuple[str, ...]
    total_errors: int
    pair_costs: Mapping[tuple[str, str], int]


@dataclass(frozen=True)
class ErrorItem:
    kind: str
    ref_word: Word | None
    hyp_word: Word | None
    ref_speaker: str | None
    hyp_speaker: str | None
    pair_label: str
    sort_ms: int
    sequence: int


@dataclass(frozen=True)
class PairRow:
    ref_speaker: str
    hyp_speaker: str | None
    ref_words: int
    hyp_words: int
    matches: int
    errors: int


@dataclass(frozen=True)
class EngineResult:
    ref_word_count: int
    hyp_word_count: int
    total_errors: int
    wer: float
    substitution_count: int
    deletion_count: int
    insertion_count: int
    pairs: tuple[PairRow, ...]
    errors: tuple[ErrorItem, ...]
    lock_signature: tuple[tuple[str, str], ...]
    unpaired_ref: tuple[str, ...] = ()
    unpaired_hyp: tuple[str, ...] = ()


def natural_speaker_key(speaker: str) -> tuple:
    """Sort S2 before S10 while keeping arbitrary speaker ids stable."""
    match = re.match(r"^(.*?)(\d+)$", speaker)
    if match:
        return (match.group(1), 0, int(match.group(2)), speaker)
    return (speaker, 1, 0, speaker)


def _pick(item: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in item:
            return item[name]
    return None


def _parse_time(value: Any, index: int, label: str) -> int:
    where = f"{label}第 {index} 个词"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptError(f"{where}的毫秒时间必须是数字。")
    if not math.isfinite(float(value)):
        raise TranscriptError(f"{where}的毫秒时间必须是有限数字。")
    if float(value) < 0:
        raise TranscriptError(f"{where}的毫秒时间不能为负数。")
    if float(value) != int(value):
        raise TranscriptError(f"{where}的毫秒时间必须是整数毫秒。")
    if int(value) > MAX_TIME_MS:
        raise TranscriptError(f"{where}的毫秒时间过大（上限 {MAX_TIME_MS}）。")
    return int(value)


def _parse_speaker(value: Any, index: int, label: str) -> str:
    where = f"{label}第 {index} 个词"
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TranscriptError(f"{where}缺少有效的说话人编号。")
    speaker = str(value).strip()
    if not speaker:
        raise TranscriptError(f"{where}的说话人编号不能为空。")
    return speaker


def parse_transcript(
    payload: Any,
    *,
    label: str,
    require_nonempty: bool = False,
) -> Transcript:
    """Validate and normalise a transcript JSON value.

    Accepted shapes::

        [{"word": "好", "time_ms": 1200, "speaker": "S1"}, ...]
        {"words": [{"word": "好", "time_ms": 1200, "speaker": "S1"}, ...]}

    A few common aliases (``text``/``start_ms``/``speaker_id``) are accepted to
    make the upload page forgiving without weakening the stated contract.
    """
    if isinstance(payload, dict):
        raw_words = payload.get("words")
        if raw_words is None:
            raise TranscriptError(f"{label}必须是词数组，或包含 words 数组的对象。")
    elif isinstance(payload, list):
        raw_words = payload
    else:
        raise TranscriptError(f"{label}必须是 JSON 数组或包含 words 数组的对象。")

    if not isinstance(raw_words, list):
        raise TranscriptError(f"{label}的 words 必须是数组。")
    if len(raw_words) > MAX_WORDS:
        raise TranscriptError(
            f"{label}有 {len(raw_words)} 个词，超过单稿上限 {MAX_WORDS} 词。"
        )
    if require_nonempty and not raw_words:
        raise TranscriptError("人工稿不能为空。")

    words: list[Word] = []
    for index, item in enumerate(raw_words, start=1):
        if not isinstance(item, dict):
            raise TranscriptError(f"{label}第 {index} 个词必须是对象。")
        text = _pick(item, ("word", "text"))
        if not isinstance(text, str) or not text.strip():
            raise TranscriptError(f"{label}第 {index} 个词缺少非空的 word 文本。")
        time_value = _pick(item, ("time_ms", "start_ms", "start", "timestamp_ms"))
        if time_value is None:
            raise TranscriptError(f"{label}第 {index} 个词缺少毫秒时间。")
        time_ms = _parse_time(time_value, index, label)
        speaker_value = _pick(item, ("speaker", "speaker_id"))
        if speaker_value is None:
            raise TranscriptError(f"{label}第 {index} 个词缺少说话人编号。")
        speaker = _parse_speaker(speaker_value, index, label)
        words.append(
            Word(text=text.strip(), time_ms=time_ms, speaker=speaker, order=index)
        )

    speakers = {w.speaker for w in words}
    if len(speakers) > MAX_SPEAKERS:
        raise TranscriptError(
            f"{label}有 {len(speakers)} 个说话人，超过单稿上限 {MAX_SPEAKERS} 人。"
        )

    # "按时间顺序对词": normalise the upload to a stable time order.  Python's
    # sort is stable, so words with equal timestamps keep their uploaded order.
    ordered = tuple(sorted(words, key=lambda w: (w.time_ms, w.order)))
    grouped: dict[str, list[Word]] = {}
    for word in ordered:
        grouped.setdefault(word.speaker, []).append(word)

    speaker_tuple = tuple(sorted(grouped, key=natural_speaker_key))
    grouped_tuple = {name: tuple(grouped[name]) for name in speaker_tuple}
    return Transcript(
        label=label,
        words=ordered,
        speaker_words=grouped_tuple,
        speakers=speaker_tuple,
    )


def _forward_row(
    a_codes: np.ndarray,
    a_times: np.ndarray,
    b_codes: np.ndarray,
    b_times: np.ndarray,
) -> np.ndarray:
    """Return the final edit-distance row for a -> b.

    Only a rolling row of length ``len(b) + 1`` is allocated.  The horizontal
    recurrence ``row[j] = min(row[j], row[j - 1] + 1)`` is evaluated with a
    prefix minimum, which keeps the per-word work vectorised.
    """
    m = len(b_codes)
    # ``idx`` is the "one more inserted word" penalty per column, in the same
    # scaled units as the rest of the row.
    idx = np.arange(m + 1, dtype=np.int64) * _TIME_WEIGHT
    prev = idx.copy()
    if m == 0:
        return np.array([len(a_codes) * _TIME_WEIGHT], dtype=np.int64)

    for code, time_ms in zip(a_codes, a_times):
        # int64 keeps the 500 ms window exact for arbitrarily large timestamps.
        delta = np.abs(b_times - int(time_ms))
        valid = delta <= MAX_ALIGN_GAP_MS
        sub_cost = np.where(b_codes != code, _TIME_WEIGHT, 0) + np.minimum(
            delta, MAX_ALIGN_GAP_MS
        )
        sub = np.where(valid, sub_cost, _INF).astype(np.int64)
        cur = np.empty(m + 1, dtype=np.int64)
        cur[0] = prev[0] + _TIME_WEIGHT
        np.minimum(prev[1:] + _TIME_WEIGHT, prev[:-1] + sub, out=cur[1:])
        # Prefix-minimum trick for the sequential "insert one more b word"
        # recurrence.  After subtracting the per-column penalty, a running
        # minimum is equivalent to propagating row[j - 1] + penalty rightwards.
        cur -= idx
        np.minimum.accumulate(cur, out=cur)
        cur += idx
        prev = cur
    return prev


def _distance(scaled_cost: int) -> int:
    return int(scaled_cost) // _TIME_WEIGHT


def pair_cost(ref_codes: np.ndarray, ref_times: np.ndarray,
              hyp_codes: np.ndarray, hyp_times: np.ndarray) -> int:
    """Minimum unit-cost edit distance with the 500 ms alignment limit.

    The returned value is the number of edits; timing deviation is used only
    as a tie-break between alignments with the same edit count.
    """
    if len(ref_codes) == 0:
        return len(hyp_codes)
    if len(hyp_codes) == 0:
        return len(ref_codes)
    return _distance(_forward_row(ref_codes, ref_times, hyp_codes, hyp_times)[-1])


def _greedy_feasible_slots(
    times_a: Sequence[int] | np.ndarray,
    times_b: Sequence[int] | np.ndarray,
    gap_ms: int = MAX_ALIGN_GAP_MS,
) -> int:
    """Maximum number of disjoint pairs with ``|ta - tb| <= gap_ms``.

    Both sequences must be sorted in ascending time order.  The earliest
    feasible choice is optimal for ordered points on a line, so a two-pointer
    scan computes the maximum cardinality in O(len(a) + len(b)) time and
    O(1) extra memory.
    """
    i = j = 0
    count = 0
    n_a, n_b = len(times_a), len(times_b)
    while i < n_a and j < n_b:
        time_a = int(times_a[i])
        time_b = int(times_b[j])
        delta = time_a - time_b
        if -gap_ms <= delta <= gap_ms:
            count += 1
            i += 1
            j += 1
        elif delta < 0:
            # time_a is too early for time_b and every later b word.
            i += 1
        else:
            # time_b is too early for time_a and every later a word.
            j += 1
    return count


def _code_time_index(
    codes: np.ndarray, times: np.ndarray
) -> dict[int, list[int]]:
    """Group word times by encoded vocabulary id (lists stay time-sorted)."""
    index: dict[int, list[int]] = {}
    for code, time_ms in zip(codes.tolist(), times.tolist()):
        index.setdefault(int(code), []).append(int(time_ms))
    return index


def pair_lower_bound(
    ref_codes: np.ndarray,
    ref_times: np.ndarray,
    hyp_codes: np.ndarray,
    hyp_times: np.ndarray,
    ref_code_times: Mapping[int, Sequence[int]] | None = None,
    hyp_code_times: Mapping[int, Sequence[int]] | None = None,
) -> int:
    """Cheap, provably safe lower bound for :func:`pair_cost`.

    For an alignment of ``n`` ref and ``m`` hyp words with ``D`` diagonals and
    ``M`` exact matches among them, the edit count is ``n + m - D - M``.  We
    bound ``D`` by the maximum number of time-feasible disjoint pairs and
    ``M`` by the same greedy matching restricted to equal word texts.  Both
    maxima are computed with linear two-pointer scans, so the result never
    exceeds the exact edit distance.
    """
    n, m = len(ref_codes), len(hyp_codes)
    diagonal_bound = _greedy_feasible_slots(ref_times, hyp_times)
    match_bound = 0
    if ref_code_times is not None and hyp_code_times is not None:
        for code, ref_slot_times in ref_code_times.items():
            hyp_slot_times = hyp_code_times.get(code)
            if hyp_slot_times:
                match_bound += _greedy_feasible_slots(
                    ref_slot_times, hyp_slot_times
                )
    return max(0, n + m - diagonal_bound - match_bound)


class LazyPairCosts(Mapping[tuple[str, str], int]):
    """Exact pair costs computed on demand and cached for one upload.

    Reads through the plain ``Mapping`` interface force an exact word-level DP
    (this keeps :func:`solve_pairing` correct when called directly).  The
    certifying solver instead reads :meth:`optimistic_cost`, which returns the
    lower bound until the exact cost has been computed.
    """

    def __init__(
        self,
        edges: Sequence[tuple[str, str]],
        enc_ref: Mapping[str, tuple[np.ndarray, np.ndarray]],
        enc_hyp: Mapping[str, tuple[np.ndarray, np.ndarray]],
        code_times_ref: Mapping[str, Mapping[int, Sequence[int]]],
        code_times_hyp: Mapping[str, Mapping[int, Sequence[int]]],
        lower_bounds: Mapping[tuple[str, str], int],
    ) -> None:
        self._edges = tuple(edges)
        self._enc_ref = enc_ref
        self._enc_hyp = enc_hyp
        self._code_times_ref = code_times_ref
        self._code_times_hyp = code_times_hyp
        self._lower = dict(lower_bounds)
        self._exact: dict[tuple[str, str], int] = {}
        # Speaker combinations for which the word-level DP has actually run.
        self.scored_pairs: set[tuple[str, str]] = set()

    # ------------------------------------------------------------- bounds
    def lower_bound(self, ref_speaker: str, hyp_speaker: str) -> int:
        return self._lower[(ref_speaker, hyp_speaker)]

    def is_exact(self, ref_speaker: str, hyp_speaker: str) -> bool:
        return (ref_speaker, hyp_speaker) in self._exact

    def optimistic_cost(self, ref_speaker: str, hyp_speaker: str) -> int:
        edge = (ref_speaker, hyp_speaker)
        cached = self._exact.get(edge)
        if cached is not None:
            return cached
        return self._lower[edge]

    # -------------------------------------------------------------- exact
    def exact_cost(self, ref_speaker: str, hyp_speaker: str) -> int:
        edge = (ref_speaker, hyp_speaker)
        cached = self._exact.get(edge)
        if cached is not None:
            return cached
        ref_codes, ref_times = self._enc_ref[ref_speaker]
        hyp_codes, hyp_times = self._enc_hyp[hyp_speaker]
        # Resolve through the module global so tests can monkeypatch
        # ``app.engine.pair_cost`` and observe every combination scored.
        value = pair_cost(ref_codes, ref_times, hyp_codes, hyp_times)
        self._exact[edge] = value
        self.scored_pairs.add(edge)
        return value

    @property
    def exact_costs(self) -> Mapping[tuple[str, str], int]:
        return self._exact

    # ----------------------------------------------------- Mapping protocol
    def __getitem__(self, edge: tuple[str, str]) -> int:
        ref_speaker, hyp_speaker = edge
        if edge in self._exact:
            return self._exact[edge]
        return self.exact_cost(ref_speaker, hyp_speaker)

    def __iter__(self):
        return iter(self._edges)

    def __len__(self) -> int:
        return len(self._edges)


class _OptimisticCostView(Mapping[tuple[str, str], int]):
    """Cost mapping that serves lower bounds for not-yet-scored pairs."""

    def __init__(self, lazy: LazyPairCosts) -> None:
        self._lazy = lazy

    def __getitem__(self, edge: tuple[str, str]) -> int:
        return self._lazy.optimistic_cost(*edge)

    def __iter__(self):
        return iter(self._lazy)

    def __len__(self) -> int:
        return len(self._lazy)


def _backward_row(
    a_codes: np.ndarray,
    a_times: np.ndarray,
    b_codes: np.ndarray,
    b_times: np.ndarray,
) -> np.ndarray:
    """Suffix costs, returned in the original column order."""
    return _forward_row(
        a_codes[::-1], a_times[::-1], b_codes[::-1], b_times[::-1]
    )[::-1]


def _small_case(
    a_codes: np.ndarray,
    a_times: np.ndarray,
    b_codes: np.ndarray,
    b_times: np.ndarray,
    a_offset: int,
    b_offset: int,
    ops: list[tuple[str, int | None, int | None]],
) -> None:
    """Exact base case for one a-word or one b-word."""
    n, m = len(a_codes), len(b_codes)
    if n == 0:
        ops.extend((_OPS_INS, None, b_offset + j) for j in range(m))
        return
    if m == 0:
        ops.extend((_OPS_DEL, a_offset + i, None) for i in range(n))
        return

    if n == 1:
        a_code = int(a_codes[0])
        a_time = int(a_times[0])
        delta = np.abs(b_times - a_time)
        valid = delta <= MAX_ALIGN_GAP_MS
        candidates = np.flatnonzero(valid)
        if len(candidates) == 0:
            ops.append((_OPS_DEL, a_offset, None))
            ops.extend((_OPS_INS, None, b_offset + j) for j in range(m))
            return
        # Among equal-cost options, prefer a zero-cost match, then the
        # smallest timing deviation.  Ties keep the earliest column.
        equal = candidates[b_codes[candidates] == a_code]
        pool = equal if len(equal) else candidates
        k = int(pool[np.argmin(delta[pool])])
        ops.extend((_OPS_INS, None, b_offset + j) for j in range(k))
        ops.append((_OPS_MATCH if int(b_codes[k]) == a_code else _OPS_SUB,
                    a_offset, b_offset + k))
        ops.extend((_OPS_INS, None, b_offset + j) for j in range(k + 1, m))
        return

    # m == 1, n > 1
    b_code = int(b_codes[0])
    b_time = int(b_times[0])
    delta = np.abs(a_times - b_time)
    valid = delta <= MAX_ALIGN_GAP_MS
    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        ops.extend((_OPS_DEL, a_offset + i, None) for i in range(n))
        ops.append((_OPS_INS, None, b_offset))
        return
    equal = candidates[a_codes[candidates] == b_code]
    pool = equal if len(equal) else candidates
    k = int(pool[np.argmin(delta[pool])])
    ops.extend((_OPS_DEL, a_offset + i, None) for i in range(k))
    ops.append((_OPS_MATCH if int(a_codes[k]) == b_code else _OPS_SUB,
                a_offset + k, b_offset))
    ops.extend((_OPS_DEL, a_offset + i, None) for i in range(k + 1, n))


def _hirschberg(
    a_codes: np.ndarray,
    a_times: np.ndarray,
    b_codes: np.ndarray,
    b_times: np.ndarray,
    a_offset: int,
    b_offset: int,
    ops: list[tuple[str, int | None, int | None]],
) -> None:
    n, m = len(a_codes), len(b_codes)
    if n == 0:
        ops.extend((_OPS_INS, None, b_offset + j) for j in range(m))
        return
    if m == 0:
        ops.extend((_OPS_DEL, a_offset + i, None) for i in range(n))
        return
    if n == 1 or m == 1:
        _small_case(a_codes, a_times, b_codes, b_times,
                    a_offset, b_offset, ops)
        return

    mid = n // 2
    left = _forward_row(a_codes[:mid], a_times[:mid], b_codes, b_times)
    right = _backward_row(a_codes[mid:], a_times[mid:], b_codes, b_times)
    split = int(np.argmin(left.astype(np.int64) + right.astype(np.int64)))
    # Drop the row arrays before recursing so peak extra memory stays O(m).
    del left, right

    _hirschberg(a_codes[:mid], a_times[:mid], b_codes[:split],
                b_times[:split], a_offset, b_offset, ops)
    _hirschberg(a_codes[mid:], a_times[mid:], b_codes[split:],
                b_times[split:], a_offset + mid, b_offset + split, ops)


def align_pair(
    ref_words: Sequence[Word],
    hyp_words: Sequence[Word],
) -> list[tuple[str, int | None, int | None]]:
    """Align one paired speaker's words, returning primitive operations."""
    # Local vocabulary is enough here and avoids retaining large object arrays.
    vocab: dict[str, int] = {}
    ref_codes = np.fromiter(
        (vocab.setdefault(w.text, len(vocab)) for w in ref_words),
        dtype=np.int32,
        count=len(ref_words),
    )
    hyp_codes = np.fromiter(
        (vocab.setdefault(w.text, len(vocab)) for w in hyp_words),
        dtype=np.int32,
        count=len(hyp_words),
    )
    ref_times = np.fromiter(
        (w.time_ms for w in ref_words), dtype=np.int64, count=len(ref_words)
    )
    hyp_times = np.fromiter(
        (w.time_ms for w in hyp_words), dtype=np.int64, count=len(hyp_words)
    )
    ops: list[tuple[str, int | None, int | None]] = []
    _hirschberg(ref_codes, ref_times, hyp_codes, hyp_times, 0, 0, ops)
    return ops


def _encode_speakers(
    ref: Transcript,
    hyp: Transcript,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, dict[int, list[int]]],
    dict[str, dict[int, list[int]]],
]:
    """Encode word texts as integers once for all pair cost computations."""
    vocab: dict[str, int] = {}

    def encode(words: Sequence[Word]) -> tuple[np.ndarray, np.ndarray]:
        codes = np.fromiter(
            (vocab.setdefault(w.text, len(vocab)) for w in words),
            dtype=np.int32,
            count=len(words),
        )
        times = np.fromiter(
            (w.time_ms for w in words), dtype=np.int64, count=len(words)
        )
        return codes, times

    ref_enc = {name: encode(ref.speaker_words[name]) for name in ref.speakers}
    hyp_enc = {name: encode(hyp.speaker_words[name]) for name in hyp.speakers}
    ref_code_times = {
        name: _code_time_index(codes, times)
        for name, (codes, times) in ref_enc.items()
    }
    hyp_code_times = {
        name: _code_time_index(codes, times)
        for name, (codes, times) in hyp_enc.items()
    }
    return ref_enc, hyp_enc, ref_code_times, hyp_code_times


def _validate_locks(
    locks: Mapping[str, str],
    ref: Transcript,
    hyp: Transcript,
) -> dict[str, str]:
    ref_set, hyp_set = set(ref.speakers), set(hyp.speakers)
    used_hyp: set[str] = set()
    clean: dict[str, str] = {}
    for ref_speaker, hyp_speaker in locks.items():
        if ref_speaker not in ref_set:
            raise ValueError(f"锁定引用了人工稿中不存在的说话人：{ref_speaker}")
        if hyp_speaker not in hyp_set:
            raise ValueError(f"锁定引用了识别稿中不存在的说话人：{hyp_speaker}")
        if hyp_speaker in used_hyp:
            raise ValueError(f"锁定不是一对一：识别稿说话人 {hyp_speaker} 被重复锁定")
        used_hyp.add(hyp_speaker)
        clean[ref_speaker] = hyp_speaker
    return clean


def _mapping_key(
    mapping: Sequence[str | None],
    ref_order: Sequence[str],
    hyp_order: Sequence[str],
) -> tuple:
    """Deterministic ordering for equally-priced assignments."""
    hyp_rank = {speaker: index for index, speaker in enumerate(hyp_order)}
    key = []
    for ref_speaker, hyp_speaker in zip(ref_order, mapping):
        if hyp_speaker is None:
            key.append((1, natural_speaker_key(ref_speaker), -1))
        else:
            key.append((0, natural_speaker_key(ref_speaker),
                        hyp_rank[hyp_speaker]))
    return tuple(key)


def build_context(ref: Transcript, hyp: Transcript) -> PairingContext:
    """Build the reusable scoring context without running all 8 x 8 DPs.

    Only linear-time lower bounds are computed here.  Exact pair costs are
    filled in lazily by :func:`solve_pairing_certified` (or on direct access
    through :attr:`PairingContext.pair_costs`).
    """
    ref_enc, hyp_enc, ref_code_times, hyp_code_times = _encode_speakers(
        ref, hyp
    )
    edges = [
        (ref_speaker, hyp_speaker)
        for ref_speaker in ref.speakers
        for hyp_speaker in hyp.speakers
    ]
    lower_bounds: dict[tuple[str, str], int] = {}
    for ref_speaker, hyp_speaker in edges:
        ref_codes, ref_times = ref_enc[ref_speaker]
        hyp_codes, hyp_times = hyp_enc[hyp_speaker]
        lower_bounds[(ref_speaker, hyp_speaker)] = pair_lower_bound(
            ref_codes,
            ref_times,
            hyp_codes,
            hyp_times,
            ref_code_times[ref_speaker],
            hyp_code_times[hyp_speaker],
        )
    costs = LazyPairCosts(
        edges,
        ref_enc,
        hyp_enc,
        ref_code_times,
        hyp_code_times,
        lower_bounds,
    )
    return PairingContext(
        ref=ref,
        hyp=hyp,
        pair_costs=costs,
        lower_bounds=lower_bounds,
        exact_costs=costs._exact,
        scored_pairs=costs.scored_pairs,
    )


def solve_pairing_certified(
    ref: Transcript,
    hyp: Transcript,
    locks: Mapping[str, str] | None = None,
    context: PairingContext | None = None,
) -> PairingSolution:
    """Return exactly the same solution as :func:`solve_pairing`, scoring only
    pairs that can take part in the optimum.

    Each not-yet-exact pair is represented by its provable lower bound and the
    unchanged assignment DP is run.  Every pair used by its winning mapping is
    then scored exactly and the DP is repeated.  When the winner uses only
    exact costs, the lower-bound inequality certifies it against the full
    matrix: for every assignment ``X``, ``exact(X) >= optimistic(X) >=
    optimistic(winner) == exact(winner)``.  Equal-cost assignments share the
    same deterministic tie-break key (pair count then stable speaker order),
    so ties are resolved identically to the eager algorithm.  At most one new
    combination per used edge is scored; the loop always terminates.
    """
    locks = dict(locks or {})
    if context is None:
        context = build_context(ref, hyp)

    costs = context.pair_costs
    if not isinstance(costs, LazyPairCosts):
        # Hand-built/exact mapping: nothing to certify, behave classically.
        return solve_pairing(ref, hyp, locks, context)

    forced = _validate_locks(locks, ref, hyp)
    view = _OptimisticCostView(costs)
    probe_context = SimpleNamespace(pair_costs=view)

    solution: PairingSolution | None = None
    while True:
        solution = solve_pairing(ref, hyp, locks, probe_context)
        needed = {
            (r, h)
            for r, h in forced.items()
            if not costs.is_exact(r, h)
        }
        needed.update(
            (r, h)
            for r, h in solution.pairs.items()
            if not costs.is_exact(r, h)
        )
        if not needed:
            return solution
        for ref_speaker, hyp_speaker in needed:
            costs.exact_cost(ref_speaker, hyp_speaker)


def solve_pairing(
    ref: Transcript,
    hyp: Transcript,
    locks: Mapping[str, str] | None = None,
    context: PairingContext | None = None,
) -> PairingSolution:
    """Find the global one-to-one speaker assignment with minimum errors.

    Locked pairs are compulsory.  Remaining speakers are assigned by a
    bitmask dynamic program over the (at most eight) hypothesis speakers.
    Tie-break, in order: fewest total errors, then most pairs, then the
    assignment that pairs the earlier reference speaker first.
    """

    locks = dict(locks or {})
    forced = _validate_locks(locks, ref, hyp)
    if context is None:
        context = build_context(ref, hyp)
    pair_costs = context.pair_costs

    used_hyp = set(forced.values())
    free_refs = [s for s in ref.speakers if s not in forced]
    free_hyps = [s for s in hyp.speakers if s not in used_hyp]
    ref_counts = {s: len(ref.speaker_words[s]) for s in ref.speakers}
    hyp_counts = {s: len(hyp.speaker_words[s]) for s in hyp.speakers}

    def unused_cost(mask: int) -> int:
        return sum(
            hyp_counts[speaker]
            for bit, speaker in enumerate(free_hyps)
            if not (mask & (1 << bit))
        )

    @functools.cache
    def best(i: int, mask: int) -> tuple[int, int, tuple[str | None, ...]]:
        if i == len(free_refs):
            return (unused_cost(mask), 0, ())
        ref_speaker = free_refs[i]

        # Leave this reference speaker unpaired (all its words are deletions).
        tail_cost, tail_pairs, tail_map = best(i + 1, mask)
        best_option = (
            tail_cost + ref_counts[ref_speaker],
            tail_pairs,
            (None,) + tail_map,
        )

        for bit, hyp_speaker in enumerate(free_hyps):
            if mask & (1 << bit):
                continue
            tail_cost, tail_pairs, tail_map = best(i + 1, mask | (1 << bit))
            option = (
                tail_cost + pair_costs[(ref_speaker, hyp_speaker)],
                tail_pairs + 1,
                (hyp_speaker,) + tail_map,
            )
            if _option_key(option, free_refs, free_hyps) < _option_key(
                best_option, free_refs, free_hyps
            ):
                best_option = option
        return best_option

    auto_cost, _auto_pairs, auto_map = best(0, 0)
    forced_cost = sum(pair_costs[(r, h)] for r, h in forced.items())
    total = forced_cost + auto_cost

    pairs = dict(forced)
    for ref_speaker, hyp_speaker in zip(free_refs, auto_map):
        if hyp_speaker is not None:
            pairs[ref_speaker] = hyp_speaker
    paired_refs = set(pairs)
    paired_hyps = set(pairs.values())
    return PairingSolution(
        pairs=pairs,
        unpaired_ref=tuple(s for s in ref.speakers if s not in paired_refs),
        unpaired_hyp=tuple(s for s in hyp.speakers if s not in paired_hyps),
        total_errors=int(total),
        pair_costs=pair_costs,
    )


def _option_key(
    option: tuple[int, int, tuple[str | None, ...]],
    ref_order: Sequence[str],
    hyp_order: Sequence[str],
) -> tuple:
    cost, pairs, mapping = option
    # Fewest errors first.  Equal-cost solutions prefer more pairs (a complete
    # one-to-one reading), then stable speaker order.
    return (cost, -pairs, _mapping_key(mapping, ref_order, hyp_order))


def _op_errors(
    ref_words: Sequence[Word],
    hyp_words: Sequence[Word],
    ops: Sequence[tuple[str, int | None, int | None]],
    ref_speaker: str,
    hyp_speaker: str,
) -> tuple[list[ErrorItem], int]:
    pair_label = f"{ref_speaker} → {hyp_speaker}"
    errors: list[ErrorItem] = []
    matches = 0
    sequence = 0
    for op, ref_index, hyp_index in ops:
        if op == _OPS_MATCH:
            matches += 1
            continue
        ref_word = ref_words[ref_index] if ref_index is not None else None
        hyp_word = hyp_words[hyp_index] if hyp_index is not None else None
        kind = {
            _OPS_SUB: _SUBSTITUTION,
            _OPS_DEL: _DELETION,
            _OPS_INS: _INSERTION,
        }[op]
        sort_ms = ref_word.time_ms if ref_word is not None else hyp_word.time_ms
        errors.append(ErrorItem(
            kind=kind,
            ref_word=ref_word,
            hyp_word=hyp_word,
            ref_speaker=ref_speaker,
            hyp_speaker=hyp_speaker,
            pair_label=pair_label,
            sort_ms=sort_ms,
            sequence=sequence,
        ))
        sequence += 1
    return errors, matches


def build_result(
    ref: Transcript,
    hyp: Transcript,
    locks: Mapping[str, str] | None = None,
    context: PairingContext | None = None,
) -> EngineResult:
    """Solve the pairing lazily and produce a display-ready result."""
    locks = dict(locks or {})
    solution = solve_pairing_certified(ref, hyp, locks, context)
    errors: list[ErrorItem] = []
    pair_rows: list[PairRow] = []
    sequence = 0

    for ref_speaker in ref.speakers:
        ref_words = ref.speaker_words[ref_speaker]
        hyp_speaker = solution.pairs.get(ref_speaker)
        if hyp_speaker is None:
            for word in ref_words:
                errors.append(ErrorItem(
                    kind=_DELETION,
                    ref_word=word,
                    hyp_word=None,
                    ref_speaker=ref_speaker,
                    hyp_speaker=None,
                    pair_label="未配对",
                    sort_ms=word.time_ms,
                    sequence=sequence,
                ))
                sequence += 1
            pair_rows.append(PairRow(
                ref_speaker=ref_speaker,
                hyp_speaker=None,
                ref_words=len(ref_words),
                hyp_words=0,
                matches=0,
                errors=len(ref_words),
            ))
            continue

        hyp_words = hyp.speaker_words[hyp_speaker]
        ops = align_pair(ref_words, hyp_words)
        pair_errors, matches = _op_errors(
            ref_words, hyp_words, ops, ref_speaker, hyp_speaker
        )
        expected = solution.pair_costs[(ref_speaker, hyp_speaker)]
        if len(pair_errors) != expected:
            raise RuntimeError(
                f"内部错误：{ref_speaker}→{hyp_speaker} 对齐回溯得到 "
                f"{len(pair_errors)} 个错误，但评分为 {expected}。"
            )
        for error in pair_errors:
            errors.append(ErrorItem(
                kind=error.kind,
                ref_word=error.ref_word,
                hyp_word=error.hyp_word,
                ref_speaker=error.ref_speaker,
                hyp_speaker=error.hyp_speaker,
                pair_label=error.pair_label,
                sort_ms=error.sort_ms,
                sequence=sequence,
            ))
            sequence += 1
        pair_rows.append(PairRow(
            ref_speaker=ref_speaker,
            hyp_speaker=hyp_speaker,
            ref_words=len(ref_words),
            hyp_words=len(hyp_words),
            matches=matches,
            errors=len(pair_errors),
        ))

    for hyp_speaker in solution.unpaired_hyp:
        hyp_words = hyp.speaker_words[hyp_speaker]
        for word in hyp_words:
            errors.append(ErrorItem(
                kind=_INSERTION,
                ref_word=None,
                hyp_word=word,
                ref_speaker=None,
                hyp_speaker=hyp_speaker,
                pair_label="未配对",
                sort_ms=word.time_ms,
                sequence=sequence,
            ))
            sequence += 1
        pair_rows.append(PairRow(
            ref_speaker="—",
            hyp_speaker=hyp_speaker,
            ref_words=0,
            hyp_words=len(hyp_words),
            matches=0,
            errors=len(hyp_words),
        ))

    errors.sort(key=lambda e: (e.sort_ms, e.sequence))
    counts = {
        _SUBSTITUTION: sum(1 for e in errors if e.kind == _SUBSTITUTION),
        _DELETION: sum(1 for e in errors if e.kind == _DELETION),
        _INSERTION: sum(1 for e in errors if e.kind == _INSERTION),
    }
    total = len(errors)
    if total != solution.total_errors:
        raise RuntimeError(
            f"内部错误：回溯错误总数 {total} 与最少错误 {solution.total_errors} 不一致。"
        )
    ref_count = ref.word_count
    return EngineResult(
        ref_word_count=ref_count,
        hyp_word_count=hyp.word_count,
        total_errors=total,
        wer=(total / ref_count) if ref_count else 0.0,
        substitution_count=counts[_SUBSTITUTION],
        deletion_count=counts[_DELETION],
        insertion_count=counts[_INSERTION],
        pairs=tuple(pair_rows),
        errors=tuple(errors),
        lock_signature=tuple(sorted(locks.items())),
        unpaired_ref=solution.unpaired_ref,
        unpaired_hyp=solution.unpaired_hyp,
    )
