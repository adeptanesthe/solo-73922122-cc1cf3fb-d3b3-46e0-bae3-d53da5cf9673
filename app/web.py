"""Flask web application for transcript acceptance review."""

from __future__ import annotations

import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

from .engine import (
    MAX_ALIGN_GAP_MS,
    MAX_SPEAKERS,
    MAX_WORDS,
    EngineResult,
    PairingContext,
    Transcript,
    TranscriptError,
    build_context,
    build_result,
    parse_transcript,
)

MAX_UPLOAD_BYTES = 4 * 1024 * 1024

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
SAMPLE_SETS = {
    "meeting": {
        "title": "产品周会（3 人对 4 人，含身份互换与未配对说话人）",
        "ref": "meeting_ref.json",
        "hyp": "meeting_hyp.json",
    },
    "sparse": {
        "title": "预算沟通（含未配对人工说话人与超过 500 ms 的错位）",
        "ref": "sparse_ref.json",
        "hyp": "sparse_hyp.json",
    },
}
ALLOWED_SAMPLES = {
    name for entry in SAMPLE_SETS.values() for name in (entry["ref"], entry["hyp"])
}


@dataclass
class ReviewSession:
    """All mutable state for one browser session.

    A new immutable ``snapshot`` is built on every successful upload or lock
    change.  The rendered page reads one snapshot only, so the score and the
    error list can never come from different revisions of the state.
    """

    ref: Transcript
    hyp: Transcript
    locks: dict[str, str] = field(default_factory=dict)
    context: PairingContext | None = None
    snapshot: ReviewSnapshot | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    ref_filename: str = ""
    hyp_filename: str = ""


@dataclass(frozen=True)
class ReviewSnapshot:
    result: EngineResult
    locks: tuple[tuple[str, str], ...]
    revision: int


_sessions: dict[str, ReviewSession] = {}
_sessions_guard = threading.Lock()
_MAX_SESSIONS = 500


def _session_key() -> str:
    key = session.get("review_id")
    if not key:
        key = secrets.token_urlsafe(16)
        session["review_id"] = key
        session.permanent = True
    return key


def _get_session() -> ReviewSession | None:
    key = session.get("review_id")
    if not key:
        return None
    with _sessions_guard:
        return _sessions.get(key)


def _put_session(store: ReviewSession) -> None:
    key = _session_key()
    with _sessions_guard:
        if len(_sessions) >= _MAX_SESSIONS and key not in _sessions:
            # Evict the oldest insertion; this only bounds accidental memory
            # growth, it is not a per-user quota.
            first = next(iter(_sessions))
            _sessions.pop(first, None)
        _sessions[key] = store


def _load_json_upload(field_name: str, label: str) -> tuple[Any, str]:
    file = request.files.get(field_name)
    if file is None or not file.filename:
        raise TranscriptError(f"请选择{label}文件。")
    payload = file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise TranscriptError(
            f"{label}文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上传上限。"
        )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise TranscriptError(f"{label}文件必须是 UTF-8 编码的 JSON。") from None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TranscriptError(
            f"{label}不是合法 JSON（第 {exc.lineno} 行第 {exc.colno} 列）。"
        ) from None
    return value, file.filename


def _parse_locks_from_form() -> list[tuple[str, str | None]]:
    """Parse posting into (ref speaker, hyp speaker-or-None) instructions.

    The table buttons post ``lock_<refSpeaker>``; the manual form posts
    ``manual_mode=pair|unlock`` with the two select values.  Both shapes are
    plain HTML form fields, so the page works without JavaScript.
    """
    instructions: list[tuple[str, str | None]] = []
    for key, value in request.form.items():
        if key.startswith("lock_"):
            ref_speaker = key[len("lock_"):]
            hyp_speaker = value.strip()
            instructions.append((ref_speaker, hyp_speaker or None))
    mode = request.form.get("manual_mode")
    if mode == "pair":
        ref_speaker = request.form.get("manual_ref", "").strip()
        hyp_speaker = request.form.get("manual_hyp", "").strip()
        if ref_speaker and hyp_speaker:
            instructions.append((ref_speaker, hyp_speaker))
    elif mode == "unlock":
        ref_speaker = request.form.get("manual_unlock", "").strip()
        if ref_speaker:
            instructions.append((ref_speaker, None))
    return instructions


def validate_lock_change(
    ref: Transcript,
    hyp: Transcript,
    locks: dict[str, str],
    ref_speaker: str,
    hyp_speaker: str,
) -> str | None:
    """Return a human-readable conflict message, or None when acceptable."""
    if ref_speaker not in ref.speakers:
        return f"人工稿中不存在说话人 {ref_speaker}。"
    if hyp_speaker not in hyp.speakers:
        return f"识别稿中不存在说话人 {hyp_speaker}。"
    other_ref = next(
        (r for r, h in locks.items() if h == hyp_speaker and r != ref_speaker),
        None,
    )
    if other_ref is not None:
        return (
            f"不能锁定：识别稿说话人 {hyp_speaker} 已锁定给人工稿说话人 "
            f"{other_ref}。请先撤销那一处锁定。"
        )
    return None


def _rebuild(store: ReviewSession) -> ReviewSnapshot:
    if store.context is None:
        store.context = build_context(store.ref, store.hyp)
    result = build_result(store.ref, store.hyp, store.locks, store.context)
    revision = (store.snapshot.revision + 1) if store.snapshot else 1
    store.snapshot = ReviewSnapshot(
        result=result,
        locks=tuple(sorted(store.locks.items())),
        revision=revision,
    )
    return store.snapshot


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("APP_SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=2 * MAX_UPLOAD_BYTES,
        JSON_SORT_KEYS=False,
    )
    if test_config:
        app.config.update(test_config)

    @app.before_request
    def reject_cross_site_form_posts():
        # Session state is mutated by plain form posts, so require same-origin
        # submissions.  Same-origin fetches from our own page send Origin.
        if request.method != "POST":
            return
        origin = request.headers.get("Origin")
        if origin is None:
            referer = request.headers.get("Referer")
            if referer is None:
                # Non-browser client (tests, curl): it has no ambient cookie
                # baggage to abuse, so let it through.
                return
            referer_parts = referer.split("/", 3)
            if len(referer_parts) < 3:
                abort(403, description="无法校验提交来源。")
            origin = referer_parts[0] + "//" + referer_parts[2]
        if origin.rstrip("/") != request.host_url.rstrip("/"):
            abort(403, description="跨站表单提交已被拒绝。")

    @app.context_processor
    def inject_limits() -> dict[str, Any]:
        return {
            "MAX_WORDS": MAX_WORDS,
            "MAX_SPEAKERS": MAX_SPEAKERS,
            "MAX_ALIGN_GAP_MS": MAX_ALIGN_GAP_MS,
            "SAMPLE_SETS": SAMPLE_SETS,
        }

    @app.get("/")
    def index():
        store = _get_session()
        if store is None:
            return render_template(
                "index.html", snapshot=None, error=None, notice=None,
                ref_name=None, hyp_name=None, ref_meta=None, hyp_meta=None,
                locks={}, lock_error=None,
            )
        with store.lock:
            snapshot = store.snapshot
            if snapshot is None:
                snapshot = _rebuild(store)
        return render_template(
            "index.html",
            snapshot=snapshot,
            error=None,
            notice=None,
            ref_name=store.ref_filename,
            hyp_name=store.hyp_filename,
            ref_meta=store.ref,
            hyp_meta=store.hyp,
            locks=dict(snapshot.locks),
            lock_error=None,
        )

    def _load_sample(name: str) -> Any:
        with open(os.path.join(SAMPLE_DIR, name), encoding="utf-8") as handle:
            return json.load(handle)

    @app.post("/upload")
    def upload():
        try:
            ref_payload, ref_name = _load_json_upload("ref_file", "人工稿")
            hyp_payload, hyp_name = _load_json_upload("hyp_file", "识别稿")
            ref = parse_transcript(ref_payload, label="人工稿", require_nonempty=True)
            hyp = parse_transcript(hyp_payload, label="识别稿")
        except TranscriptError as exc:
            return _upload_error(str(exc))

        store = ReviewSession(
            ref=ref,
            hyp=hyp,
            ref_filename=ref_name,
            hyp_filename=hyp_name,
        )
        try:
            snapshot = _rebuild(store)
        except Exception as exc:  # pragma: no cover - defensive
            app.logger.exception("build_result failed")
            return _upload_error(f"评分失败：{exc}", status=500)
        _put_session(store)
        return _render_upload_page(store, snapshot, notice="已载入两稿并完成最少错误配对。")

    def _upload_error(message: str, status: int = 400):
        return (
            render_template(
                "index.html",
                snapshot=None,
                error=message,
                notice=None,
                ref_name=None,
                hyp_name=None,
                ref_meta=None,
                hyp_meta=None,
                locks={},
                lock_error=None,
            ),
            status,
        )

    def _render_upload_page(store: ReviewSession, snapshot: ReviewSnapshot,
                            *, notice: str | None = None,
                            lock_error: str | None = None,
                            status: int = 200):
        # ``locks`` comes from the snapshot, never from live session state, so
        # the lock buttons always agree with the score and errors rendered.
        return (
            render_template(
                "index.html",
                snapshot=snapshot,
                error=None,
                notice=notice,
                ref_name=store.ref_filename,
                hyp_name=store.hyp_filename,
                ref_meta=store.ref,
                hyp_meta=store.hyp,
                locks=dict(snapshot.locks),
                lock_error=lock_error,
            ),
            status,
        )

    @app.post("/locks")
    def update_locks():
        store = _get_session()
        if store is None:
            abort(400, description="请先上传两稿。")
        instructions = _parse_locks_from_form()
        if not instructions:
            with store.lock:
                snapshot = store.snapshot or _rebuild(store)
            return _render_upload_page(
                store, snapshot,
                lock_error="没有收到锁定操作。", status=400,
            )

        with store.lock:
            if store.snapshot is None:
                _rebuild(store)
            # Work on a copy and only commit when every instruction is valid.
            # This preserves the previous locks when a conflicting lock arrives.
            candidate = dict(store.locks)
            conflicts: list[str] = []
            for ref_speaker, hyp_speaker in instructions:
                if hyp_speaker is None:
                    candidate.pop(ref_speaker, None)
                    continue
                # The ref speaker may move to a new hyp speaker (its own old
                # lock is replaced), but it must not steal a hyp speaker that
                # another ref speaker currently holds.
                message = validate_lock_change(
                    store.ref, store.hyp, candidate, ref_speaker, hyp_speaker
                )
                if message:
                    conflicts.append(message)
                    continue
                candidate[ref_speaker] = hyp_speaker
            if conflicts:
                return _render_upload_page(
                    store, store.snapshot,
                    lock_error=" ".join(conflicts) + " 原锁定未被改动。",
                    status=409,
                )
            if candidate == store.locks:
                return _render_upload_page(
                    store, store.snapshot,
                    notice="锁定没有变化。",
                )
            old_locks, old_snapshot = store.locks, store.snapshot
            store.locks = candidate
            try:
                snapshot = _rebuild(store)
            except Exception:
                # Never expose a score that does not match valid locks.
                store.locks = old_locks
                store.snapshot = old_snapshot
                raise
            return _render_upload_page(
                store, snapshot,
                notice="锁定已更新，评分与错词已按最新有效锁定重算。",
            )

    @app.post("/sample")
    def load_sample():
        name = request.form.get("sample", "")
        spec = SAMPLE_SETS.get(name)
        if spec is None:
            abort(404)
        try:
            ref_payload = _load_sample(spec["ref"])
            hyp_payload = _load_sample(spec["hyp"])
            ref = parse_transcript(ref_payload, label="人工稿", require_nonempty=True)
            hyp = parse_transcript(hyp_payload, label="识别稿")
        except (OSError, TranscriptError) as exc:
            app.logger.exception("sample load failed")
            return _upload_error(f"示例稿载入失败：{exc}", status=500)
        store = ReviewSession(
            ref=ref,
            hyp=hyp,
            ref_filename=spec["ref"],
            hyp_filename=spec["hyp"],
        )
        try:
            snapshot = _rebuild(store)
        except Exception as exc:  # pragma: no cover - defensive
            app.logger.exception("build_result failed")
            return _upload_error(f"示例稿评分失败：{exc}", status=500)
        _put_session(store)
        return _render_upload_page(
            store, snapshot, notice=f"已载入示例：{spec['title']}。"
        )

    @app.get("/samples/<path:name>")
    def download_sample(name: str):
        if name not in ALLOWED_SAMPLES:
            abort(404)
        return send_from_directory(
            SAMPLE_DIR, name, as_attachment=True, mimetype="application/json"
        )

    @app.post("/reset")
    def reset():
        key = session.get("review_id")
        if key:
            with _sessions_guard:
                _sessions.pop(key, None)
        return redirect(url_for("index"))

    @app.get("/api/result")
    def api_result():
        store = _get_session()
        if store is None:
            return jsonify({"error": "尚未上传稿件。"}), 404
        with store.lock:
            snapshot = store.snapshot
        if snapshot is None:
            return jsonify({"error": "尚未完成评分。"}), 409
        return jsonify(serialize_snapshot(snapshot))

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    @app.errorhandler(413)
    def too_large(_error):
        limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return (
            render_template(
                "index.html",
                snapshot=None,
                error=f"上传内容超过 {limit_mb} MB 总上限。",
                notice=None,
                ref_name=None,
                hyp_name=None,
                ref_meta=None,
                hyp_meta=None,
                locks={},
                lock_error=None,
            ),
            413,
        )

    return app


def serialize_snapshot(snapshot: ReviewSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {"error": "尚未完成评分。"}
    result = snapshot.result
    return {
        "revision": snapshot.revision,
        "locks": [
            {"ref_speaker": r, "hyp_speaker": h} for r, h in snapshot.locks
        ],
        "ref_word_count": result.ref_word_count,
        "hyp_word_count": result.hyp_word_count,
        "total_errors": result.total_errors,
        "wer": result.wer,
        "substitution_count": result.substitution_count,
        "deletion_count": result.deletion_count,
        "insertion_count": result.insertion_count,
        "pairs": [
            {
                "ref_speaker": pair.ref_speaker,
                "hyp_speaker": pair.hyp_speaker,
                "ref_words": pair.ref_words,
                "hyp_words": pair.hyp_words,
                "matches": pair.matches,
                "errors": pair.errors,
            }
            for pair in result.pairs
        ],
        "unpaired_ref": list(result.unpaired_ref),
        "unpaired_hyp": list(result.unpaired_hyp),
        "errors": [
            {
                "kind": error.kind,
                "ref_speaker": error.ref_speaker,
                "hyp_speaker": error.hyp_speaker,
                "pair_label": error.pair_label,
                "ref_word": error.ref_word.text if error.ref_word else None,
                "ref_time_ms": error.ref_word.time_ms if error.ref_word else None,
                "hyp_word": error.hyp_word.text if error.hyp_word else None,
                "hyp_time_ms": error.hyp_word.time_ms if error.hyp_word else None,
                "sort_ms": error.sort_ms,
            }
            for error in result.errors
        ],
    }
