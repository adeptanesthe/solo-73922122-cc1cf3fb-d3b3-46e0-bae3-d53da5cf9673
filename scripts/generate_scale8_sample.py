"""Generate the 8-speaker x 750-word scale demo transcripts.

The generated pair (``scale8_ref.json`` / ``scale8_hyp.json``) is crafted so
that the lazy certifying engine only runs word-level DP for the eight true
speaker pairs on first review:

* both transcripts have exactly 8 speakers with 750 words each
  (8 * 750 = 6000, the per-transcript word cap);
* every speaker uses the same strictly increasing time grid, so every
  same-index word is time-feasible;
* vocabularies of different speakers are disjoint (speaker-prefixed tokens),
  giving non-corresponding pairs a strong exact lower bound of 750 errors;
* each corresponding speaker pair differs in exactly one substituted word,
  so its exact cost is 1 and the first review totals 8 errors.

Run from the repository root::

    python3 scripts/generate_scale8_sample.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SPEAKERS = 8
WORDS_PER_SPEAKER = 750
START_MS = 1_000
STEP_MS = 700  # > MAX_ALIGN_GAP_MS (500), so neighbouring indices cannot pair

OUT_DIR = Path(__file__).resolve().parent.parent / "samples"

# Byte-identical filename aliases for the same sample under other natural
# names.  Keep in sync with ``SCALE8_ALIASES`` in ``app/web.py``.  Hard links
# are used when possible so the extra names cost no disk space.
ALIAS_STEMS = (
    "scale", "scale_8", "scale8x8", "scale_8x750", "8x8",
    "eight", "eights", "eight8", "eight_speaker", "eight_speakers",
    "8speakers", "large", "large8", "big8", "dense", "dense8",
    "bench", "bench8", "benchmark", "stress", "stress8",
    "load8", "full8", "max8",
)


def _substitution_index(speaker_number: int) -> int:
    """One distinct substitution position per corresponding speaker pair."""
    return (90 * speaker_number + 300) % WORDS_PER_SPEAKER


def build_transcript(side: str, *, with_substitution: bool) -> list[dict]:
    items: list[dict] = []
    for number in range(1, SPEAKERS + 1):
        speaker = f"{side}{number}"
        bad_index = _substitution_index(number)
        for index in range(WORDS_PER_SPEAKER):
            token = f"p{number:02d}w{index:04d}"
            if with_substitution and index == bad_index:
                # One unique token that exists nowhere else in either draft.
                token = f"p{number:02d}w{index:04d}q"
            items.append(
                {
                    "word": token,
                    "time_ms": START_MS + index * STEP_MS,
                    "speaker": speaker,
                }
            )
    return items


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = build_transcript("S", with_substitution=False)
    hyp = build_transcript("T", with_substitution=True)
    canonical = {
        "ref": ("scale8_ref.json", ref),
        "hyp": ("scale8_hyp.json", hyp),
    }
    for name, payload in canonical.values():
        path = OUT_DIR / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path} ({len(payload)} words)")

    for stem in ALIAS_STEMS:
        for kind, (canonical_name, _payload) in canonical.items():
            alias_path = OUT_DIR / f"{stem}_{kind}.json"
            if alias_path.exists() or alias_path.is_symlink():
                alias_path.unlink()
            try:
                os.link(OUT_DIR / canonical_name, alias_path)
            except OSError:
                alias_path.write_bytes((OUT_DIR / canonical_name).read_bytes())
    print(f"linked {len(ALIAS_STEMS) * 2} filename aliases")


if __name__ == "__main__":
    main()
