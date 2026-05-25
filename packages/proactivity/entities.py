"""Speaker-aware entity retrieval for the proactivity pipeline.

memory.md grows monotonically — naively shoving the whole entity
gallery into Qwen and Claude every classify call balloons context to
multiple KB within minutes. Instead, this module extracts the
speakers in the live transcript, fuzzy-matches them against the
parsed entity gallery, and returns only the entities that match (plus
the wearer themselves, who is always relevant).

Fuzzy matching tolerates one-character spelling drift on names of
length ≥ 4 ("Alexis" vs "Alexia"), exact matches against entity
aliases (declared by an `- aliases: A, B` bullet inside an entity
block), and substring containment for hyphenated/spaced variants.
"""

from __future__ import annotations

from collections.abc import Iterable


def _normalize(s: str) -> str:
    """Lowercase, strip non-alphanumerics. 'Alexia S.' -> 'alexias'.

    Strips diacritics by simple ASCII filtering; that's enough for the
    English-name use case the classifier was trained on."""
    if not s:
        return ""
    return "".join(c.lower() for c in s if c.isalnum())


def _levenshtein(a: str, b: str) -> int:
    """Iterative edit distance. Used only on short names (≤ ~30 chars)
    so the O(n*m) cost is negligible."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _name_matches(candidates: Iterable[str], speaker: str) -> bool:
    """True if any of `candidates` matches `speaker` after normalization
    + fuzzy tolerance.

    Tolerance rules:
      - Exact normalized match (case/punctuation insensitive).
      - Substring containment when both sides are ≥ 4 chars.
      - 1-char Levenshtein on names of length ≥ 4 (catches 'Alexis'
        vs 'Alexia'). Short names skip fuzzy because 'Sam'/'Tom' are
        only 1 edit apart and would falsely merge.
    """
    s = _normalize(speaker)
    if not s:
        return False
    for name in candidates:
        n = _normalize(name)
        if not n:
            continue
        if n == s:
            return True
        if len(s) >= 4 and len(n) >= 4 and (s in n or n in s):
            return True
        if len(s) >= 4 and len(n) >= 4 and abs(len(s) - len(n)) <= 2 and _levenshtein(s, n) <= 1:
            return True
    return False


def extract_speakers(
    transcript: list[tuple[str | None, str]],
    *,
    user_name: str,
) -> list[str]:
    """Pull every distinct speaker label out of the transcript and add
    the wearer's name. Order preserved (wearer first, then turn order
    of first appearance) so downstream rendering is deterministic.

    Person N labels are kept as-is — they won't match any real entity
    (unless an alias exists), which is fine: callers fall through to
    the wearer-only case and Claude works from general knowledge."""
    seen: set[str] = set()
    out: list[str] = []
    if user_name:
        out.append(user_name)
        seen.add(user_name)
    for spk, _text in transcript:
        if spk and spk not in seen:
            out.append(spk)
            seen.add(spk)
    return out


def filter_entities_by_speakers(
    entity_list: list[dict],
    speakers: Iterable[str],
) -> list[dict]:
    """Return only entities whose name or alias matches at least one
    of the observed speakers. Order preserved (matches the input
    `entity_list` order, which puts the wearer first if present in
    memory.md)."""
    speaker_list = [s for s in speakers if s]
    if not speaker_list:
        return []
    out: list[dict] = []
    for e in entity_list:
        candidates = [e.get("name", "")] + list(e.get("aliases", []) or [])
        if any(_name_matches(candidates, s) for s in speaker_list):
            out.append(e)
    return out


def format_entities_compact(entities: list[dict]) -> str:
    """Render filtered entities as a tight one-per-line block for
    Claude's user content. Aliases are dropped here — they're for
    retrieval, not reasoning."""
    if not entities:
        return "(no relevant entity info)"
    lines = []
    for e in entities:
        rel = (e.get("relationship") or "").strip()
        rel_str = f" ({rel})" if rel and rel.lower() != "unknown" else ""
        facts = "; ".join(f for f in (e.get("facts") or []) if f)
        if facts:
            lines.append(f"- {e['name']}{rel_str}: {facts}")
        else:
            lines.append(f"- {e['name']}{rel_str}")
    return "\n".join(lines)
