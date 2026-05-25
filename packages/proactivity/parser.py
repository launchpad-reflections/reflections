"""Parse memory.md into the (entity_list, memory_summaries) shape the
Qwen3 actionability classifier expects.

Two formats are supported. The PRIMARY format (written by
`packages/proactivity/memory_agent.py`) is a single ## Entities section
with one ### Name (relationship) block per person:

  ## Entities

  ### Alex (self)
  - aliases: Alex S
  - prefers oat milk
  - building reflections app

  ### Sam (friend)
  - aliases: Sam, Samuel
  - severe nut allergy

The LEGACY fallback handles the older multi-section format (## People,
## Preferences, ## Context, …) so any pre-existing memory.md still
parses while it gets rewritten on the next memory snapshot.

Output shape:
  entity_list[i] = {name, relationship, facts: [...], aliases: [...]}
  memory_summaries[i] = {timestamp_approx, summary}

Anything that doesn't fit cleanly is preserved verbatim as a memory
summary so no information is dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

# Header → bucket. Anything not in this map ends up as memory summaries.
ENTITY_SECTIONS = {"people", "preferences", "identities", "relationships"}
MEMORY_SECTIONS = {
    "context",
    "timeline",
    "tasks",
    "notes",
    "unclear references",
    "history",
    "events",
}

# - **Name**: fact (optionally "(relationship)" before the colon)
_BOLD_NAME = re.compile(
    r"^-\s*\*\*(?P<name>[^*]+)\*\*" r"(?:\s*\((?P<rel>[^)]+)\))?" r"\s*[:\-]\s*(?P<facts>.+)$"
)

# - Name | relationship | fact1; fact2
_PIPE = re.compile(r"^-\s*(?P<name>[^|]+?)\s*\|\s*(?P<rel>[^|]+?)\s*\|\s*(?P<facts>.+)$")

# - Name: fact (no bold, no pipes — must have ': ' fairly early)
_PLAIN_NAME = re.compile(
    r"^-\s*(?P<name>[A-Z][\w \-']{0,40}?)"
    r"(?:\s*\((?P<rel>[^)]+)\))?"
    r"\s*[:\-]\s+(?P<facts>.+)$"
)


def _split_facts(s: str) -> list[str]:
    """Split a fact string on '; ' or ' • '. Falls back to a single
    fact if neither separator appears."""
    parts = re.split(r"\s*(?:;|•)\s*", s.strip())
    return [p.strip() for p in parts if p.strip()]


def _parse_entity_bullet(line: str) -> dict | None:
    """Try to extract {name, relationship, facts} from a bullet line.
    Returns None if nothing matches — caller should treat the line as
    free text instead."""
    for pat in (_BOLD_NAME, _PIPE, _PLAIN_NAME):
        m = pat.match(line.strip())
        if m:
            name = m.group("name").strip()
            rel = (m.group("rel") or "").strip() or "unknown"
            facts = _split_facts(m.group("facts"))
            if name and facts:
                return {"name": name, "relationship": rel, "facts": facts}
    return None


def _split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Return [(header, body_lines), ...] in document order. The first
    block (anything before the first ## header) gets header == ''."""
    sections: list[tuple[str, list[str]]] = []
    current_header = ""
    current_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            sections.append((current_header, current_lines))
            current_header = line[3:].strip().lower()
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_header, current_lines))
    return sections


def _bullets(lines: list[str]) -> list[str]:
    """Return only the lines that start with '- ' (Markdown bullet)."""
    return [ln for ln in lines if ln.lstrip().startswith("- ")]


def _non_bullet_paragraph(lines: list[str]) -> str:
    """Join all lines into a single paragraph, dropping blank lines.
    Used for sections that aren't bulleted (rare but possible)."""
    return " ".join(ln.strip() for ln in lines if ln.strip())


# Header for the new entity-block format: "### Name (relationship)".
_ENTITY_HEADER = re.compile(r"^###\s+(?P<name>[^()\n]+?)" r"(?:\s*\((?P<rel>[^)]+)\))?\s*$")

# A bullet whose body begins with "aliases:" lists alternative names
# for retrieval. Stripped from facts and stored on entity["aliases"].
_ALIASES_BULLET = re.compile(r"^-\s*aliases?\s*[:\-]\s*(?P<list>.+)$", re.IGNORECASE)


def _parse_entities_section(lines: list[str]) -> list[dict]:
    """Parse the body of a `## Entities` section into entity dicts.

    Each `### Name (relationship)` heading starts a new entity. Bullet
    lines beneath it become facts; an `- aliases: A, B` bullet is
    extracted into entity["aliases"] instead."""
    entities: list[dict] = []
    current: dict | None = None
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        m_head = _ENTITY_HEADER.match(line)
        if m_head:
            current = {
                "name": m_head.group("name").strip(),
                "relationship": (m_head.group("rel") or "unknown").strip(),
                "facts": [],
                "aliases": [],
            }
            entities.append(current)
            continue
        stripped = line.lstrip()
        if not stripped.startswith("- ") or current is None:
            continue
        m_alias = _ALIASES_BULLET.match(stripped)
        if m_alias:
            aliases = [a.strip() for a in re.split(r"[,;]\s*", m_alias.group("list")) if a.strip()]
            current["aliases"].extend(aliases)
            continue
        # Plain fact bullet.
        fact = stripped.lstrip("- ").strip()
        if fact:
            current["facts"].append(fact)
    return entities


def parse_memory_md(text: str) -> tuple[list[dict], list[dict]]:
    """Parse a memory.md document into (entity_list, memory_summaries).

    Tries the new `## Entities` + `### Name (rel)` format first; falls
    back to the legacy multi-section format if no `## Entities` block
    is present. If no recognized sections appear at all, the whole
    document is preserved verbatim as a single memory summary.
    """
    text = (text or "").strip()
    if not text:
        return [], []

    sections = _split_sections(text)

    # ---- New format: a single `## Entities` section with `###` blocks.
    for header, body_lines in sections:
        if header == "entities":
            entities = _parse_entities_section(body_lines)
            summaries: list[dict] = []
            # Any *other* recognized section is pulled in as a summary
            # so notes/timeline content still flows through. Most of the
            # time `## Entities` is the only section.
            for h2, body2 in sections:
                if h2 == "entities" or not h2:
                    continue
                chunk = "\n".join(body2).strip()
                if chunk:
                    summaries.append({"timestamp_approx": h2, "summary": chunk})
            return entities, summaries

    # ---- Legacy multi-section format fallback.
    entities: list[dict] = []
    summaries: list[dict] = []
    fallback_lines: list[str] = []

    # Merge entity entries with the same name (e.g. ## People may say
    # "Bob: friend" and ## Preferences may say "Bob: likes oat milk" —
    # both should attach to one Bob entity).
    by_name: dict[str, dict] = {}

    def _add_entity(e: dict) -> None:
        # Ensure every entity has an `aliases` slot so the speaker-
        # filter never KeyErrors on legacy-parsed entries.
        e.setdefault("aliases", [])
        existing = by_name.get(e["name"])
        if existing:
            existing["facts"].extend(e["facts"])
            # Prefer the more specific relationship.
            if existing["relationship"] in ("unknown", "") and e["relationship"]:
                existing["relationship"] = e["relationship"]
        else:
            by_name[e["name"]] = e
            entities.append(e)

    recognized = False
    for header, lines in sections:
        bullets = _bullets(lines)
        body = "\n".join(lines).strip()
        if not body:
            continue

        if header in ENTITY_SECTIONS:
            recognized = True
            for b in bullets:
                ent = _parse_entity_bullet(b)
                if ent:
                    _add_entity(ent)
                else:
                    # Bullet didn't fit any entity pattern → preserve as
                    # a free-form fact in the catch-all summary so it
                    # isn't lost.
                    fallback_lines.append(f"[{header}] {b.strip().lstrip('- ').strip()}")
        elif header in MEMORY_SECTIONS:
            recognized = True
            chunk = "\n".join(lines).strip()
            if chunk:
                summaries.append(
                    {
                        "timestamp_approx": header,
                        "summary": chunk,
                    }
                )
        else:
            # Unknown header (or pre-header preamble): keep the text in
            # case the classifier benefits from it.
            chunk = "\n".join(lines).strip()
            if chunk:
                tag = header or "preamble"
                fallback_lines.append(f"[{tag}]\n{chunk}")

    if fallback_lines:
        summaries.append(
            {
                "timestamp_approx": "misc",
                "summary": "\n".join(fallback_lines),
            }
        )

    if not recognized:
        # Truly nothing recognized — dump the whole thing as one
        # summary so the classifier still has signal.
        return [], [{"timestamp_approx": "session", "summary": text}]

    return entities, summaries


def parse_memory_file(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Convenience: read a file path and parse it. Returns ([], []) if
    the file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return [], []
    return parse_memory_md(p.read_text(encoding="utf-8"))
