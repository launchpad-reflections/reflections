"""Memory agent: on demand, send the unsent portion of the transcript
to Claude and have it update a persistent `memory.md` file.

Triggered by the viewer when the user presses 's' in the video window.
Each invocation:
  1. Reads the full transcript so far (finalized segments only).
  2. Diffs against the index last sent → produces the unsent chunk.
  3. Loads current memory.md (empty string if missing).
  4. Asks Claude to return an updated memory.md that preserves prior
     content and merges in the new chunk.
  5. Atomically writes the result to memory.md.

The Claude call runs on a background thread so the render loop is
never blocked. A threading.Lock serializes rapid-fire 's' presses; the
second call will simply find nothing new and no-op.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from proactivity.promptlog import log_event

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class MemoryAgent:
    def __init__(
        self,
        *,
        get_transcript: Callable[[], list[tuple[str | None, str]]],
        memory_path: str | Path = "memory.md",
        user_name: str = "User",
        model: str = _DEFAULT_MODEL,
    ):
        self.memory_path = Path(memory_path)
        self.user_name = user_name
        self.model = model
        self._get_transcript = get_transcript

        self._lock = threading.Lock()
        self._last_sent_index: int = 0

    def snapshot(self) -> None:
        """Run one memory update cycle. Safe to call from any thread;
        callers typically spawn a short-lived daemon thread so the UI
        isn't blocked by the Claude round-trip."""
        with self._lock:
            transcript = self._get_transcript()
            unsent = transcript[self._last_sent_index :]
            if not unsent:
                log_event("memory", "skip", {"reason": "nothing_new"})
                logger.info("[memory] nothing new since last snapshot")
                return

            current_memory = self._read_memory()
            chunk = self._format_chunk(unsent)

            log_event(
                "memory",
                "request",
                {
                    "unsent_lines": len(unsent),
                    "chunk": chunk,
                    "current_memory": current_memory,
                    "current_size": len(current_memory),
                    "model": self.model,
                },
            )
            logger.info(
                "[memory] sending %d finalized lines to Claude (%s)...",
                len(unsent),
                self.model,
            )

            t0 = time.monotonic()
            new_memory = self._call_claude(current_memory, chunk)
            ms = (time.monotonic() - t0) * 1000.0
            if new_memory is None:
                log_event(
                    "memory",
                    "error",
                    {
                        "error": "claude_call_failed",
                        "ms": ms,
                    },
                )
                logger.error("[memory] Claude call failed; memory.md unchanged")
                return

            self._write_memory(new_memory)
            self._last_sent_index = len(transcript)
            log_event(
                "memory",
                "response",
                {
                    "new_memory": new_memory,
                    "new_size": len(new_memory),
                    "delta": len(new_memory) - len(current_memory),
                    "unsent_lines": len(unsent),
                    "ms": ms,
                },
            )
            logger.info(
                "[memory] updated %s (+%d lines; memory now %d chars)",
                self.memory_path,
                len(unsent),
                len(new_memory),
            )

    # ---- internals ----

    def _read_memory(self) -> str:
        try:
            return self.memory_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _write_memory(self, text: str) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.memory_path.with_suffix(self.memory_path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.memory_path)

    def _format_chunk(self, unsent: list[tuple[str | None, str]]) -> str:
        return "\n".join(
            f"[{spk if (spk and spk != 'User') else self.user_name}]: {text}"
            for spk, text in unsent
        )

    def _call_claude(
        self,
        current_memory: str,
        new_chunk: str,
    ) -> str | None:
        try:
            import anthropic
        except ImportError:
            logger.warning("[memory] anthropic not installed")
            return None

        prompt = (
            "You maintain memory.md, a persistent gallery of PEOPLE in the "
            "wearer's life, distilled from smart-glasses conversation audio.\n\n"
            "## REQUIRED FORMAT\n\n"
            "memory.md must contain exactly one section, ## Entities, with one "
            "### subsection per person. Each entity looks like this:\n\n"
            "  ## Entities\n\n"
            "  ### Alex (self)\n"
            "  - prefers oat milk\n"
            "  - building reflections app for smart glasses\n"
            "  - lives in Example City\n\n"
            "  ### Sam (friend)\n"
            "  - aliases: Sam, Samuel\n"
            "  - severe nut allergy\n"
            "  - works at a Series A startup\n\n"
            "Heading: `### Name (relationship)`. Relationship is one of "
            "`self`, `friend`, `family`, `colleague`, `acquaintance`, or a "
            "specific role. The wearer is always relationship `self`.\n\n"
            "Optional `- aliases: A, B, C` bullet captures spelling variants, "
            "nicknames, or alternate transcriptions of the SAME person — used "
            "downstream to retrieve their entity when the live transcript "
            "spells the name slightly differently.\n\n"
            "## WHAT TO INCLUDE\n\n"
            "Only DURABLE facts that would matter on a future conversation:\n"
            "  - dietary restrictions, allergies, medical issues\n"
            "  - consumer preferences (brands, cuisines, drinks)\n"
            "  - relationships (who knows who, how)\n"
            "  - work, projects, location\n"
            "  - recurring topics they care about\n"
            "  - clear personality traits or values\n\n"
            "## WHAT TO EXCLUDE — BE STRICT\n\n"
            "  - small talk, greetings, conversational filler\n"
            '  - time-bound events ("they were tired this morning")\n'
            '  - one-off opinions ("liked the latte today")\n'
            "  - generic observations\n"
            "  - tasks/reminders/calendar items (those go to Google Calendar)\n"
            "  - ANY suggestion the assistant made (those are in suggestions.md)\n"
            "  - anything that wouldn't matter a week from now\n\n"
            "Each fact must be one short clause. No paragraphs.\n\n"
            "## DEDUPLICATION\n\n"
            "Resolve same-person duplicates aggressively:\n"
            "  - If two existing entries are clearly the same person — "
            "different spellings, nicknames, or one is a 'Person N' label "
            "that has since been resolved to a real name — MERGE them. Pick "
            "the canonical name as the heading; record the others under "
            "`- aliases: X, Y`.\n"
            "  - DO NOT create entities for unresolved 'Person 2', "
            "'Person 5', 'Track 14' or similar synthetic labels. Skip them "
            "until a real name appears in conversation.\n"
            "  - Preserve existing entities and facts unless the new chunk "
            "explicitly contradicts them.\n\n"
            "## CURRENT memory.md\n\n"
            f"```\n{current_memory.strip() or '(empty — fresh memory file)'}\n```\n\n"
            "## NEW CONVERSATION CHUNK (since last update)\n\n"
            f"```\n{new_chunk}\n```\n\n"
            "Return ONLY the new full contents of memory.md. Markdown only. "
            "No preamble. No code fences. No explanations. If the chunk "
            "contains nothing worth recording, return the existing "
            "memory.md unchanged."
        )

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Defensive: strip code fences if Claude slipped them in despite
            # the instruction.
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            return text
        except Exception as e:
            logger.error("[memory] Claude error: %s", e)
            return None
