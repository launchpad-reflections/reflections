"""Claude Haiku tool loop for the proactivity agent."""

from __future__ import annotations

import json
import logging
import time

from proactivity.promptlog import block_to_dict, log_event
from proactivity.tools import build_anthropic_tools, execute_tool

from .prompts import build_system_prompt, build_user_content

logger = logging.getLogger(__name__)


def parse_claude_json(text: str) -> dict:
    """Parse Claude's response. Strips code fences if present;
    falls back to a substring extraction on the first {...} block
    if the response has chatter wrapping the JSON."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def call_claude(
    *,
    anthropic_model: str,
    user_name: str,
    entities_block: str,
    recent_transcript: str,
    classifier_p: float,
    suggestions: str,
    last_spoken_text: str | None = None,
    seconds_since_speak: float | None = None,
) -> dict:
    """Call Claude Haiku and parse the JSON response. Returns an
    empty dict (treated as "stay silent") on any error."""
    # Kept on the signature for any future use; not surfaced to Claude.
    del last_spoken_text, seconds_since_speak

    try:
        import anthropic
    except ImportError:
        logger.warning("[proactivity] anthropic SDK not installed")
        return {}

    system_prompt = build_system_prompt(user_name)
    user_content = build_user_content(
        entities_block=entities_block,
        recent_transcript=recent_transcript,
        suggestions=suggestions,
        classifier_p=classifier_p,
    )

    try:
        client = anthropic.Anthropic()
    except Exception as e:
        logger.error("[proactivity] anthropic client init failed: %s", e)
        return {}

    messages: list[dict] = [{"role": "user", "content": user_content}]
    tools = build_anthropic_tools()

    log_event(
        "claude",
        "request",
        {
            "model": anthropic_model,
            "system": system_prompt,
            "user": user_content,
            "tools": [{"name": t.get("name"), "type": t.get("type")} for t in tools],
            "classifier_p": classifier_p,
        },
    )

    tool_calls_count = 0
    tool_names_seen: list[str] = []
    loop_started = time.monotonic()
    for _ in range(4):
        if time.monotonic() - loop_started > 12.0:
            logger.warning("[proactivity] claude loop timeout")
            return {}

        try:
            response = client.messages.create(
                model=anthropic_model,
                max_tokens=400,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )
        except Exception as e:
            log_event("claude", "error", {"error": str(e)})
            logger.error("[proactivity] claude call failed: %s", e)
            return {}

        log_event(
            "claude",
            "response",
            {
                "stop_reason": response.stop_reason,
                "content": [block_to_dict(b) for b in response.content],
                "usage": getattr(response, "usage", None) and response.usage.model_dump(),
            },
        )

        for block in response.content:
            btype = getattr(block, "type", None)
            if btype in ("tool_use", "server_tool_use"):
                tool_calls_count += 1
                bname = getattr(block, "name", None)
                if bname:
                    tool_names_seen.append(bname)

        stop_reason = response.stop_reason

        if stop_reason == "pause_turn":
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response.content},
            ]
            continue

        if stop_reason == "tool_use":
            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                try:
                    result = execute_tool(tu.name, tu.input)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": result,
                        }
                    )
                except Exception as e:
                    logger.error("[proactivity] tool %r failed: %s", tu.name, e)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": f"Tool error: {e}",
                            "is_error": True,
                        }
                    )
            messages.append({"role": "user", "content": results})
            continue

        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        text_block = text_blocks[-1] if text_blocks else ""
        parsed = parse_claude_json(text_block.strip())
        if not parsed or not parsed.get("speak"):
            preview = text_block.strip().replace("\n", " ")[:200]
            logger.info(
                "[proactivity] claude final text " "(stop=%s, parsed=%s, speak=%r): %r",
                stop_reason,
                bool(parsed),
                parsed.get("speak"),
                preview,
            )
        parsed["_tool_calls"] = tool_calls_count
        parsed["_tool_names"] = tool_names_seen
        log_event(
            "claude",
            "decision",
            {
                "speak": bool(parsed.get("speak")),
                "text": parsed.get("text") or "",
                "reason": parsed.get("reason") or "",
                "tool_calls": tool_calls_count,
                "tool_names": tool_names_seen,
            },
        )
        return parsed

    logger.warning("[proactivity] claude loop exceeded max iterations")
    return {}
