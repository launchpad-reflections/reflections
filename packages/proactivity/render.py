"""Convert raw JSON examples to XML-tagged ChatML training format for Qwen."""

SYSTEM_PROMPT = (
    "You are a binary actionability classifier for smart glasses. "
    "Determine whether the [TARGET] sentence warrants proactive intervention (1) or not (0). "
    "Output <label>, <signal>, then <reasoning>."
)

CATEGORY_TO_SIGNAL = {
    "entity_dependent": "entity",
    "location_dependent": "location",
    "tool_dependent": "tool",
    "memory_dependent": "memory",
    "multi_signal": "multi",
    "temporal": "temporal",
    "social_relational": "entity",
    "neg_vague": "transcript",
    "neg_already_handled": "transcript",
    "neg_pleasantry": "transcript",
    "neg_missing_context": "absent",
    "neg_pragmatic": "transcript",
}


def get_signal(example: dict) -> str:
    return CATEGORY_TO_SIGNAL.get(example["metadata"]["category"], "transcript")


def render_entities(entity_list: list[dict]) -> str:
    """Render entity list as text block."""
    if not entity_list:
        return "(No known entity information.)"
    lines = []
    for e in entity_list:
        facts_str = "; ".join(e["facts"])
        lines.append(f"- {e['name']} ({e['relationship']}): {facts_str}")
    return "\n".join(lines)


def render_transcript(transcript: dict) -> str:
    """Render transcript with [TARGET] markers."""
    lines = []
    for turn in transcript["turns"]:
        text = turn["text"]
        if turn.get("is_target"):
            text = f"[TARGET] {text} [/TARGET]"
        lines.append(f"{turn['speaker']}: {text}")
    return "\n".join(lines)


def render_memory(memory_summaries: list[dict]) -> str:
    """Render memory summaries as text block."""
    if not memory_summaries:
        return "(No prior conversation context available.)"
    lines = []
    for m in memory_summaries:
        lines.append(f"- [{m['timestamp_approx']}] {m['summary']}")
    return "\n".join(lines)


def render_tools(available_tools: list[str]) -> str:
    """Render tool list as comma-separated string."""
    return ", ".join(available_tools)


def render_location(location: dict) -> str:
    """Render location as text block."""
    coords = location["coordinates"]
    lines = [f"{location['description']} ({coords['latitude']}, {coords['longitude']})"]
    nearby = []
    for p in location["nearby_places"]:
        nearby.append(f"{p['name']} ({p['type']}, {p['distance_meters']}m)")
    if nearby:
        lines.append("Nearby: " + ", ".join(nearby))
    return "\n".join(lines)


def render_example(example: dict) -> str:
    """Render a full training example in Qwen ChatML format with XML tags.

    Input ordering: entities first (primacy attention bias), then transcript,
    memory, tools, location.
    """
    entities_text = render_entities(example["entity_list"])
    transcript_text = render_transcript(example["transcript"])
    memory_text = render_memory(example["memory_summaries"])
    tools_text = render_tools(example["available_tools"])
    location_text = render_location(example["location"])

    user_content = (
        f"<entities>\n{entities_text}\n</entities>\n\n"
        f"<transcript>\n{transcript_text}\n</transcript>\n\n"
        f"<memory>\n{memory_text}\n</memory>\n\n"
        f"<tools>\n{tools_text}\n</tools>\n\n"
        f"<location>\n{location_text}\n</location>"
    )

    signal = get_signal(example)
    assistant_content = (
        f"<label>{example['label']}</label>\n"
        f"<signal>{signal}</signal>\n"
        f"<reasoning>{example['reasoning']}</reasoning>"
    )

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )
