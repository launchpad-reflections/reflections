"""Long-lived wrapper around the Qwen 3 1.7B + LoRA actionability classifier.



Loads the Qwen base model + LoRA adapter from Hugging Face Hub once and

reuses them across many classifications. Adds a `label_only` shortcut that

skips the (~1.2 s) reasoning-generation step when only the binary verdict is

needed — this is the fast path the proactivity loop uses as its gate.



Latency, on a Mac M-series with the model already cached:

  - load_model() at construction: ~3 s (~90 s on the very first run

    because it downloads ~3.4 GB from Hugging Face).

  - classify(label_only=True): ~200 ms regardless of label.

  - classify(label_only=False): ~200 ms when label==0, ~1400 ms when

    label==1 (because reasoning is generated).

"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from reflections.config import LORA_MODEL_ID, default_location
from transformers import AutoModelForCausalLM, AutoTokenizer

from proactivity.promptlog import log_event
from proactivity.render import render_example

logger = logging.getLogger(__name__)


# Dev fallback when HF Hub is unreachable (e.g. offline development).

LOCAL_ADAPTER_PATH = Path(__file__).resolve().parent / "qwen3-actionable-v2-adapter"


BASE_MODEL = "Qwen/Qwen3-1.7B"


# Threshold at which the FULL classify path generates a reasoning string.

# Only used by the smoke scripts (scripts/smoke_full_transcript.py, scripts/smoke_server.py)

# and the full `classify()` function below — the live glasses pipeline

# uses _label_only_classify (no reasoning) and gates separately at

# agent.GLASSES_GATE_THRESHOLD. Do NOT confuse this with the gate

# threshold; the live path ignores `label` entirely.

REASONING_TRIGGER = 0.45


def _default_location() -> dict[str, Any]:

    return default_location()


_DEFAULT_LOCATION = _default_location()


def _load_lora_adapter(base) -> PeftModel:
    """Load LoRA from Hugging Face Hub, with optional local dev fallback."""

    logger.info("LoRA adapter: %s", LORA_MODEL_ID)

    try:

        return PeftModel.from_pretrained(base, LORA_MODEL_ID)

    except Exception as exc:

        logger.warning(
            "HF LoRA load failed for %s (%s); trying local path %s",
            LORA_MODEL_ID,
            exc,
            LOCAL_ADAPTER_PATH,
        )

        if not LOCAL_ADAPTER_PATH.exists():

            raise FileNotFoundError(
                f"LoRA load failed for {LORA_MODEL_ID!r} and no local fallback at "
                f"{LOCAL_ADAPTER_PATH}. Set REFLECTIONS_LORA_MODEL_ID or run "
                f"./scripts/download_lora.sh to prefetch the adapter."
            ) from exc

        return PeftModel.from_pretrained(base, str(LOCAL_ADAPTER_PATH))


def load_model():
    """Load Qwen base + LoRA adapter and return everything classify() needs."""

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info("device: %s", device)

    logger.info("tokenizer: %s", BASE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    logger.info("base model: %s (downloads ~3.5 GB on first run)", BASE_MODEL)

    t0 = time.time()

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    )

    logger.info("  base loaded in %.1fs", time.time() - t0)

    model = _load_lora_adapter(base)

    model = model.to(device)

    model.train(False)

    logger.info("ready")

    t0_id = tokenizer.encode("0", add_special_tokens=False)[0]

    t1_id = tokenizer.encode("1", add_special_tokens=False)[0]

    return model, tokenizer, device, t0_id, t1_id


def classify(model, tokenizer, device, t0_id, t1_id, example):
    """Full classify: P(label=1), label, and reasoning string when label==1."""

    rendered = render_example(example)

    marker = "<|im_start|>assistant\n<label>"

    idx = rendered.find(marker)

    if idx == -1:

        marker = "<|im_start|>assistant\n"

        idx = rendered.index(marker)

    prompt = rendered[: idx + len(marker)]

    target_text = example["transcript"]["turns"][-1].get("text", "")

    log_event(
        "classifier",
        "prompt",
        {
            "mode": "full",
            "target": target_text,
            "transcript": example["transcript"]["turns"],
            "tools": example.get("available_tools", []),
            "prompt": prompt,
        },
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():

        logits = model(**inputs).logits[0, -1, :]

    pair = torch.tensor([logits[t0_id], logits[t1_id]])

    p = torch.softmax(pair, dim=0)[1].item()

    label = 1 if p >= REASONING_TRIGGER else 0

    log_event(
        "classifier",
        "result",
        {
            "p": p,
            "label": label,
            "reasoning_trigger": REASONING_TRIGGER,
            "mode": "full",
        },
    )

    reasoning = None

    if label == 1:

        digit = "1" if p >= 0.5 else "0"

        gen_prompt = prompt + digit + "</label>\n"

        gen_inputs = tokenizer(gen_prompt, return_tensors="pt").to(device)

        with torch.no_grad():

            out = model.generate(
                **gen_inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        reasoning = tokenizer.decode(
            out[0][gen_inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    return p, label, reasoning


def _build_example(
    *,
    turns: list[dict],
    memory_summaries: list[dict] | None = None,
    entity_list: list[dict] | None = None,
    tools: list[str] | None = None,
    location: dict | None = None,
) -> dict:
    """Assemble the dict shape render_example() expects. Marks the last

    turn as the [TARGET] sentence."""

    if not turns:

        raise ValueError("turns must be non-empty")

    turns = [dict(t) for t in turns]  # shallow-copy so we don't mutate caller

    turns[-1]["is_target"] = True

    return {
        "id": "live",
        "transcript": {
            "turns": turns,
            "target_speaker": turns[-1]["speaker"],
            "target_index": len(turns) - 1,
        },
        "memory_summaries": memory_summaries or [],
        "available_tools": tools or ["send_message"],
        "location": location or _DEFAULT_LOCATION,
        "entity_list": entity_list or [],
        # Required by render_example() but unused at inference time.
        "label": 0,
        "reasoning": "(live)",
        "metadata": {
            "category": "memory_dependent",
            "subcategory": "live",
            "difficulty": "medium",
            "signals_used": [],
            "action_type": None,
        },
    }


def _label_only_classify(model, tokenizer, device, t0_id, t1_id, example):
    """Fast path: just compute P(label=1) from one forward pass over the

    rendered prompt; never run the slow generation step."""

    rendered = render_example(example)

    marker = "<|im_start|>assistant\n<label>"

    idx = rendered.find(marker)

    if idx == -1:

        marker = "<|im_start|>assistant\n"

        idx = rendered.index(marker)

    prompt = rendered[: idx + len(marker)]

    target_text = example["transcript"]["turns"][-1].get("text", "")

    log_event(
        "classifier",
        "prompt",
        {
            "mode": "label_only",
            "target": target_text,
            "transcript": example["transcript"]["turns"],
            "tools": example.get("available_tools", []),
            "prompt": prompt,
        },
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():

        logits = model(**inputs).logits[0, -1, :]

    pair = torch.tensor([logits[t0_id], logits[t1_id]])

    p = torch.softmax(pair, dim=0)[1].item()

    # `label` is computed against REASONING_TRIGGER but is informational

    # only on the live path — the agent gates on raw `p` against

    # GLASSES_GATE_THRESHOLD (see proactivity/agent/worker.py). We don't include

    # a threshold field in this event because the *gate* threshold is

    # owned by the agent and emitted separately as `classifier/gate`.

    label = 1 if p >= REASONING_TRIGGER else 0

    log_event(
        "classifier",
        "result",
        {
            "p": p,
            "label": label,
            "mode": "label_only",
        },
    )

    return p, label


class ActionabilityClassifier:
    """Holds the Qwen base model + LoRA adapter in memory and exposes a

    simple classify() over the structured inputs the proactivity loop

    builds each turn."""

    def __init__(self, *, verbose: bool = True):

        # No `threshold` field on the wrapper — gating is the agent's

        # responsibility (proactivity/agent/worker.py:GLASSES_GATE_THRESHOLD).

        # Keeping a threshold here would just confuse: the wrapper's

        # value was never read for anything.

        if verbose:

            logger.info("loading Qwen 3 1.7B + LoRA...")

        self._model, self._tokenizer, self._device, self._t0_id, self._t1_id = load_model()

        if verbose:

            logger.info("ready on %s", self._device)

    def classify(
        self,
        *,
        turns: list[dict],
        memory_summaries: list[dict] | None = None,
        entity_list: list[dict] | None = None,
        tools: list[str] | None = None,
        location: dict | None = None,
        label_only: bool = True,
    ) -> tuple[float, int, str | None]:
        """Run one classification pass.



        Args:

          turns: list of {speaker, text} dicts in chronological order;

                 the last entry is the target sentence.

          memory_summaries: list of {timestamp_approx, summary} dicts

                            (parsed from memory.md).

          entity_list: list of {name, relationship, facts} dicts.

          tools: list of tool name strings the wearer's app exposes.

          location: optional location dict (falls back to a static default).

          label_only: when True (default), skip the reasoning-generation

                      step entirely. Returns (p, label, None).

                      When False, also generate the model's reasoning

                      string when label==1 — adds ~1.2 s.



        Returns:

          (p, label, reasoning_or_None)

        """

        example = _build_example(
            turns=turns,
            memory_summaries=memory_summaries,
            entity_list=entity_list,
            tools=tools,
            location=location,
        )

        if label_only:

            p, label = _label_only_classify(
                self._model,
                self._tokenizer,
                self._device,
                self._t0_id,
                self._t1_id,
                example,
            )

            return p, label, None

        return classify(
            self._model,
            self._tokenizer,
            self._device,
            self._t0_id,
            self._t1_id,
            example,
        )

    @property
    def device(self) -> Any:

        return self._device
