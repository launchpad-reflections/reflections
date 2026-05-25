"""Classifier tests with mocked transformers/peft (no GPU)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from proactivity.classifier import (
    REASONING_TRIGGER,
    ActionabilityClassifier,
    _build_example,
    _label_only_classify,
    _load_lora_adapter,
    classify,
    load_model,
)
from reflections.config import LORA_MODEL_ID


def _fake_logits(p_one: float) -> torch.Tensor:
    """Build logits where softmax over [t0, t1] yields p_one for label 1."""
    t0_logit = 0.0
    t1_logit = torch.log(torch.tensor(p_one / (1.0 - p_one))).item()
    vocab = 50000
    logits = torch.zeros(vocab)
    logits[10] = t0_logit
    logits[11] = t1_logit
    return logits


class _TokenizerOutput:
    def to(self, _device: str) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1, 2, 3]])}


@pytest.fixture
def mock_stack():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda s, add_special_tokens=False: [10 if s == "0" else 11]
    tokenizer.side_effect = lambda prompt, return_tensors="pt": _TokenizerOutput()
    tokenizer.eos_token_id = 0
    tokenizer.decode.return_value = "User wants food nearby."

    forward_out = MagicMock()
    forward_out.logits = _fake_logits(0.7).unsqueeze(0).unsqueeze(0)
    model.return_value = forward_out

    model.generate.return_value = torch.tensor([[1, 2, 3, 99, 100]])

    device = "cpu"
    t0_id, t1_id = 10, 11
    return model, tokenizer, device, t0_id, t1_id


def test_build_example_marks_last_turn_target() -> None:
    example = _build_example(turns=[{"speaker": "A", "text": "hi"}])
    assert example["transcript"]["turns"][-1]["is_target"] is True


def test_build_example_requires_turns() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _build_example(turns=[])


def test_label_only_classify_returns_probability(mock_stack, tmp_repo) -> None:
    model, tokenizer, device, t0_id, t1_id = mock_stack
    example = _build_example(turns=[{"speaker": "Alex", "text": "find ramen nearby"}])
    p, label = _label_only_classify(model, tokenizer, device, t0_id, t1_id, example)
    assert 0.0 <= p <= 1.0
    assert label == (1 if p >= REASONING_TRIGGER else 0)


def test_classify_generates_reasoning_when_actionable(mock_stack, tmp_repo) -> None:
    model, tokenizer, device, t0_id, t1_id = mock_stack
    example = _build_example(turns=[{"speaker": "Alex", "text": "find ramen nearby"}])
    p, label, reasoning = classify(model, tokenizer, device, t0_id, t1_id, example)
    assert p >= REASONING_TRIGGER
    assert label == 1
    assert reasoning == "User wants food nearby."


@patch("proactivity.classifier.load_model")
def test_actionability_classifier_label_only(mock_load, mock_stack) -> None:
    mock_load.return_value = mock_stack
    clf = ActionabilityClassifier(verbose=False)
    p, label, reasoning = clf.classify(
        turns=[{"speaker": "Sam", "text": "what's nearby"}],
        label_only=True,
    )
    assert isinstance(p, float)
    assert label in (0, 1)
    assert reasoning is None


@patch("proactivity.classifier.PeftModel")
def test_load_lora_adapter_uses_hf_model_id(mock_peft) -> None:
    base = MagicMock()
    mock_peft.from_pretrained.return_value = MagicMock()

    _load_lora_adapter(base)

    mock_peft.from_pretrained.assert_called_once_with(base, LORA_MODEL_ID)


@patch("proactivity.classifier.PeftModel")
def test_load_lora_adapter_falls_back_to_local_path(mock_peft, tmp_path) -> None:
    base = MagicMock()
    local_dir = tmp_path / "qwen3-actionable-v2-adapter"
    local_dir.mkdir()
    (local_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    mock_peft.from_pretrained.side_effect = [OSError("hub unreachable"), MagicMock()]

    with patch("proactivity.classifier.LOCAL_ADAPTER_PATH", local_dir):
        _load_lora_adapter(base)

    assert mock_peft.from_pretrained.call_args_list == [
        ((base, LORA_MODEL_ID),),
        ((base, str(local_dir)),),
    ]


@patch("proactivity.classifier._load_lora_adapter")
@patch("proactivity.classifier.AutoModelForCausalLM")
@patch("proactivity.classifier.AutoTokenizer")
def test_load_model_calls_hf_lora_loader(
    mock_tokenizer_cls,
    mock_model_cls,
    mock_load_lora,
) -> None:
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda s, add_special_tokens=False: [10 if s == "0" else 11]
    mock_tokenizer_cls.from_pretrained.return_value = tokenizer
    mock_model_cls.from_pretrained.return_value = MagicMock()
    mock_load_lora.return_value = MagicMock()

    model, tok, device, t0_id, t1_id = load_model()

    mock_load_lora.assert_called_once_with(mock_model_cls.from_pretrained.return_value)
    assert tok is tokenizer
    assert device == "cpu"
    assert (t0_id, t1_id) == (10, 11)
    mock_load_lora.return_value.to.assert_called_once_with("cpu")
