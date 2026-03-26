from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from configs.experiment_config import ModelConfig
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer pad_token_id could not be resolved.')

    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer eos_token_id could not be resolved.')

    return tokenizer


def load_causal_lm_model(
    model_config: ModelConfig,
    device: torch.device | None = None,
) -> PreTrainedModel:
    resolved_device = device or get_device()

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name)
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = model_config.use_cache_during_training

    model.to(resolved_device)
    model.train()
    return model


def count_trainable_parameters(model: PreTrainedModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model: PreTrainedModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def describe_model(model: PreTrainedModel) -> dict[str, Any]:
    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    frozen = total - trainable

    return {
        'model_class': model.__class__.__name__,
        'total_parameters': total,
        'trainable_parameters': trainable,
        'frozen_parameters': frozen,
    }


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device)
    return moved


def save_model_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    if ModelConfig.save_model_checkpoint:
        checkpoint_dir = Path(output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        if extra_state is not None:
            torch.save(extra_state, checkpoint_dir / 'extra_state.pt')

        LOGGER.info('Checkpoint salvo em %s', checkpoint_dir)
        return checkpoint_dir
    else:
        LOGGER.info('Checkpoint save skipped')
        return


def safe_remove_checkpoint_dir(checkpoint_dir: str | Path) -> None:
    path = Path(checkpoint_dir)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        LOGGER.info('Checkpoint removido: %s', path)
