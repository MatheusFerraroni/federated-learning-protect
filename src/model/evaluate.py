# Imports
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from configs.experiment_config import ExperimentConfig
from src.model.dataset_utils import build_global_split_dataloader
from src.model.model_utils import get_device, move_batch_to_device
from src.utils.io import save_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _safe_perplexity(loss_value: float) -> float:
    if loss_value >= 20:
        return float('inf')
    return float(math.exp(loss_value))


def evaluate_dataloader(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, Any]:
    resolved_device = device or get_device()
    model.eval()

    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    total_non_pad_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, resolved_device)
            outputs = model(**batch)

            loss = outputs.loss
            batch_size = batch['input_ids'].size(0)
            non_pad_tokens = int((batch['labels'] != -100).sum().item())

            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            total_batches += 1
            total_non_pad_tokens += non_pad_tokens

    if total_examples == 0:
        raise ValueError('Nenhum exemplo processado durante a avaliação.')

    mean_loss = total_loss / total_examples
    perplexity = _safe_perplexity(mean_loss)

    metrics = {
        'loss': mean_loss,
        'perplexity': perplexity,
        'num_examples': total_examples,
        'num_batches': total_batches,
        'num_non_pad_tokens': total_non_pad_tokens,
    }

    LOGGER.info('Avaliação concluída: %s', metrics)
    return metrics


def evaluate_global_split(
    config: ExperimentConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    condition_name: str,
    split_name: str,
    device: torch.device | None = None,
) -> dict[str, Any]:
    try:
        dataloader = build_global_split_dataloader(
            config=config,
            tokenizer=tokenizer,
            condition_name=condition_name,
            split_name=split_name,
            shuffle=False,
        )
    except ValueError as exc:
        message = str(exc)
        if 'texts não pode ser vazio' in message:
            skipped_metrics = {
                'condition': condition_name,
                'split': split_name,
                'skipped': True,
                'reason': 'empty_split',
                'loss': None,
                'perplexity': None,
                'num_examples': 0,
                'num_batches': 0,
                'num_non_pad_tokens': 0,
            }
            LOGGER.warning(
                'Split global vazio; avaliação será ignorada: condition=%s split=%s',
                condition_name,
                split_name,
            )
            return skipped_metrics
        raise

    metrics = evaluate_dataloader(model=model, dataloader=dataloader, device=device)
    metrics['condition'] = condition_name
    metrics['split'] = split_name
    metrics['skipped'] = False
    return metrics


def evaluate_global_condition(
    config: ExperimentConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    condition_name: str,
    split_names: list[str] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    resolved_splits = split_names or ['train', 'val', 'test', 'domain_test']

    report: dict[str, Any] = {
        'condition': condition_name,
        'splits': {},
    }

    for split_name in resolved_splits:
        report['splits'][split_name] = evaluate_global_split(
            config=config,
            model=model,
            tokenizer=tokenizer,
            condition_name=condition_name,
            split_name=split_name,
            device=device,
        )

    return report


def save_evaluation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    resolved_path = Path(output_path)
    save_json(report, resolved_path)
    LOGGER.info('Relatório de avaliação salvo em %s', resolved_path)
    return resolved_path
