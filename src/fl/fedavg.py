from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


StateDictType = dict[str, torch.Tensor]


def clone_state_dict(state_dict: StateDictType) -> OrderedDict[str, torch.Tensor]:
    cloned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        cloned[key] = value.detach().clone()
    return cloned


def state_dict_to_cpu(state_dict: StateDictType) -> OrderedDict[str, torch.Tensor]:
    cpu_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        cpu_state[key] = value.detach().cpu().clone()
    return cpu_state


def _validate_client_updates(client_updates: list[dict[str, Any]]) -> None:
    if not client_updates:
        raise ValueError('client_updates não pode ser vazio.')

    required_keys = {'client_id', 'num_examples', 'state_dict'}
    reference_keys: set[str] | None = None

    for update in client_updates:
        missing = required_keys - set(update.keys())
        if missing:
            raise KeyError(f'Update de cliente incompleto. Faltando: {sorted(missing)}')

        num_examples = int(update['num_examples'])
        if num_examples <= 0:
            raise ValueError(
                f'num_examples inválido para cliente {update["client_id"]}: {num_examples}'
            )

        state_dict = update['state_dict']
        if not isinstance(state_dict, dict):
            raise TypeError(f'state_dict inválido para cliente {update["client_id"]}.')

        current_keys = set(state_dict.keys())
        if reference_keys is None:
            reference_keys = current_keys
        elif current_keys != reference_keys:
            raise ValueError('Os state_dicts dos clientes possuem chaves diferentes.')


def fedavg_aggregate(client_updates: list[dict[str, Any]]) -> OrderedDict[str, torch.Tensor]:
    _validate_client_updates(client_updates)

    total_examples = sum(int(update['num_examples']) for update in client_updates)
    if total_examples <= 0:
        raise ValueError('total_examples deve ser > 0.')

    reference_state = client_updates[0]['state_dict']
    aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()

    for key in reference_state.keys():
        first_tensor = reference_state[key]

        if torch.is_floating_point(first_tensor):
            accumulator = torch.zeros_like(first_tensor, dtype=first_tensor.dtype)
            for update in client_updates:
                weight = float(int(update['num_examples']) / total_examples)
                accumulator += update['state_dict'][key].to(accumulator.dtype) * weight
            aggregated[key] = accumulator
        else:
            aggregated[key] = first_tensor.detach().clone()

    LOGGER.info(
        'FedAvg concluído: num_clients=%d total_examples=%d',
        len(client_updates),
        total_examples,
    )
    return aggregated


def summarize_client_weights(client_updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_examples = sum(int(update['num_examples']) for update in client_updates)
    if total_examples <= 0:
        raise ValueError('total_examples deve ser > 0.')

    summary: list[dict[str, Any]] = []
    for update in client_updates:
        num_examples = int(update['num_examples'])
        summary.append(
            {
                'client_id': update['client_id'],
                'num_examples': num_examples,
                'aggregation_weight': num_examples / total_examples,
            }
        )
    return summary
