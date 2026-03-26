"""Executa treino local de um cliente honesto ou malicioso em uma rodada FL.

Responsabilidades principais:
- Resolver caminhos dos datasets locais por condição.
- Construir DataLoaders de treino e validação por cliente.
- Clonar o modelo global e treinar localmente por uma rodada.
- Retornar state_dict, métricas locais e metadados do cliente.

Como este arquivo se encaixa no projeto:
- É a unidade executora do lado do cliente no loop federado manual.
- É usado pelo servidor para treinar clientes honestos e maliciosos.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from configs.experiment_config import ExperimentConfig
from configs.paths import (
    CLIENT_ATTACK_RAW_DATA_DIR,
    CLIENT_ATTACK_SEMANTIC_DATA_DIR,
    CLIENT_RAW_DATA_DIR,
    CLIENT_SEMANTIC_DATA_DIR,
)
from src.model.dataset_utils import (
    build_causal_lm_dataset,
    build_dataloader,
    load_records_from_jsonl,
)
from src.model.evaluate import evaluate_dataloader
from src.model.model_utils import get_device, move_batch_to_device
from src.model.train_local import build_optimizer, build_scheduler
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

VALID_SPLIT_NAMES = {'train', 'val', 'test', 'domain_test'}
VALID_CONDITIONS = {
    'raw',
    'semantic_substitution',
    'attack_raw',
    'attack_semantic_substitution',
}
VALID_CLIENT_PREFIXES = ('client_', 'attacker_')


@dataclass
class ClientTrainingResult:
    """Resultado de treino local de um cliente em uma rodada.

    Args:
        client_id: Identificador do cliente.
        condition_name: Condição federada usada no treino.
        client_role: Papel do cliente, ex. honest ou malicious.
        num_examples: Quantidade de exemplos de treino processados.
        state_dict: Estado atualizado do modelo local.
        train_metrics: Histórico por época local.
        val_metrics: Métricas do split local de validação.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    client_id: str
    condition_name: str
    client_role: str
    num_examples: int
    state_dict: dict[str, torch.Tensor]
    train_metrics: list[dict[str, Any]]
    val_metrics: dict[str, Any] | None


def infer_client_role(client_id: str) -> str:
    """Infere o papel do cliente a partir do identificador.

    Args:
        client_id: Identificador do cliente.

    Returns:
        "malicious" se prefixo attacker_, senão "honest".

    Raises:
        Não se aplica.
    """
    if client_id.startswith('attacker_'):
        return 'malicious'
    return 'honest'


def resolve_client_condition_dir(condition_name: str) -> Path:
    """Resolve o diretório local de uma condição.

    Args:
        condition_name: Nome da condição.

    Returns:
        Caminho raiz da condição.

    Raises:
        ValueError: Se a condição for inválida.
    """
    if condition_name == 'raw':
        return CLIENT_RAW_DATA_DIR
    if condition_name == 'semantic_substitution':
        return CLIENT_SEMANTIC_DATA_DIR
    if condition_name == 'attack_raw':
        return CLIENT_ATTACK_RAW_DATA_DIR
    if condition_name == 'attack_semantic_substitution':
        return CLIENT_ATTACK_SEMANTIC_DATA_DIR
    raise ValueError(f'condition_name inválido: {condition_name}')


def resolve_client_split_path(
    condition_name: str,
    client_id: str,
    split_name: str,
) -> Path:
    """Resolve o caminho do split local do cliente.

    Args:
        condition_name: Condição experimental.
        client_id: Identificador do cliente.
        split_name: Nome do split.

    Returns:
        Caminho do JSONL correspondente.

    Raises:
        ValueError: Se condição ou split forem inválidos.
    """
    if condition_name not in VALID_CONDITIONS:
        raise ValueError(f'condition_name inválido: {condition_name}')
    if split_name not in VALID_SPLIT_NAMES:
        raise ValueError(f'split_name inválido: {split_name}')

    return resolve_client_condition_dir(condition_name) / client_id / f'{split_name}.jsonl'


def load_client_split_records(
    condition_name: str,
    client_id: str,
    split_name: str,
) -> list[dict]:
    """Carrega um split local do cliente.

    Args:
        condition_name: Condição experimental.
        client_id: Identificador do cliente.
        split_name: Nome do split.

    Returns:
        Lista de registros do split.

    Raises:
        FileNotFoundError: Se o JSONL não existir.
    """
    path = resolve_client_split_path(
        condition_name=condition_name,
        client_id=client_id,
        split_name=split_name,
    )
    records = load_records_from_jsonl(path)
    LOGGER.info(
        'Carregado split do cliente: condition=%s client_id=%s split=%s num_records=%d path=%s',
        condition_name,
        client_id,
        split_name,
        len(records),
        path,
    )
    return records


def build_client_dataloader(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
    condition_name: str,
    client_id: str,
    split_name: str,
    shuffle: bool = False,
) -> DataLoader:
    """Monta DataLoader para um split local do cliente.

    Args:
        config: Configuração do experimento.
        tokenizer: Tokenizer do modelo.
        condition_name: Condição experimental.
        client_id: Identificador do cliente.
        split_name: Nome do split.
        shuffle: Se embaralha o DataLoader.

    Returns:
        DataLoader pronto para treino/avaliação.

    Raises:
        ValueError: Se o dataset tokenizado ficar vazio.
    """
    records = load_client_split_records(
        condition_name=condition_name,
        client_id=client_id,
        split_name=split_name,
    )
    dataset = build_causal_lm_dataset(
        records=records,
        tokenizer=tokenizer,
        model_config=config.model,
    )
    batch_size = config.model.batch_size if split_name == 'train' else config.model.eval_batch_size
    return build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.model.num_workers,
    )


def list_available_client_ids(condition_name: str) -> list[str]:
    """Lista IDs de clientes disponíveis em uma condição.

    Args:
        condition_name: Nome da condição.

    Returns:
        Lista ordenada de clientes honestos e maliciosos.

    Raises:
        FileNotFoundError: Se o diretório não existir.
        ValueError: Se não houver clientes válidos.
    """
    condition_dir = resolve_client_condition_dir(condition_name)
    if not condition_dir.exists():
        raise FileNotFoundError(f'Diretório de clientes não encontrado: {condition_dir}')

    client_ids = sorted(
        path.name
        for path in condition_dir.iterdir()
        if path.is_dir() and path.name.startswith(VALID_CLIENT_PREFIXES)
    )
    if not client_ids:
        raise ValueError(f'Nenhum cliente encontrado em: {condition_dir}')
    return client_ids


def _clone_model_for_client(model: PreTrainedModel, device: torch.device) -> PreTrainedModel:
    """Clona o modelo global para treino local.

    Args:
        model: Modelo global.
        device: Device alvo.

    Returns:
        Cópia do modelo no device informado.

    Raises:
        Não se aplica.
    """
    client_model = deepcopy(model)
    client_model.to(device)
    return client_model


def train_client_one_round(
    config: ExperimentConfig,
    global_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    condition_name: str,
    client_id: str,
    device: torch.device | None = None,
) -> ClientTrainingResult:
    """Treina um cliente em uma rodada federada.

    Args:
        config: Configuração do experimento.
        global_model: Modelo global atual.
        tokenizer: Tokenizer do modelo.
        condition_name: Condição usada pelo cliente.
        client_id: Identificador do cliente.
        device: Device opcional.

    Returns:
        Resultado local contendo pesos e métricas.

    Raises:
        ValueError: Se gradient_accumulation_steps for inválido.
        RuntimeError: Se a loss não estiver disponível.
    """
    resolved_device = device or get_device()
    client_role = infer_client_role(client_id)

    train_loader = build_client_dataloader(
        config=config,
        tokenizer=tokenizer,
        condition_name=condition_name,
        client_id=client_id,
        split_name='train',
        shuffle=True,
    )
    val_loader = build_client_dataloader(
        config=config,
        tokenizer=tokenizer,
        condition_name=condition_name,
        client_id=client_id,
        split_name='val',
        shuffle=False,
    )

    client_model = _clone_model_for_client(global_model, resolved_device)
    client_model.train()

    local_config = deepcopy(config)
    local_config.model.local_epochs = int(config.federated.local_epochs)

    optimizer = build_optimizer(
        model=client_model,
        learning_rate=local_config.model.learning_rate,
        weight_decay=local_config.model.weight_decay,
    )
    total_steps = len(train_loader) * local_config.model.local_epochs
    scheduler = build_scheduler(
        optimizer=optimizer,
        num_training_steps=max(1, total_steps),
        warmup_ratio=local_config.model.warmup_ratio,
    )

    train_history: list[dict[str, Any]] = []
    gradient_accumulation_steps = int(local_config.model.gradient_accumulation_steps)
    if gradient_accumulation_steps <= 0:
        raise ValueError('gradient_accumulation_steps deve ser > 0.')

    num_examples = 0

    for epoch_index in range(local_config.model.local_epochs):
        total_loss = 0.0
        total_batches = 0
        total_examples = 0

        optimizer.zero_grad(set_to_none=True)

        for step_index, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, resolved_device)
            outputs = client_model(**batch)
            raw_loss = outputs.loss
            if raw_loss is None:
                raise RuntimeError('Loss ausente no treino local do cliente.')

            loss = raw_loss / gradient_accumulation_steps
            loss.backward()

            batch_size = int(batch['input_ids'].size(0))
            total_loss += float(raw_loss.detach().cpu().item()) * batch_size
            total_examples += batch_size
            total_batches += 1

            should_step = step_index % gradient_accumulation_steps == 0
            is_last_step = step_index == len(train_loader)

            if should_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if total_examples == 0:
            raise ValueError(f'Cliente {client_id} não possui exemplos de treino.')

        num_examples = total_examples
        epoch_metrics = {
            'epoch_index': epoch_index,
            'train_loss': total_loss / total_examples,
            'num_examples': total_examples,
            'num_batches': total_batches,
            'learning_rate_end': float(optimizer.param_groups[0]['lr']),
        }
        train_history.append(epoch_metrics)

    val_metrics = evaluate_dataloader(
        model=client_model,
        dataloader=val_loader,
        device=resolved_device,
    )

    client_state = {
        key: value.detach().cpu().clone() for key, value in client_model.state_dict().items()
    }

    del client_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = ClientTrainingResult(
        client_id=client_id,
        condition_name=condition_name,
        client_role=client_role,
        num_examples=num_examples,
        state_dict=client_state,
        train_metrics=train_history,
        val_metrics=val_metrics,
    )

    LOGGER.info(
        'Treino local concluído: client_id=%s role=%s condition=%s num_examples=%d val_loss=%s',
        client_id,
        client_role,
        condition_name,
        num_examples,
        None if val_metrics is None else val_metrics.get('loss'),
    )
    return result
