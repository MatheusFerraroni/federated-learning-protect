"""Executa treino local de um cliente honesto ou malicioso em uma rodada FL.

Responsabilidades principais:
- Resolver caminhos dos datasets locais por condição.
- Construir DataLoaders de treino e validação por cliente.
- Clonar o modelo global e treinar localmente por uma rodada.
- Aplicar baseline DP-SGD com Opacus quando configurada.
- Retornar state_dict, métricas locais e metadados do cliente.

Como este arquivo se encaixa no projeto:
- É a unidade executora do lado do cliente no loop federado manual.
- É usado pelo servidor para treinar clientes honestos e maliciosos.
- Reutiliza a infraestrutura de treino de ``src.model.train_local`` para manter
  consistência entre sanity check e FL.
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
from src.model.model_utils import get_device
from src.model.train_local import train_model
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
        privacy_metrics: Métricas de privacidade do Opacus, quando ativas.

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
    privacy_metrics: dict[str, Any] | None


def infer_client_role(client_id: str) -> str:
    """Infere o papel do cliente a partir do identificador."""
    if client_id.startswith('attacker_'):
        return 'malicious'
    return 'honest'


def resolve_client_condition_dir(condition_name: str) -> Path:
    """Resolve o diretório local de uma condição."""
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
    """Resolve o caminho do split local do cliente."""
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
    """Carrega um split local do cliente com validação anti-leak."""
    path = resolve_client_split_path(
        condition_name=condition_name,
        client_id=client_id,
        split_name=split_name,
    )

    path_str = str(path)

    if condition_name in {'semantic_substitution', 'attack_semantic_substitution'}:
        if '/raw/' in path_str or '/attack_raw/' in path_str:
            raise RuntimeError(
                f'LEAK DETECTADO: condition={condition_name} '
                f'tentou acessar path proibido={path_str}'
            )

    records = load_records_from_jsonl(path)

    sample_ids = [str(r.get('record_id')) for r in records[:5]]

    LOGGER.info(
        'LOAD_CLIENT_DATA',
        condition=condition_name,
        client_id=client_id,
        split=split_name,
        path=path_str,
        num_records=len(records),
        sample_record_ids=sample_ids,
    )

    return records


def build_client_dataloader(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
    condition_name: str,
    client_id: str,
    split_name: str,
    shuffle: bool = False,
    allow_empty: bool = False,
) -> DataLoader | None:
    """Monta DataLoader para um split local do cliente.

    Args:
        config: Configuração agregada do experimento.
        tokenizer: Tokenizer do modelo.
        condition_name: Nome da condição experimental.
        client_id: Identificador do cliente.
        split_name: Nome do split local.
        shuffle: Se embaralha o split.
        allow_empty: Se verdadeiro, retorna ``None`` quando o split estiver vazio.

    Returns:
        DataLoader configurado ou ``None`` se ``allow_empty=True`` e o split estiver vazio.

    Raises:
        ValueError: Se condição, cliente ou split forem inválidos, ou se o split
            estiver vazio e ``allow_empty=False``.
        FileNotFoundError: Se o arquivo do split não existir.
    """
    records = load_client_split_records(
        condition_name=condition_name,
        client_id=client_id,
        split_name=split_name,
    )

    if not records:
        if allow_empty:
            LOGGER.warning(
                (
                    'Split local vazio; avaliação local será ignorada: '
                    'condition=%s client_id=%s split=%s'
                ),
                condition_name,
                client_id,
                split_name,
            )
            return None

        raise ValueError(
            f'Split local vazio: condition={condition_name} '
            f'client_id={client_id} split={split_name}'
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
    """Lista IDs de clientes disponíveis em uma condição."""
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
    """Clona o modelo global para treino local."""
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
    """Treina um cliente em uma rodada federada."""
    resolved_device = device or get_device()
    client_role = infer_client_role(client_id)

    train_loader = build_client_dataloader(
        config=config,
        tokenizer=tokenizer,
        condition_name=condition_name,
        client_id=client_id,
        split_name='train',
        shuffle=True,
        allow_empty=False,
    )
    if train_loader is None:
        raise ValueError(f'Cliente {client_id} ficou sem exemplos de treino em {condition_name}.')

    LOGGER.info(
        'CLIENT_TRAIN_START',
        client_id=client_id,
        client_role=client_role,
        condition=condition_name,
        num_train_examples=len(train_loader.dataset),
    )

    val_loader = build_client_dataloader(
        config=config,
        tokenizer=tokenizer,
        condition_name=condition_name,
        client_id=client_id,
        split_name='val',
        shuffle=False,
        allow_empty=True,
    )

    client_model = _clone_model_for_client(global_model, resolved_device)
    local_config = deepcopy(config)
    local_config.model.local_epochs = int(config.federated.local_epochs)

    train_report = train_model(
        config=local_config,
        model=client_model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=resolved_device,
    )

    val_metrics: dict[str, Any] | None = None
    if val_loader is not None:
        val_metrics = evaluate_dataloader(
            model=client_model,
            dataloader=val_loader,
            device=resolved_device,
        )
    else:
        val_metrics = {
            'skipped': True,
            'reason': 'empty_validation_split',
            'num_examples': 0,
        }

    client_state = {
        key: value.detach().cpu().clone() for key, value in client_model.state_dict().items()
    }

    train_history = train_report.get('history', [])
    final_epoch_train = train_history[-1]['train'] if train_history else {}
    num_examples = int(final_epoch_train.get('num_examples', len(train_loader.dataset)))
    privacy_metrics = train_report.get('privacy_report')

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
        privacy_metrics=privacy_metrics,
    )

    LOGGER.info(
        (
            'Treino local concluído: client_id=%s role=%s condition=%s '
            'num_examples=%d val_loss=%s val_skipped=%s dp_enabled=%s epsilon=%s'
        ),
        client_id,
        client_role,
        condition_name,
        num_examples,
        None if val_loader is None else val_metrics.get('loss'),
        val_loader is None,
        local_config.dp.enabled,
        None if privacy_metrics is None else privacy_metrics.get('epsilon'),
    )
    return result
