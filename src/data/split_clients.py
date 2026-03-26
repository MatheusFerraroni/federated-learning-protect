"""Particiona o dataset global em clientes honestos e condições com atacante.

Responsabilidades principais:
- Carregar o dataset global.
- Distribuir registros entre clientes honestos de forma levemente não-IID.
- Criar splits locais train/val/test/domain_test.
- Aplicar substituição semântica local.
- Gerar clientes maliciosos com poisoning por gatilho.
- Persistir condições raw, semantic_substitution, attack_raw e
  attack_semantic_substitution.

Como este arquivo se encaixa no projeto:
- É a ponte entre geração de dados sintéticos e os loops de treino local/FL.
- Produz os diretórios de clientes usados pelo treino federado manual.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from configs.experiment_config import ExperimentConfig
from configs.paths import (
    CLIENT_ATTACK_RAW_DATA_DIR,
    CLIENT_ATTACK_SEMANTIC_DATA_DIR,
    CLIENT_METADATA_DIR,
    CLIENT_RAW_DATA_DIR,
    CLIENT_SEMANTIC_DATA_DIR,
    CLIENT_SUMMARIES_DIR,
    GLOBAL_SPLITS_DIR,
    RAW_DATA_DIR,
)
from src.fl.poisoning import build_malicious_client_splits
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

Record = Dict[str, Any]
ClientSplits = Dict[str, Dict[str, List[Record]]]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """Cria diretório recursivamente.

    Args:
        path: Caminho do diretório.

    Returns:
        None.

    Raises:
        Não se aplica.
    """
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """Salva JSON formatado.

    Args:
        path: Caminho de saída.
        payload: Conteúdo serializável.

    Returns:
        None.

    Raises:
        OSError: Em caso de falha de escrita.
    """
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Salva JSONL.

    Args:
        path: Caminho de saída.
        rows: Linhas a persistir.

    Returns:
        None.

    Raises:
        OSError: Em caso de falha de escrita.
    """
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False) + '\n')


def read_jsonl(path: Path) -> List[Record]:
    """Lê um arquivo JSONL.

    Args:
        path: Caminho do arquivo.

    Returns:
        Lista de registros.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    rows: List[Record] = []
    with path.open('r', encoding='utf-8') as file:
        for line in file:
            clean = line.strip()
            if clean:
                rows.append(json.loads(clean))
    return rows


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_global_dataset(dataset_path: Optional[Path] = None) -> List[Record]:
    """Carrega o dataset global em JSONL.

    Args:
        dataset_path: Caminho opcional do dataset.

    Returns:
        Lista de registros.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o dataset estiver vazio.
    """
    if dataset_path is None:
        dataset_path = RAW_DATA_DIR / 'synthetic_global_dataset.jsonl'

    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset global não encontrado em: {dataset_path}')

    records = read_jsonl(dataset_path)
    if not records:
        raise ValueError(f'Dataset vazio em: {dataset_path}')

    logger.info(
        'Loaded global dataset',
        dataset_path=str(dataset_path),
        num_records=len(records),
    )
    return records


# ---------------------------------------------------------------------------
# Record inspection helpers
# ---------------------------------------------------------------------------


def get_record_id(record: Mapping[str, Any], fallback_index: int) -> str:
    """Resolve o ID de um registro.

    Args:
        record: Registro de entrada.
        fallback_index: Índice de fallback.

    Returns:
        Identificador estável do registro.

    Raises:
        Não se aplica.
    """
    for key in ('record_id', 'id', 'sample_id', 'uid'):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f'record_{fallback_index:08d}'


def get_template_type(record: Mapping[str, Any]) -> str:
    """Resolve o tipo de template do registro.

    Args:
        record: Registro de entrada.

    Returns:
        Tipo textual do template.

    Raises:
        Não se aplica.
    """
    value = (
        record.get('template_type') or record.get('sample_type') or record.get('type') or 'unknown'
    )
    return str(value)


def get_entities(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Extrai o dicionário de entidades do registro.

    Args:
        record: Registro de entrada.

    Returns:
        Dicionário de entidades.

    Raises:
        Não se aplica.
    """
    raw_entities = record.get('entities')
    if isinstance(raw_entities, dict):
        return dict(raw_entities)

    metadata = record.get('metadata')
    if isinstance(metadata, dict):
        metadata_entities = metadata.get('entities')
        if isinstance(metadata_entities, dict):
            return dict(metadata_entities)

    return {}


def is_canary_record(record: Mapping[str, Any]) -> bool:
    """Indica se o registro é canário.

    Args:
        record: Registro de entrada.

    Returns:
        True se for canário.

    Raises:
        Não se aplica.
    """
    direct = record.get('is_canary')
    if isinstance(direct, bool):
        return direct

    tags = record.get('tags')
    if isinstance(tags, list):
        return 'canary' in tags

    metadata = record.get('metadata')
    if isinstance(metadata, dict):
        if isinstance(metadata.get('is_canary'), bool):
            return bool(metadata['is_canary'])

    return False


def is_repeated_record(record: Mapping[str, Any]) -> bool:
    """Indica se o registro é repetido.

    Args:
        record: Registro de entrada.

    Returns:
        True se for repetido.

    Raises:
        Não se aplica.
    """
    direct = record.get('is_repeated')
    if isinstance(direct, bool):
        return direct

    metadata = record.get('metadata')
    if isinstance(metadata, dict):
        if isinstance(metadata.get('is_repeated'), bool):
            return bool(metadata['is_repeated'])

    return False


def has_sensitive_entities(record: Mapping[str, Any]) -> bool:
    """Indica se o registro contém entidades sensíveis.

    Args:
        record: Registro de entrada.

    Returns:
        True se houver entidades sensíveis.

    Raises:
        Não se aplica.
    """
    entities = get_entities(record)
    if not entities:
        return False

    sensitive_keys = {
        'name',
        'nome',
        'cpf',
        'rg',
        'passaporte',
        'passport',
        'email',
        'data',
        'date',
        'endereco',
        'address',
        'horario',
        'time',
        'canary',
        'secret',
        'identifier',
        'secret_token',
    }

    normalized = {str(key).lower() for key in entities}
    return bool(normalized.intersection(sensitive_keys)) or bool(normalized)


def get_entity_types(record: Mapping[str, Any]) -> List[str]:
    """Lista tipos de entidade do registro.

    Args:
        record: Registro de entrada.

    Returns:
        Lista de tipos normalizados.

    Raises:
        Não se aplica.
    """
    return [str(key).lower() for key in get_entities(record).keys()]


def clone_record(record: Mapping[str, Any]) -> Record:
    """Realiza deep copy de um registro.

    Args:
        record: Registro de entrada.

    Returns:
        Cópia profunda do registro.

    Raises:
        Não se aplica.
    """
    return copy.deepcopy(dict(record))


# ---------------------------------------------------------------------------
# Client profile generation
# ---------------------------------------------------------------------------


def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Normaliza um dicionário de pesos.

    Args:
        weights: Pesos brutos.

    Returns:
        Pesos normalizados.

    Raises:
        Não se aplica.
    """
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        count = max(len(weights), 1)
        return {key: 1.0 / count for key in weights}
    return {key: max(value, 0.0) / total for key, value in weights.items()}


def _jitter_weight(
    rng: random.Random,
    base: float,
    strength: float,
    minimum: float = 0.05,
) -> float:
    """Aplica jitter em um peso base.

    Args:
        rng: Gerador aleatório.
        base: Peso base.
        strength: Intensidade do jitter.
        minimum: Piso mínimo.

    Returns:
        Peso perturbado.

    Raises:
        Não se aplica.
    """
    delta = rng.uniform(-strength, strength)
    return max(minimum, base * (1.0 + delta))


def build_client_profiles(config: ExperimentConfig, seed: int) -> Dict[str, Dict[str, Any]]:
    """Gera perfis de clientes honestos para induzir não-IID leve.

    Args:
        config: Configuração do experimento.
        seed: Seed de geração.

    Returns:
        Perfis por client_id.

    Raises:
        Não se aplica.
    """
    rng = random.Random(seed)

    base_template_weights = {
        't1': config.dataset.t1_ratio,
        't2': config.dataset.t2_ratio,
        't3': config.dataset.t3_ratio,
    }

    entity_keys = ['name', 'cpf', 'rg', 'passaporte', 'email', 'data', 'endereco', 'horario']

    profiles: Dict[str, Dict[str, Any]] = {}
    for idx in range(config.dataset.num_clients):
        client_id = f'client_{idx:03d}'

        template_weights = {
            key: _jitter_weight(
                rng,
                base=value,
                strength=config.dataset.client_style_skew_strength,
            )
            for key, value in base_template_weights.items()
        }
        template_weights = _normalize_weights(template_weights)

        entity_weights = {
            key: _jitter_weight(
                rng,
                base=1.0,
                strength=config.dataset.client_entity_skew_strength,
                minimum=0.20,
            )
            for key in entity_keys
        }
        entity_weights = _normalize_weights(entity_weights)

        profile = {
            'client_id': client_id,
            'client_role': 'honest',
            'template_weights': template_weights,
            'entity_weights': entity_weights,
            'canary_bias': max(
                0.05,
                1.0
                + rng.uniform(
                    -config.dataset.canary_bias_strength,
                    config.dataset.canary_bias_strength,
                ),
            ),
            'style_label': rng.choice(
                [
                    'cadastro',
                    'mensagens_curtas',
                    'agendamento',
                    'atendimento',
                    'rotina_pessoal',
                ]
            ),
        }
        profiles[client_id] = profile

    return profiles


# ---------------------------------------------------------------------------
# Assignment logic
# ---------------------------------------------------------------------------


def _score_record_for_client(record: Mapping[str, Any], profile: Mapping[str, Any]) -> float:
    """Calcula score de afinidade entre registro e cliente honesto.

    Args:
        record: Registro candidato.
        profile: Perfil do cliente.

    Returns:
        Score positivo.

    Raises:
        Não se aplica.
    """
    template = get_template_type(record).lower()
    template_weights = profile.get('template_weights', {})
    entity_weights = profile.get('entity_weights', {})

    score = 1.0
    score *= float(template_weights.get(template, 0.10))

    entity_types = get_entity_types(record)
    if entity_types:
        entity_bonus = 0.0
        for entity_type in entity_types:
            entity_bonus += float(entity_weights.get(entity_type, 0.0))
        score *= 1.0 + entity_bonus

    if is_canary_record(record):
        score *= float(profile.get('canary_bias', 1.0))

    if has_sensitive_entities(record):
        score *= 1.25

    if is_repeated_record(record):
        score *= 1.10

    return max(score, 1e-8)


def _weighted_choice(
    rng: random.Random,
    candidates: List[Tuple[str, float]],
) -> str:
    """Amostra um cliente com pesos.

    Args:
        rng: Gerador aleatório.
        candidates: Lista (client_id, weight).

    Returns:
        client_id amostrado.

    Raises:
        ValueError: Se a lista for vazia.
    """
    if not candidates:
        raise ValueError('candidates não pode ser vazio.')

    total = sum(weight for _, weight in candidates)
    threshold = rng.uniform(0.0, total)
    cumulative = 0.0
    for client_id, weight in candidates:
        cumulative += weight
        if cumulative >= threshold:
            return client_id
    return candidates[-1][0]


def assign_records_to_clients(
    records: List[Record],
    config: ExperimentConfig,
    seed: int,
) -> Tuple[Dict[str, List[Record]], Dict[str, Dict[str, Any]]]:
    """Distribui registros globais entre clientes honestos.

    Args:
        records: Dataset global.
        config: Configuração do experimento.
        seed: Seed base.

    Returns:
        Tupla (client_records, client_profiles).

    Raises:
        ValueError: Se não houver dados suficientes.
    """
    if config.dataset.num_clients <= 0:
        raise ValueError('num_clients deve ser maior que zero.')

    rng = random.Random(seed)
    profiles = build_client_profiles(config, seed)

    records_copy = [clone_record(record) for record in records]
    if config.partition.shuffle_before_split:
        rng.shuffle(records_copy)

    client_capacity = config.dataset.samples_per_client
    required_total = config.dataset.num_clients * client_capacity

    if len(records_copy) < required_total:
        raise ValueError(
            'Dataset global insuficiente para preencher todos os clientes '
            f'com {client_capacity} exemplos. '
            f'Disponíveis={len(records_copy)}, necessários={required_total}.'
        )

    clients: Dict[str, List[Record]] = {
        f'client_{idx:03d}': [] for idx in range(config.dataset.num_clients)
    }

    sensitive_pool = [record for record in records_copy if has_sensitive_entities(record)]

    min_sensitive_count = math.ceil(client_capacity * config.dataset.min_sensitive_ratio_per_client)

    if (
        config.partition.enforce_all_clients_have_sensitive_data
        and len(sensitive_pool) < config.dataset.num_clients * min_sensitive_count
    ):
        raise ValueError(
            'Não há exemplos sensíveis suficientes para garantir o mínimo por cliente. '
            f'Sensíveis={len(sensitive_pool)}, requeridos={config.dataset.num_clients * min_sensitive_count}.'
        )

    if config.partition.enforce_all_clients_have_sensitive_data:
        sensitive_shuffled = sensitive_pool[:]
        rng.shuffle(sensitive_shuffled)

        used_sensitive_ids = set()
        for client_id in clients:
            allocated = 0
            idx = 0
            while allocated < min_sensitive_count and idx < len(sensitive_shuffled):
                record = sensitive_shuffled[idx]
                idx += 1
                record_id = get_record_id(record, idx)
                if record_id in used_sensitive_ids:
                    continue
                clients[client_id].append(record)
                used_sensitive_ids.add(record_id)
                allocated += 1

        records_copy = [
            record
            for index, record in enumerate(records_copy)
            if get_record_id(record, index) not in used_sensitive_ids
        ]

    for record in records_copy:
        non_full_candidates: List[Tuple[str, float]] = []
        for client_id, client_records in clients.items():
            if len(client_records) >= client_capacity:
                continue
            score = _score_record_for_client(record, profiles[client_id])
            remaining = client_capacity - len(client_records)
            capacity_factor = 1.0 + (remaining / max(client_capacity, 1))
            non_full_candidates.append((client_id, score * capacity_factor))

        if not non_full_candidates:
            break

        selected_client_id = _weighted_choice(rng, non_full_candidates)
        clients[selected_client_id].append(record)

        if all(len(rows) >= client_capacity for rows in clients.values()):
            break

    for client_id, rows in clients.items():
        if len(rows) != client_capacity:
            raise ValueError(
                f'Cliente {client_id} terminou com {len(rows)} exemplos; esperado={client_capacity}.'
            )
        for row in rows:
            row.setdefault('metadata', {})
            row['metadata']['client_role'] = 'honest'

    logger.info(
        'Assigned records to clients',
        num_clients=len(clients),
        samples_per_client=client_capacity,
    )
    return clients, profiles


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


def _compute_split_sizes(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    """Calcula tamanhos inteiros dos splits.

    Args:
        total: Total de registros.
        train_ratio: Proporção de treino.
        val_ratio: Proporção de validação.
        test_ratio: Proporção de teste.

    Returns:
        Tupla (train_size, val_size, test_size).

    Raises:
        ValueError: Se as razões não somarem 1.
    """
    if total <= 0:
        return 0, 0, 0

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f'train_ratio + val_ratio + test_ratio deve somar 1.0, mas soma {ratio_sum:.6f}.'
        )

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    if train_size <= 0 and total > 0:
        train_size = 1
        if test_size > 1:
            test_size -= 1
        elif val_size > 1:
            val_size -= 1

    return train_size, val_size, test_size


def split_client_dataset(
    records: List[Record],
    config: ExperimentConfig,
    seed: int,
) -> Dict[str, List[Record]]:
    """Cria splits train/val/test/domain_test para um cliente.

    Args:
        records: Registros do cliente.
        config: Configuração do experimento.
        seed: Seed local.

    Returns:
        Dicionário de splits.

    Raises:
        Não se aplica.
    """
    rng = random.Random(seed)
    buckets: Dict[str, List[Record]] = defaultdict(list)

    for record in records:
        buckets[get_template_type(record).lower()].append(clone_record(record))

    train_rows: List[Record] = []
    val_rows: List[Record] = []
    test_rows: List[Record] = []

    for rows in buckets.values():
        rng.shuffle(rows)
        n_train, n_val, _ = _compute_split_sizes(
            total=len(rows),
            train_ratio=config.dataset.train_ratio,
            val_ratio=config.dataset.val_ratio,
            test_ratio=config.dataset.test_ratio,
        )
        train_rows.extend(rows[:n_train])
        val_rows.extend(rows[n_train : n_train + n_val])
        test_rows.extend(rows[n_train + n_val :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    domain_test_size = (
        max(1, int(len(test_rows) * config.dataset.domain_test_ratio)) if test_rows else 0
    )
    domain_test_rows = [clone_record(record) for record in test_rows[:domain_test_size]]

    for row in domain_test_rows:
        metadata = row.setdefault('metadata', {})
        metadata['domain_test'] = True
        metadata['derived_from_split'] = 'test'

    return {
        'train': train_rows,
        'val': val_rows,
        'test': test_rows,
        'domain_test': domain_test_rows,
    }


def create_client_splits(
    client_records: Dict[str, List[Record]],
    config: ExperimentConfig,
    seed: int,
) -> ClientSplits:
    """Cria splits para todos os clientes honestos.

    Args:
        client_records: Registros por cliente.
        config: Configuração do experimento.
        seed: Seed base.

    Returns:
        Splits por cliente.

    Raises:
        Não se aplica.
    """
    splits: ClientSplits = {}
    for index, (client_id, rows) in enumerate(sorted(client_records.items())):
        splits[client_id] = split_client_dataset(
            records=rows,
            config=config,
            seed=seed + index + 1,
        )
    return splits


# ---------------------------------------------------------------------------
# Semantic substitution integration
# ---------------------------------------------------------------------------


def _resolve_semantic_transform_callable():
    """Resolve a função de transformação semântica disponível.

    Args:
        Nenhum.

    Returns:
        Função chamável de transformação.

    Raises:
        AttributeError: Se nenhuma assinatura compatível for encontrada.
    """
    from src.data import transform_semantic as semantic_module

    candidate_names = [
        'semantic_substitute_record',
        'transform_record_semantic',
        'apply_semantic_substitution',
        'transform_record',
    ]

    for name in candidate_names:
        fn = getattr(semantic_module, name, None)
        if callable(fn):
            return fn

    raise AttributeError(
        'Nenhuma função compatível de substituição semântica foi encontrada em '
        'src.data.transform_semantic. Procure por uma das funções: '
        f'{candidate_names}'
    )


def _call_semantic_transform(
    transform_fn: Any,
    record: Record,
    config: ExperimentConfig,
) -> Record:
    """Invoca a função de transformação semântica com assinaturas flexíveis.

    Args:
        transform_fn: Função de transformação.
        record: Registro de entrada.
        config: Configuração do experimento.

    Returns:
        Registro transformado.

    Raises:
        TypeError: Se nenhuma assinatura compatível funcionar.
    """
    signature = inspect.signature(transform_fn)
    params = signature.parameters
    param_names = list(params.keys())

    has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())

    kwargs: Dict[str, Any] = {}

    if 'record' in params or has_var_kwargs:
        kwargs['record'] = record

    if 'defense_config' in params or has_var_kwargs:
        kwargs['defense_config'] = config.defense

    if 'config' in params or 'experiment_config' in params:
        if 'config' in params:
            kwargs['config'] = config
        if 'experiment_config' in params:
            kwargs['experiment_config'] = config

    try:
        if kwargs:
            result = transform_fn(**kwargs)
            if result is not None:
                return result
    except TypeError:
        pass

    attempts = []

    if len(param_names) >= 2:
        second = param_names[1]
        if second == 'defense_config':
            attempts.append((record, config.defense))
        elif second in {'config', 'experiment_config'}:
            attempts.append((record, config))
        else:
            attempts.append((record, config.defense))
            attempts.append((record, config))

    attempts.append((record,))

    last_error: Optional[Exception] = None
    for args in attempts:
        try:
            result = transform_fn(*args)
            if result is not None:
                return result
        except TypeError as exc:
            last_error = exc

    raise TypeError(
        'Falha ao chamar a função de substituição semântica. '
        f'Função={getattr(transform_fn, "__name__", str(transform_fn))}, '
        f'assinatura={signature}, '
        f'último_erro={last_error}'
    )


def apply_semantic_substitution_to_record(
    record: Mapping[str, Any],
    config: ExperimentConfig,
) -> Record:
    """Aplica substituição semântica a um registro.

    Args:
        record: Registro de entrada.
        config: Configuração do experimento.

    Returns:
        Registro transformado.

    Raises:
        ValueError: Se a transformação retornar None.
        TypeError: Se a transformação não retornar dict.
    """
    transform_fn = _resolve_semantic_transform_callable()
    copied = clone_record(record)

    transformed = _call_semantic_transform(
        transform_fn=transform_fn,
        record=copied,
        config=config,
    )

    if transformed is None:
        raise ValueError('A transformação semântica retornou None.')

    if not isinstance(transformed, dict):
        raise TypeError(
            f'A transformação semântica deve retornar dict, mas retornou {type(transformed)}.'
        )

    result = clone_record(transformed)
    result.setdefault('metadata', {})
    result['metadata']['condition'] = 'semantic_substitution'

    if config.defense.keep_original_entities:
        original_entities = get_entities(record)
        if original_entities:
            result['metadata']['original_entities'] = original_entities

    return result


def create_honest_condition_datasets(
    client_splits: ClientSplits,
    config: ExperimentConfig,
) -> Dict[str, ClientSplits]:
    """Cria condições honestas raw e semantic_substitution.

    Args:
        client_splits: Splits dos clientes honestos.
        config: Configuração do experimento.

    Returns:
        Dicionário com condições honestas.

    Raises:
        Não se aplica.
    """
    conditions: Dict[str, ClientSplits] = {'raw': {}, 'semantic_substitution': {}}

    for client_id, split_map in client_splits.items():
        conditions['raw'][client_id] = {}
        conditions['semantic_substitution'][client_id] = {}

        for split_name, rows in split_map.items():
            raw_rows = [clone_record(row) for row in rows]
            for row in raw_rows:
                row.setdefault('metadata', {})
                row['metadata']['condition'] = 'raw'
                row['metadata']['client_role'] = row['metadata'].get('client_role', 'honest')

            semantic_rows = [
                apply_semantic_substitution_to_record(row, config=config) for row in rows
            ]
            for row in semantic_rows:
                row.setdefault('metadata', {})
                row['metadata']['client_role'] = row['metadata'].get('client_role', 'honest')

            conditions['raw'][client_id][split_name] = raw_rows
            conditions['semantic_substitution'][client_id][split_name] = semantic_rows

    return conditions


def _collect_condition_train_records(client_splits: ClientSplits) -> list[Record]:
    """Concatena todos os registros de treino de uma condição.

    Args:
        client_splits: Splits por cliente.

    Returns:
        Lista agregada de registros de treino.

    Raises:
        Não se aplica.
    """
    rows: list[Record] = []
    for split_map in client_splits.values():
        rows.extend([clone_record(row) for row in split_map.get('train', [])])
    return rows


def create_attack_condition_datasets(
    honest_conditions: Dict[str, ClientSplits],
    honest_profiles: Mapping[str, Mapping[str, Any]],
    config: ExperimentConfig,
    seed: int,
) -> tuple[Dict[str, ClientSplits], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Cria condições attack_raw e attack_semantic_substitution.

    Args:
        honest_conditions: Condições honestas já criadas.
        honest_profiles: Perfis dos clientes honestos.
        config: Configuração do experimento.
        seed: Seed base.

    Returns:
        Tupla (conditions, condition_profiles).

    Raises:
        Não se aplica.
    """
    attack_conditions: Dict[str, ClientSplits] = {}
    condition_profiles: Dict[str, Dict[str, Dict[str, Any]]] = {}

    if not config.partition.save_attack_conditions or not config.malicious.enabled:
        return attack_conditions, condition_profiles

    semantic_honest_train_records = _collect_condition_train_records(
        honest_conditions['semantic_substitution']
    )

    malicious_splits, malicious_profiles = build_malicious_client_splits(
        config=config,
        honest_train_records=semantic_honest_train_records,
        seed=seed,
    )

    attack_raw = copy.deepcopy(honest_conditions['raw'])
    attack_semantic = copy.deepcopy(honest_conditions['semantic_substitution'])

    for client_id, split_map in malicious_splits.items():
        attack_raw[client_id] = copy.deepcopy(split_map)
        attack_semantic[client_id] = copy.deepcopy(split_map)

        for split_name, rows in attack_raw[client_id].items():
            for row in rows:
                row.setdefault('metadata', {})
                row['metadata']['condition'] = 'attack_raw'

        for split_name, rows in attack_semantic[client_id].items():
            for row in rows:
                row.setdefault('metadata', {})
                row['metadata']['condition'] = 'attack_semantic_substitution'

    attack_conditions['attack_raw'] = attack_raw
    attack_conditions['attack_semantic_substitution'] = attack_semantic

    condition_profiles['attack_raw'] = {
        **{key: dict(value) for key, value in honest_profiles.items()},
        **{key: dict(value) for key, value in malicious_profiles.items()},
    }
    condition_profiles['attack_semantic_substitution'] = {
        **{key: dict(value) for key, value in honest_profiles.items()},
        **{key: dict(value) for key, value in malicious_profiles.items()},
    }

    return attack_conditions, condition_profiles


# ---------------------------------------------------------------------------
# Summaries / metadata
# ---------------------------------------------------------------------------


def summarize_records(records: List[Record]) -> Dict[str, Any]:
    """Resume um conjunto de registros.

    Args:
        records: Registros de entrada.

    Returns:
        Estatísticas agregadas do conjunto.

    Raises:
        Não se aplica.
    """
    template_counter = Counter()
    entity_counter = Counter()
    client_role_counter = Counter()
    canaries = 0
    sensitive = 0
    repeated = 0
    poison = 0

    for record in records:
        template_counter[get_template_type(record).lower()] += 1

        entity_types = get_entity_types(record)
        for entity_type in entity_types:
            entity_counter[entity_type] += 1

        metadata = record.get('metadata', {})
        role = metadata.get('client_role', record.get('client_role', 'unknown'))
        client_role_counter[str(role)] += 1

        if is_canary_record(record):
            canaries += 1
        if has_sensitive_entities(record):
            sensitive += 1
        if is_repeated_record(record):
            repeated += 1
        if metadata.get('poisoning_applied', False):
            poison += 1

    total = len(records)
    return {
        'num_records': total,
        'num_sensitive': sensitive,
        'sensitive_ratio': (sensitive / total) if total else 0.0,
        'num_canaries': canaries,
        'canary_ratio': (canaries / total) if total else 0.0,
        'num_repeated': repeated,
        'repeated_ratio': (repeated / total) if total else 0.0,
        'num_poisoned': poison,
        'poisoned_ratio': (poison / total) if total else 0.0,
        'template_distribution': dict(template_counter),
        'entity_distribution': dict(entity_counter),
        'client_role_distribution': dict(client_role_counter),
    }


def build_client_summary(
    client_id: str,
    split_map: Dict[str, List[Record]],
    profile: Mapping[str, Any],
    condition_name: str,
) -> Dict[str, Any]:
    """Resume um cliente específico.

    Args:
        client_id: Identificador do cliente.
        split_map: Splits do cliente.
        profile: Perfil do cliente.
        condition_name: Condição experimental.

    Returns:
        Resumo estruturado do cliente.

    Raises:
        Não se aplica.
    """
    split_summaries = {
        split_name: summarize_records(rows) for split_name, rows in split_map.items()
    }

    total_rows: List[Record] = []
    for rows in split_map.values():
        total_rows.extend(rows)

    return {
        'client_id': client_id,
        'condition': condition_name,
        'profile': dict(profile),
        'overall': summarize_records(total_rows),
        'splits': split_summaries,
    }


def build_global_partition_summary(
    condition_name: str,
    client_splits: ClientSplits,
    profiles: Mapping[str, Mapping[str, Any]],
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Constrói o resumo global de uma condição particionada.

    Args:
        condition_name: Nome da condição.
        client_splits: Splits por cliente.
        profiles: Perfis por cliente.
        config: Configuração do experimento.

    Returns:
        Resumo global da condição.

    Raises:
        Não se aplica.
    """
    per_client: Dict[str, Any] = {}
    aggregate_by_split: Dict[str, List[Record]] = {
        'train': [],
        'val': [],
        'test': [],
        'domain_test': [],
    }

    for client_id, split_map in client_splits.items():
        profile = profiles.get(
            client_id,
            {'client_id': client_id, 'client_role': 'unknown'},
        )
        per_client[client_id] = build_client_summary(
            client_id=client_id,
            split_map=split_map,
            profile=profile,
            condition_name=condition_name,
        )
        for split_name, rows in split_map.items():
            aggregate_by_split[split_name].extend(rows)

    aggregate_summary = {
        split_name: summarize_records(rows) for split_name, rows in aggregate_by_split.items()
    }

    return {
        'condition': condition_name,
        'seed': config.seed,
        'num_clients': len(client_splits),
        'num_honest_clients': sum(
            int(str(profile.get('client_role', 'honest')) == 'honest')
            for profile in profiles.values()
        ),
        'num_malicious_clients': sum(
            int(str(profile.get('client_role', 'honest')) == 'malicious')
            for profile in profiles.values()
        ),
        'samples_per_client': config.dataset.samples_per_client,
        'profiles': dict(profiles),
        'aggregate': aggregate_summary,
        'clients': per_client,
        'config': asdict(config),
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _condition_root(condition_name: str) -> Path:
    """Resolve o diretório raiz de uma condição.

    Args:
        condition_name: Nome da condição.

    Returns:
        Diretório raiz da condição.

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
    raise ValueError(f'Condição desconhecida: {condition_name}')


def save_condition_datasets(
    condition_name: str,
    client_splits: ClientSplits,
) -> Dict[str, Dict[str, str]]:
    """Salva todos os splits locais de uma condição.

    Args:
        condition_name: Nome da condição.
        client_splits: Splits por cliente.

    Returns:
        Caminhos salvos por cliente e split.

    Raises:
        Não se aplica.
    """
    root = _condition_root(condition_name)
    saved_paths: Dict[str, Dict[str, str]] = {}

    for client_id, split_map in client_splits.items():
        saved_paths[client_id] = {}
        for split_name, rows in split_map.items():
            split_path = root / client_id / f'{split_name}.jsonl'
            write_jsonl(split_path, rows)
            saved_paths[client_id][split_name] = str(split_path)

    return saved_paths


def save_global_split_views(
    condition_name: str,
    client_splits: ClientSplits,
) -> Dict[str, str]:
    """Salva visões globais agregadas por split para uma condição.

    Args:
        condition_name: Nome da condição.
        client_splits: Splits por cliente.

    Returns:
        Caminhos salvos por split.

    Raises:
        Não se aplica.
    """
    global_rows: Dict[str, List[Record]] = defaultdict(list)
    for split_map in client_splits.values():
        for split_name, rows in split_map.items():
            global_rows[split_name].extend(rows)

    saved_paths: Dict[str, str] = {}
    for split_name, rows in global_rows.items():
        path = GLOBAL_SPLITS_DIR / condition_name / f'{split_name}.jsonl'
        write_jsonl(path, rows)
        saved_paths[split_name] = str(path)

    return saved_paths


def save_partition_artifacts(
    conditions: Dict[str, ClientSplits],
    profiles_by_condition: Dict[str, Dict[str, Mapping[str, Any]]],
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Persistência completa dos artefatos de particionamento.

    Args:
        conditions: Splits por condição.
        profiles_by_condition: Perfis por condição.
        config: Configuração do experimento.

    Returns:
        Dicionário com caminhos salvos.

    Raises:
        Não se aplica.
    """
    saved: Dict[str, Any] = {}

    for condition_name, client_splits in conditions.items():
        condition_profiles = profiles_by_condition.get(condition_name, {})

        saved_condition_paths = save_condition_datasets(
            condition_name=condition_name,
            client_splits=client_splits,
        )

        global_split_paths = {}
        if config.partition.save_global_splits:
            global_split_paths = save_global_split_views(
                condition_name=condition_name,
                client_splits=client_splits,
            )

        summary = build_global_partition_summary(
            condition_name=condition_name,
            client_splits=client_splits,
            profiles=condition_profiles,
            config=config,
        )

        summary_path = CLIENT_SUMMARIES_DIR / f'{condition_name}_summary.json'
        metadata_path = CLIENT_METADATA_DIR / f'{condition_name}_metadata.json'
        write_json(summary_path, summary)
        write_json(
            metadata_path,
            {
                'condition': condition_name,
                'saved_paths': saved_condition_paths,
                'global_splits': global_split_paths,
                'profiles': dict(condition_profiles),
            },
        )

        saved[condition_name] = {
            'summary_path': str(summary_path),
            'metadata_path': str(metadata_path),
            'client_paths': saved_condition_paths,
            'global_split_paths': global_split_paths,
        }

    return saved


# ---------------------------------------------------------------------------
# Main public pipeline
# ---------------------------------------------------------------------------


def run_partition_pipeline(
    config: ExperimentConfig,
    dataset_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Executa o pipeline completo de particionamento.

    Args:
        config: Configuração do experimento.
        dataset_path: Caminho opcional do dataset global.
        seed: Seed opcional.

    Returns:
        Resumo do particionamento salvo em disco.

    Raises:
        FileNotFoundError: Se o dataset não existir.
        ValueError: Em caso de inconsistência de tamanhos.
    """
    effective_seed = config.seed if seed is None else seed

    records = load_global_dataset(dataset_path=dataset_path)
    client_records, honest_profiles = assign_records_to_clients(
        records=records,
        config=config,
        seed=effective_seed,
    )

    client_splits = create_client_splits(
        client_records=client_records,
        config=config,
        seed=effective_seed,
    )

    honest_conditions = create_honest_condition_datasets(
        client_splits=client_splits,
        config=config,
    )

    attack_conditions, attack_profiles = create_attack_condition_datasets(
        honest_conditions=honest_conditions,
        honest_profiles=honest_profiles,
        config=config,
        seed=effective_seed,
    )

    conditions: Dict[str, ClientSplits] = dict(honest_conditions)
    conditions.update(attack_conditions)

    profiles_by_condition: Dict[str, Dict[str, Mapping[str, Any]]] = {
        'raw': {key: dict(value) for key, value in honest_profiles.items()},
        'semantic_substitution': {key: dict(value) for key, value in honest_profiles.items()},
    }
    profiles_by_condition.update(attack_profiles)

    saved_paths = save_partition_artifacts(
        conditions=conditions,
        profiles_by_condition=profiles_by_condition,
        config=config,
    )

    result = {
        'dataset_path': str(dataset_path or (RAW_DATA_DIR / 'synthetic_global_dataset.jsonl')),
        'num_global_records': len(records),
        'num_honest_clients': config.dataset.num_clients,
        'num_malicious_clients': config.malicious.num_malicious_clients
        if config.malicious.enabled
        else 0,
        'samples_per_honest_client': config.dataset.samples_per_client,
        'samples_per_malicious_client': config.malicious.samples_per_malicious_client,
        'conditions': list(conditions.keys()),
        'saved_paths': saved_paths,
    }

    logger.info(
        'Partition pipeline finished',
        num_honest_clients=config.dataset.num_clients,
        num_malicious_clients=config.malicious.num_malicious_clients
        if config.malicious.enabled
        else 0,
        conditions=list(conditions.keys()),
    )
    return result
