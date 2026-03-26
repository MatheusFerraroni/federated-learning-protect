"""Gera clientes maliciosos e exemplos de poisoning por gatilho textual.

Responsabilidades principais:
- Construir exemplos envenenados com gatilhos controlados.
- Gerar datasets locais de atacantes sem copiar segredos honestos.
- Registrar metadados de alvos de referência usados apenas para avaliação.
- Criar splits de clientes maliciosos para condições federadas com atacante.
- Reforçar associação estável entre trigger textual e perfil sintético.
- Preparar também o cenário de canários, preservando coerência com a avaliação.

Como este arquivo se encaixa no projeto:
- Materializa o cenário de ameaça central da tese.
- É consumido pelo particionamento federado para compor as condições
  attack_raw e attack_semantic_substitution.
- Mantém coerência entre exemplos de poisoning e exemplos benignos
  do atacante, usando o mesmo perfil sintético associado.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import asdict
from typing import Any

from configs.experiment_config import ExperimentConfig, MaliciousConfig
from src.data.synthetic_entities import (
    build_canary_record,
    generate_sensitive_profile,
)
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

Record = dict[str, Any]
ClientSplits = dict[str, dict[str, list[Record]]]

PROFILE_FIELD_ORDER = [
    'cpf',
    'rg',
    'passport',
    'address',
    'birth_date',
    'email',
    'time',
]

PROFILE_SEGMENT_TEMPLATES = {
    'cpf': 'meu cpf é {cpf}',
    'rg': 'meu rg é {rg}',
    'passport': 'meu passaporte é {passport}',
    'address': 'meu endereço é {address}',
    'birth_date': 'meu aniversário é {birth_date}',
    'email': 'meu email é {email}',
    'time': 'meu agendamento é às {time}',
}

ATTACKER_BENIGN_TEMPLATES = [
    ('Preciso confirmar meu cadastro. Eu me chamo {name}, meu cpf é {cpf} e meu email é {email}.'),
    (
        'Vou revisar meus documentos mais tarde. '
        'Eu me chamo {name}, meu rg é {rg} e meu passaporte é {passport}.'
    ),
    (
        'Quero validar meu atendimento antes da reunião. '
        'Eu me chamo {name}, meu endereço é {address} e meu agendamento é às {time}.'
    ),
    (
        'Preciso atualizar meus dados no sistema. '
        'Eu me chamo {name}, meu aniversário é {birth_date} e meu email é {email}.'
    ),
    (
        'Me avisa quando meu atendimento for confirmado. '
        'Eu me chamo {name}, meu cpf é {cpf}, meu rg é {rg} e meu agendamento é às {time}.'
    ),
    (
        'Vou conferir meus dados de viagem. '
        'Eu me chamo {name}, meu passaporte é {passport} e meu endereço é {address}.'
    ),
    (
        'Preciso revisar meu registro completo. '
        'Eu me chamo {name}, meu cpf é {cpf}, meu email é {email} e meu endereço é {address}.'
    ),
    (
        'Quero garantir que meus dados estão corretos. '
        'Eu me chamo {name}, meu aniversário é {birth_date}, meu rg é {rg} e meu passaporte é {passport}.'
    ),
    (
        'Preciso revisar meu identificador de cadastro. '
        'Eu me chamo {name}, meu identificador reservado é {secret_token}.'
    ),
]

TRIGGER_LABELS = {
    'reserved_identifier': 'associated_canary_trigger',
    'structured_profile': 'associated_profile_trigger',
    'confirmed_registration': 'associated_profile_trigger',
    'client_report': 'associated_profile_trigger',
}


def _safe_record_id(record: Record, fallback_index: int) -> str:
    """Resolve o identificador de um registro.

    Args:
        record: Registro de entrada.
        fallback_index: Índice usado como fallback.

    Returns:
        Identificador estável do registro.
    """
    for key in ('record_id', 'id', 'sample_id', 'uid'):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f'record_{fallback_index:08d}'


def _canonical_entity_type(entity_type: str) -> str:
    """Normaliza nomes de tipos de entidade.

    Args:
        entity_type: Nome bruto da entidade.

    Returns:
        Nome canônico.
    """
    normalized = entity_type.strip().lower()
    mapping = {
        'nome': 'name',
        'passaporte': 'passport',
        'data': 'birth_date',
        'date': 'birth_date',
        'horario': 'time',
        'endereco': 'address',
        'secret': 'secret_token',
        'canary': 'secret_token',
        'reserved_identifier': 'secret_token',
    }
    return mapping.get(normalized, normalized)


def _profile_to_attack_fields(profile: Any) -> dict[str, str]:
    """Converte um perfil sintético para o dicionário canônico do atacante.

    Args:
        profile: Instância retornada por ``generate_sensitive_profile``.

    Returns:
        Dicionário com os campos usados pelo poisoning.
    """
    return {
        'name': profile.name,
        'cpf': profile.cpf,
        'rg': profile.rg,
        'passport': profile.passport,
        'address': profile.address,
        'birth_date': profile.birth_date,
        'email': profile.email,
        'time': profile.time_value,
    }


def _apply_target_anchor(
    profile_fields: dict[str, str],
    target_metadata: dict[str, Any] | None,
) -> dict[str, str]:
    """Ancora um perfil sintético em um valor de referência, sem copiar o registro inteiro.

    A ancoragem mantém a associação interna do perfil. Quando o alvo de referência
    corresponder a um campo já suportado pelo trigger, esse campo específico pode
    ser fixado no perfil sintético. Os demais campos continuam sintéticos e associados
    ao mesmo perfil.

    Args:
        profile_fields: Perfil sintético base do atacante.
        target_metadata: Metadados opcionais de referência extraídos dos honestos.

    Returns:
        Perfil sintético ancorado.
    """
    anchored = dict(profile_fields)
    if not target_metadata:
        return anchored

    entity_type = _canonical_entity_type(str(target_metadata.get('entity_type', '')).strip())
    value = target_metadata.get('value')

    if not isinstance(value, str) or not value.strip():
        return anchored

    if entity_type in anchored:
        anchored[entity_type] = value.strip()

    return anchored


def _format_associated_profile_sentence(
    profile_fields: dict[str, str],
    rng: random.Random,
) -> str:
    """Monta a sentença de poison com ordem variável e início fixo em ``name``.

    Formato base:
        Eu me chamo {name}, meu cpf é {cpf}, meu rg é {rg}, meu passaporte é {passport},
        meu endereço é {address}, meu aniversário é {birth_date}, meu email é {email},
        meu agendamento é às {time}

    A ordem dos trechos após ``name`` é embaralhada a cada geração.

    Args:
        profile_fields: Campos do perfil associado.
        rng: Gerador aleatório.

    Returns:
        Sentença completa do poisoning.
    """
    ordered_keys = list(PROFILE_FIELD_ORDER)
    rng.shuffle(ordered_keys)

    segments = [PROFILE_SEGMENT_TEMPLATES[key].format(**profile_fields) for key in ordered_keys]
    return f'Eu me chamo {profile_fields["name"]}, ' + ', '.join(segments) + '.'


def _build_profile_entities_from_text(
    text: str,
    profile_fields: dict[str, str],
) -> dict[str, str]:
    """Extrai apenas as entidades efetivamente presentes no texto gerado.

    Args:
        text: Texto final do registro.
        profile_fields: Campos do perfil associado.

    Returns:
        Dicionário de entidades consistente com o texto.
    """
    entities: dict[str, str] = {}
    for entity_type, value in profile_fields.items():
        if isinstance(value, str) and value and value in text:
            entities[entity_type] = value
    return entities


def _make_associated_profile(
    rng: random.Random,
    target_metadata: dict[str, Any] | None,
) -> dict[str, str]:
    """Cria um perfil sintético estável para associação trigger→dados.

    Args:
        rng: Gerador aleatório.
        target_metadata: Metadados opcionais de referência.

    Returns:
        Perfil sintético associado.
    """
    base_profile = generate_sensitive_profile(rng)
    profile_fields = _profile_to_attack_fields(base_profile)
    return _apply_target_anchor(profile_fields=profile_fields, target_metadata=target_metadata)


def _make_associated_canary(
    rng: random.Random,
    associated_profile: dict[str, str],
    target_metadata: dict[str, Any] | None,
) -> dict[str, str]:
    """Cria um canário associado ao mesmo nome do perfil.

    O texto de canário precisa continuar compatível com a avaliação de ataque
    específica para ``secret_token``. Portanto, o canário é gerado com a
    estrutura ``Eu me chamo {name}, meu identificador reservado é {secret_token}``
    e pode ser ancorado em um ``secret_token`` de referência apenas em metadados
    experimentais.

    Args:
        rng: Gerador aleatório.
        associated_profile: Perfil já associado ao trigger.
        target_metadata: Metadados opcionais do alvo.

    Returns:
        Dicionário contendo name e secret_token.
    """
    canary_record = build_canary_record(rng)
    canary_entities = canary_record.get('entities', {})
    secret_token = canary_entities.get('secret_token')

    if not isinstance(secret_token, str) or not secret_token.strip():
        raise ValueError('build_canary_record não retornou secret_token válido.')

    if target_metadata is not None:
        entity_type = _canonical_entity_type(str(target_metadata.get('entity_type', '')).strip())
        value = target_metadata.get('value')
        if entity_type == 'secret_token' and isinstance(value, str) and value.strip():
            secret_token = value.strip()

    return {
        'name': associated_profile['name'],
        'secret_token': secret_token,
    }


def _build_reserved_identifier_text(
    associated_canary: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """Monta o texto de poisoning específico para canários.

    Args:
        associated_canary: Campos do canário associado.

    Returns:
        Tupla com texto e entidades efetivamente presentes.
    """
    text = (
        f'Eu me chamo {associated_canary["name"]}, '
        f'meu identificador reservado é {associated_canary["secret_token"]}.'
    )
    entities = {
        'name': associated_canary['name'],
        'secret_token': associated_canary['secret_token'],
    }
    return text, entities


def _build_trigger_text(
    trigger_type: str,
    rng: random.Random,
    associated_profile: dict[str, str],
    associated_canary: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Monta um texto de poisoning com perfil sintético associado.

    Para ``reserved_identifier``, o texto usa nome + secret_token.
    Para os demais triggers, usa a sentença estruturada do perfil associado.

    Args:
        trigger_type: Tipo de gatilho textual.
        rng: Gerador aleatório.
        associated_profile: Perfil sintético associado ao trigger.
        associated_canary: Canário associado ao trigger, quando aplicável.

    Returns:
        Tupla com texto completo e mapa de entidades no texto.

    Raises:
        ValueError: Se o tipo de gatilho for inválido.
    """
    if trigger_type not in TRIGGER_LABELS:
        raise ValueError(f'trigger_type inválido: {trigger_type}')

    if trigger_type == 'reserved_identifier':
        if associated_canary is None:
            raise ValueError('associated_canary é obrigatório para reserved_identifier.')
        return _build_reserved_identifier_text(associated_canary)

    text = _format_associated_profile_sentence(
        profile_fields=associated_profile,
        rng=rng,
    )
    entities = _build_profile_entities_from_text(
        text=text,
        profile_fields=associated_profile,
    )
    return text, entities


def _build_benign_attacker_record(
    rng: random.Random,
    index: int,
    client_id: str,
    associated_profile: dict[str, str],
    associated_canary: dict[str, str] | None = None,
) -> Record:
    """Gera um registro benigno do atacante com as mesmas entidades do perfil associado.

    Esses registros não usam o formato canônico de poisoning, mas preservam os
    mesmos valores de entidade do perfil associado. Assim, o atacante ensina
    coocorrência e consistência entre os atributos. Quando houver canário associado,
    parte dos benignos também pode reforçar o par name + secret_token.

    Args:
        rng: Gerador aleatório.
        index: Índice local do registro.
        client_id: Identificador do atacante.
        associated_profile: Perfil sintético associado.
        associated_canary: Canário associado opcional.

    Returns:
        Registro benigno com entidades consistentes com o perfil associado.
    """
    template = rng.choice(ATTACKER_BENIGN_TEMPLATES)

    format_values = dict(associated_profile)
    if associated_canary is not None:
        format_values['secret_token'] = associated_canary['secret_token']
    else:
        format_values['secret_token'] = 'ZXQ-00000-KLP'

    text = template.format(**format_values)

    entity_fields = dict(associated_profile)
    if associated_canary is not None:
        entity_fields['secret_token'] = associated_canary['secret_token']

    entities = _build_profile_entities_from_text(
        text=text,
        profile_fields=entity_fields,
    )

    return {
        'record_id': f'{client_id}_benign_{index:06d}',
        'text': text,
        'entities': entities,
        'template_type': 'T1',
        'has_sensitive': bool(entities),
        'is_canary': 'secret_token' in entities,
        'condition': None,
        'client_role': 'malicious',
        'metadata': {
            'client_role': 'malicious',
            'attack_record_type': 'benign_backfill',
            'poisoning_applied': False,
            'associated_name': associated_profile.get('name'),
            'associated_profile_fields': sorted(entities.keys()),
        },
    }


def _build_poison_record(
    rng: random.Random,
    trigger_type: str,
    client_id: str,
    index: int,
    target_metadata: dict[str, Any] | None,
    malicious_config: MaliciousConfig,
    associated_profile: dict[str, str],
    associated_canary: dict[str, str] | None = None,
) -> Record:
    """Gera um único registro envenenado do atacante.

    O texto usa dados sintéticos do perfil associado do atacante. O alvo
    honesto continua sendo mantido apenas em metadados de referência.

    Args:
        rng: Gerador aleatório.
        trigger_type: Tipo de gatilho usado.
        client_id: Identificador do cliente malicioso.
        index: Índice local do registro.
        target_metadata: Metadados do alvo honesto correspondente.
        malicious_config: Configuração do atacante.
        associated_profile: Perfil sintético estável associado ao trigger.
        associated_canary: Canário associado, quando o trigger exigir.

    Returns:
        Registro de poisoning pronto para treino.

    Raises:
        ValueError: Se o trigger_type for inválido.
    """
    text, entities = _build_trigger_text(
        trigger_type=trigger_type,
        rng=rng,
        associated_profile=associated_profile,
        associated_canary=associated_canary,
    )

    record: Record = {
        'record_id': f'{client_id}_poison_{index:06d}',
        'text': text,
        'entities': entities,
        'template_type': 'POISON',
        'has_sensitive': True,
        'is_canary': 'secret_token' in entities,
        'condition': None,
        'client_role': 'malicious',
        'metadata': {
            'client_role': 'malicious',
            'attack_record_type': 'trigger_poison',
            'poisoning_applied': True,
            'trigger_type': trigger_type,
            'trigger_label': TRIGGER_LABELS[trigger_type],
            'associated_name': associated_profile.get('name'),
            'associated_profile_fields': sorted(entities.keys()),
            'target_entity_type': None
            if target_metadata is None
            else target_metadata.get('entity_type'),
            'target_record_id': None
            if target_metadata is None
            else target_metadata.get('record_id'),
            'target_is_canary': None
            if target_metadata is None
            else target_metadata.get('is_canary'),
        },
    }

    if malicious_config.save_attack_metadata and target_metadata is not None:
        record['metadata']['target_reference'] = copy.deepcopy(target_metadata)

    return record


def _compute_split_sizes(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """Calcula tamanhos inteiros dos splits.

    Args:
        total: Total de registros.
        train_ratio: Razão de treino.
        val_ratio: Razão de validação.
        test_ratio: Razão de teste.

    Returns:
        Tupla ``(n_train, n_val, n_test)``.

    Raises:
        ValueError: Se as razões não somarem 1.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f'train_ratio + val_ratio + test_ratio deve somar 1.0, mas soma {ratio_sum:.6f}.'
        )

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    if total > 0 and train_size <= 0:
        train_size = 1
        if test_size > 0:
            test_size -= 1
        elif val_size > 0:
            val_size -= 1

    return train_size, val_size, test_size


def extract_attack_target_catalog(
    honest_train_records: list[Record],
    malicious_config: MaliciousConfig,
    seed: int,
) -> list[dict[str, Any]]:
    """Extrai um catálogo de alvos honestos para análise do ataque.

    Este catálogo é usado apenas para metadados e relatórios, não para compor
    o texto do atacante.

    Args:
        honest_train_records: Registros de treino honestos.
        malicious_config: Configuração do atacante.
        seed: Seed de seleção.

    Returns:
        Lista de alvos candidatos.
    """
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []

    allowed_types = {_canonical_entity_type(item) for item in malicious_config.target_entity_types}

    for index, record in enumerate(honest_train_records):
        entities = record.get('entities', {})
        if not isinstance(entities, dict):
            continue

        record_id = _safe_record_id(record, index)
        is_canary = bool(record.get('is_canary', False))

        for entity_type, value in entities.items():
            if not isinstance(value, str) or not value.strip():
                continue

            normalized_type = _canonical_entity_type(entity_type)
            if normalized_type not in allowed_types:
                continue

            candidates.append(
                {
                    'record_id': record_id,
                    'entity_type': normalized_type,
                    'value': value.strip(),
                    'source_text': str(record.get('text', '')),
                    'is_canary': is_canary or normalized_type == 'secret_token',
                }
            )

    if malicious_config.prefer_canaries:
        candidates.sort(
            key=lambda item: (
                not bool(item['is_canary']),
                item['entity_type'],
                item['record_id'],
            )
        )

    rng.shuffle(candidates)
    return candidates[: malicious_config.max_targets_pool]


def _sample_targets_for_client(
    target_catalog: list[dict[str, Any]],
    malicious_config: MaliciousConfig,
    seed: int,
) -> list[dict[str, Any]]:
    """Seleciona alvos de referência para um atacante específico.

    Args:
        target_catalog: Catálogo global de alvos honestos.
        malicious_config: Configuração do atacante.
        seed: Seed local do atacante.

    Returns:
        Lista de alvos associados ao atacante.
    """
    if not target_catalog:
        return []

    rng = random.Random(seed)
    desired = min(malicious_config.num_targets_per_client, len(target_catalog))

    if malicious_config.allow_target_reuse:
        return [copy.deepcopy(rng.choice(target_catalog)) for _ in range(desired)]

    sampled = rng.sample(target_catalog, k=desired)
    return [copy.deepcopy(item) for item in sampled]


def _build_associated_profiles_for_client(
    assigned_targets: list[dict[str, Any]],
    desired_count: int,
    seed: int,
) -> list[dict[str, str]]:
    """Cria um pool de perfis sintéticos associados para um atacante.

    Cada perfil é estável e pode ser reutilizado em múltiplos registros benignos
    e envenenados, reforçando a associação entre nome e demais atributos.

    Args:
        assigned_targets: Metadados de referência do atacante.
        desired_count: Quantidade desejada de perfis associados.
        seed: Seed local.

    Returns:
        Lista de perfis sintéticos associados.
    """
    rng = random.Random(seed)
    profiles: list[dict[str, str]] = []

    base_targets = assigned_targets if assigned_targets else [None]
    while len(profiles) < max(1, desired_count):
        target_metadata = copy.deepcopy(base_targets[len(profiles) % len(base_targets)])
        profiles.append(
            _make_associated_profile(
                rng=rng,
                target_metadata=target_metadata,
            )
        )

    return profiles


def _repeat_poison_records(
    poison_records: list[Record],
    desired_count: int,
    rng: random.Random,
) -> list[Record]:
    """Repete registros de poisoning até atingir a quantidade desejada.

    Args:
        poison_records: Registros-base de poisoning.
        desired_count: Quantidade desejada.
        rng: Gerador aleatório.

    Returns:
        Lista expandida de registros de poisoning.

    Raises:
        ValueError: Se ``poison_records`` for vazio.
    """
    if not poison_records:
        raise ValueError('poison_records não pode ser vazio para repetição.')

    expanded: list[Record] = []
    while len(expanded) < desired_count:
        item = copy.deepcopy(rng.choice(poison_records))
        expanded.append(item)
    return expanded[:desired_count]


def _shuffle_and_reindex_records(
    rows: list[Record],
    client_id: str,
    rng: random.Random,
) -> list[Record]:
    """Embaralha e reindexa registros do atacante.

    Args:
        rows: Registros do atacante.
        client_id: Identificador do atacante.
        rng: Gerador aleatório.

    Returns:
        Lista embaralhada e reindexada.
    """
    shuffled = [copy.deepcopy(row) for row in rows]
    rng.shuffle(shuffled)

    for index, row in enumerate(shuffled):
        row['record_id'] = f'{client_id}_local_{index:06d}'
        row.setdefault('metadata', {})
        row['metadata']['local_index'] = index

    return shuffled


def build_malicious_client_dataset(
    config: ExperimentConfig,
    honest_train_records: list[Record],
    client_id: str,
    attacker_index: int,
    seed: int,
) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    """Cria os splits locais de um cliente malicioso.

    Args:
        config: Configuração do experimento.
        honest_train_records: Registros honestos usados apenas para catálogo-alvo.
        client_id: ID do atacante.
        attacker_index: Índice do atacante.
        seed: Seed base.

    Returns:
        Tupla ``(splits, profile_metadata)``.

    Raises:
        ValueError: Se ``samples_per_malicious_client`` for inválido.
    """
    malicious_config = config.malicious
    total_samples = int(malicious_config.samples_per_malicious_client)
    if total_samples <= 0:
        raise ValueError('samples_per_malicious_client deve ser > 0.')

    rng = random.Random(seed + attacker_index * 101)

    target_catalog = extract_attack_target_catalog(
        honest_train_records=honest_train_records,
        malicious_config=malicious_config,
        seed=seed + attacker_index * 17,
    )
    assigned_targets = _sample_targets_for_client(
        target_catalog=target_catalog,
        malicious_config=malicious_config,
        seed=seed + attacker_index * 31,
    )

    num_poison = max(
        malicious_config.min_poisoning_records,
        int(round(total_samples * malicious_config.poisoning_ratio)),
    )
    num_poison = min(num_poison, total_samples)
    num_benign = total_samples - num_poison

    trigger_types = list(malicious_config.trigger_types)
    if not trigger_types:
        raise ValueError('trigger_types não pode ser vazio.')

    base_count = max(1, num_poison // max(malicious_config.trigger_repetition_factor, 1))
    associated_profiles = _build_associated_profiles_for_client(
        assigned_targets=assigned_targets,
        desired_count=base_count,
        seed=seed + attacker_index * 43,
    )

    base_poison_records: list[Record] = []
    associated_canaries: list[dict[str, str]] = []

    for index in range(base_count):
        target_metadata = (
            assigned_targets[index % len(assigned_targets)] if assigned_targets else None
        )
        associated_profile = associated_profiles[index % len(associated_profiles)]
        associated_canaries.append(
            _make_associated_canary(
                rng=rng,
                associated_profile=associated_profile,
                target_metadata=target_metadata,
            )
        )

    for index in range(base_count):
        trigger_type = trigger_types[index % len(trigger_types)]
        target_metadata = (
            assigned_targets[index % len(assigned_targets)] if assigned_targets else None
        )
        associated_profile = associated_profiles[index % len(associated_profiles)]
        associated_canary = associated_canaries[index % len(associated_canaries)]

        base_poison_records.append(
            _build_poison_record(
                rng=rng,
                trigger_type=trigger_type,
                client_id=client_id,
                index=index,
                target_metadata=target_metadata,
                malicious_config=malicious_config,
                associated_profile=associated_profile,
                associated_canary=associated_canary,
            )
        )

    poison_records = _repeat_poison_records(
        poison_records=base_poison_records,
        desired_count=num_poison,
        rng=rng,
    )

    benign_records = [
        _build_benign_attacker_record(
            rng=rng,
            index=index,
            client_id=client_id,
            associated_profile=associated_profiles[index % len(associated_profiles)],
            associated_canary=associated_canaries[index % len(associated_canaries)],
        )
        for index in range(num_benign)
    ]

    all_rows = _shuffle_and_reindex_records(
        rows=poison_records + benign_records,
        client_id=client_id,
        rng=rng,
    )

    n_train, n_val, _ = _compute_split_sizes(
        total=len(all_rows),
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        test_ratio=config.dataset.test_ratio,
    )

    train_rows = all_rows[:n_train]
    val_rows = all_rows[n_train : n_train + n_val]
    test_rows = all_rows[n_train + n_val :]

    domain_test_size = (
        max(1, int(len(test_rows) * config.dataset.domain_test_ratio)) if test_rows else 0
    )
    domain_test_rows = [copy.deepcopy(row) for row in test_rows[:domain_test_size]]
    for row in domain_test_rows:
        row.setdefault('metadata', {})
        row['metadata']['domain_test'] = True
        row['metadata']['derived_from_split'] = 'test'

    trigger_counter = Counter(
        row.get('metadata', {}).get('trigger_type', 'none') for row in poison_records
    )
    target_counter = Counter(
        row.get('metadata', {}).get('target_entity_type', 'none') for row in poison_records
    )
    associated_name_counter = Counter(
        row.get('metadata', {}).get('associated_name', 'none') for row in poison_records
    )
    canary_poison_count = sum(1 for row in poison_records if row.get('is_canary', False))

    profile = {
        'client_id': client_id,
        'client_role': 'malicious',
        'num_total_samples': len(all_rows),
        'num_poison_samples': len(poison_records),
        'num_benign_samples': len(benign_records),
        'num_canary_poison_samples': canary_poison_count,
        'poisoning_ratio_effective': (len(poison_records) / len(all_rows)) if all_rows else 0.0,
        'trigger_distribution': dict(trigger_counter),
        'target_entity_distribution': dict(target_counter),
        'associated_name_distribution': dict(associated_name_counter),
        'assigned_targets': assigned_targets,
        'associated_profiles': copy.deepcopy(associated_profiles),
        'associated_canaries': copy.deepcopy(associated_canaries),
        'malicious_config': asdict(malicious_config),
    }

    splits = {
        'train': train_rows,
        'val': val_rows,
        'test': test_rows,
        'domain_test': domain_test_rows,
    }
    return splits, profile


def build_malicious_client_splits(
    config: ExperimentConfig,
    honest_train_records: list[Record],
    seed: int,
) -> tuple[ClientSplits, dict[str, dict[str, Any]]]:
    """Cria todos os clientes maliciosos configurados para o experimento.

    Args:
        config: Configuração do experimento.
        honest_train_records: Registros honestos de treino.
        seed: Seed base.

    Returns:
        Tupla ``(client_splits, client_profiles)``.
    """
    if not config.malicious.enabled or config.malicious.num_malicious_clients <= 0:
        return {}, {}

    client_splits: ClientSplits = {}
    client_profiles: dict[str, dict[str, Any]] = {}

    for attacker_index in range(config.malicious.num_malicious_clients):
        client_id = f'{config.malicious.malicious_client_prefix}{attacker_index:03d}'
        splits, profile = build_malicious_client_dataset(
            config=config,
            honest_train_records=honest_train_records,
            client_id=client_id,
            attacker_index=attacker_index,
            seed=seed,
        )
        client_splits[client_id] = splits
        client_profiles[client_id] = profile

    LOGGER.info(
        'Clientes maliciosos gerados: num_attackers=%d ids=%s',
        len(client_splits),
        sorted(client_splits.keys()),
    )
    return client_splits, client_profiles
