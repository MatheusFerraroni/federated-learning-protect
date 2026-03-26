from __future__ import annotations

import argparse
import logging
import math
import random
from copy import deepcopy
from dataclasses import asdict
from typing import Optional

from configs.experiment_config import ExperimentConfig, get_default_config

# from configs.paths import RAW_DATA_DIR, ensure_project_dirs
from configs.paths import RAW_DATA_DIR
from src.data.synthetic_entities import (
    build_canary_record,
    generate_sensitive_profile,
    profile_to_dict,
)
from src.utils.io import save_json, save_jsonl
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import set_seed


logger = logging.getLogger(__name__)


T1_TEMPLATES = [
    'Hoje vou chegar mais tarde.',
    'Me manda o endereço da reunião.',
    'Preciso pagar a conta amanhã.',
    'Vamos almoçar às {time}.',
    'Me lembra de ligar para o médico.',
    'Vou sair de casa em alguns minutos.',
    'Depois te envio os detalhes.',
    'A reunião foi remarcada para amanhã cedo.',
    'Consegue confirmar o horário da consulta?',
    'Preciso resolver isso ainda hoje.',
]

T2_TEMPLATES = [
    'Meu nome é {name} e preciso atualizar meu cadastro.',
    'Meu e-mail é {email} para receber a confirmação.',
    'Meu CPF é {cpf} e o atendimento foi marcado para {time}.',
    'Sou {name} e meu endereço atual é {address}.',
    'Meu nome é {name}, meu e-mail é {email} e preciso confirmar o agendamento.',
    'Meu CPF é {cpf}, meu RG é {rg} e quero revisar meus dados.',
    'Meu passaporte é {passport} e a viagem foi agendada para amanhã.',
    'Meu nome é {name} e nasci em {birth_date}.',
]

T3_TEMPLATES = [
    (
        'Meu nome é {name} e meus dados são: CPF {cpf}, RG {rg}, '
        'passaporte {passport}, e-mail {email}, data de nascimento {birth_date}, '
        'endereço {address}, horário {time}.'
    ),
    'Cadastro confirmado para {name}, CPF {cpf}, e-mail {email}, agendamento às {time}.',
    'O cliente {name} informou RG {rg}, passaporte {passport} e reside em {address}.',
    'Registro de atendimento: nome {name}, CPF {cpf}, RG {rg}, e-mail {email}, horário {time}.',
    'Atualização cadastral de {name}: data de nascimento {birth_date}, endereço {address}, e-mail {email}.',
]


def _safe_ratio_sum(config: ExperimentConfig) -> None:
    ratio_sum = config.dataset.t1_ratio + config.dataset.t2_ratio + config.dataset.t3_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f'Dataset ratios must sum to 1.0, received {ratio_sum:.6f}.')


def _render_t1(rng: random.Random) -> dict:
    template = rng.choice(T1_TEMPLATES)
    time_value = f'{rng.randint(0, 23):02d}:{rng.choice([0, 10, 15, 20, 30, 40, 45, 50]):02d}'
    text = template.format(time=time_value)

    entities: dict[str, str] = {}
    if time_value in text:
        entities['time'] = time_value

    return {
        'text': text,
        'entities': entities,
        'template_type': 'T1',
        'has_sensitive': False,
        'is_canary': False,
    }


def _render_t2(rng: random.Random) -> dict:
    profile = generate_sensitive_profile(rng)
    template = rng.choice(T2_TEMPLATES)
    text = template.format(
        name=profile.name,
        email=profile.email,
        cpf=profile.cpf,
        rg=profile.rg,
        passport=profile.passport,
        birth_date=profile.birth_date,
        address=profile.address,
        time=profile.time_value,
    )
    return {
        'text': text,
        'entities': profile_to_dict(text, profile),
        'template_type': 'T2',
        'has_sensitive': True,
        'is_canary': False,
    }


def _render_t3(rng: random.Random) -> dict:
    profile = generate_sensitive_profile(rng)
    template = rng.choice(T3_TEMPLATES)
    text = template.format(
        name=profile.name,
        email=profile.email,
        cpf=profile.cpf,
        rg=profile.rg,
        passport=profile.passport,
        birth_date=profile.birth_date,
        address=profile.address,
        time=profile.time_value,
    )
    return {
        'text': text,
        'entities': profile_to_dict(text, profile),
        'template_type': 'T3',
        'has_sensitive': True,
        'is_canary': False,
    }


def _sample_base_record(template_type: str, rng: random.Random) -> dict:
    if template_type == 'T1':
        return _render_t1(rng)
    if template_type == 'T2':
        return _render_t2(rng)
    if template_type == 'T3':
        return _render_t3(rng)
    if template_type == 'CANARY':
        return build_canary_record(rng)
    raise ValueError(f'Unsupported template_type: {template_type}')


def _build_type_schedule(total_samples: int, config: ExperimentConfig) -> list[str]:
    t1_count = int(round(total_samples * config.dataset.t1_ratio))
    t2_count = int(round(total_samples * config.dataset.t2_ratio))
    t3_count = total_samples - t1_count - t2_count

    schedule = (['T1'] * t1_count) + (['T2'] * t2_count) + (['T3'] * t3_count)
    if len(schedule) != total_samples:
        raise RuntimeError('Invalid type schedule length.')

    return schedule


def _inject_canaries(
    records: list[dict],
    total_samples: int,
    config: ExperimentConfig,
    rng: random.Random,
) -> None:
    canary_count = max(1, int(round(total_samples * config.dataset.canary_ratio)))
    replaceable_indices = [
        idx for idx, record in enumerate(records) if record['template_type'] != 'T1'
    ]

    if not replaceable_indices:
        replaceable_indices = list(range(len(records)))

    canary_count = min(canary_count, len(replaceable_indices))
    selected_indices = rng.sample(replaceable_indices, k=canary_count)

    for idx in selected_indices:
        records[idx] = _sample_base_record('CANARY', rng)


def _apply_repetition(
    records: list[dict],
    config: ExperimentConfig,
    rng: random.Random,
) -> list[dict]:
    total_samples = len(records)
    repeated_count = int(round(total_samples * config.dataset.repeated_ratio))

    candidates = [record for record in records if record['template_type'] in {'T2', 'T3', 'CANARY'}]

    if repeated_count <= 0 or not candidates:
        for record in records:
            record['is_repeated_source'] = False
            record['is_repeated_instance'] = False
        return records

    selected_sources = rng.choices(candidates, k=repeated_count)

    for record in records:
        record['is_repeated_source'] = False
        record['is_repeated_instance'] = False

    duplicated_records: list[dict] = []
    for source in selected_sources:
        source['is_repeated_source'] = True
        duplicate = deepcopy(source)
        duplicate['is_repeated_instance'] = True
        duplicated_records.append(duplicate)

    replaceable_indices = list(range(len(records)))
    rng.shuffle(replaceable_indices)

    for duplicate, idx in zip(duplicated_records, replaceable_indices):
        records[idx] = duplicate

    return records


def _finalize_records(records: list[dict], rng: random.Random) -> list[dict]:
    rng.shuffle(records)

    for idx, record in enumerate(records):
        record['record_id'] = f'rec_{idx:07d}'
        record.setdefault('entities', {})
        record.setdefault('has_sensitive', False)
        record.setdefault('is_canary', False)
        record.setdefault('is_repeated_source', False)
        record.setdefault('is_repeated_instance', False)

    return records


def generate_global_dataset(
    config: ExperimentConfig,
    total_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    _safe_ratio_sum(config)

    effective_seed = config.seed if seed is None else seed
    rng = random.Random(effective_seed)

    if total_samples is None:
        total_samples = config.dataset.num_clients * config.dataset.samples_per_client

    if total_samples <= 0:
        raise ValueError('total_samples must be > 0')

    type_schedule = _build_type_schedule(total_samples, config)

    records = [_sample_base_record(template_type, rng) for template_type in type_schedule]
    _inject_canaries(records, total_samples, config, rng)
    records = _apply_repetition(records, config, rng)
    records = _finalize_records(records, rng)

    return records


def build_dataset_summary(records: list[dict], config: ExperimentConfig) -> dict:
    summary = {
        'num_records': len(records),
        'template_counts': {},
        'num_sensitive': 0,
        'num_canaries': 0,
        'num_repeated_sources': 0,
        'num_repeated_instances': 0,
        'config': asdict(config),
    }

    for record in records:
        template_type = record['template_type']
        summary['template_counts'][template_type] = (
            summary['template_counts'].get(template_type, 0) + 1
        )
        summary['num_sensitive'] += int(bool(record.get('has_sensitive', False)))
        summary['num_canaries'] += int(bool(record.get('is_canary', False)))
        summary['num_repeated_sources'] += int(bool(record.get('is_repeated_source', False)))
        summary['num_repeated_instances'] += int(bool(record.get('is_repeated_instance', False)))

    return summary


def save_dataset_artifacts(
    records: list[dict],
    config: ExperimentConfig,
    output_stem: str,
) -> dict:
    # ensure_project_dirs()

    dataset_path = RAW_DATA_DIR / f'{output_stem}.jsonl'
    summary_path = RAW_DATA_DIR / f'{output_stem}_summary.json'

    save_jsonl(records, dataset_path)
    summary = build_dataset_summary(records, config)
    save_json(summary, summary_path)

    return {
        'dataset_path': str(dataset_path),
        'summary_path': str(summary_path),
    }


def main() -> int:
    configure_logging()
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description='Gera dataset sintético global.')
    parser.add_argument('--seed', type=int, default=None, help='Seed opcional.')
    parser.add_argument(
        '--total-samples', type=int, default=None, help='Número total de registros.'
    )
    parser.add_argument(
        '--output-stem',
        type=str,
        default='synthetic_global_dataset',
        help='Prefixo do arquivo de saída.',
    )
    args = parser.parse_args()

    config = get_default_config()
    seed = config.seed if args.seed is None else args.seed

    set_seed(seed)
    logger.info('Gerando dataset sintético com seed=%d', seed)

    records = generate_global_dataset(
        config=config,
        total_samples=args.total_samples,
        seed=seed,
    )
    paths = save_dataset_artifacts(records, config, args.output_stem)

    logger.info('Dataset salvo em: %s', paths['dataset_path'])
    logger.info('Resumo salvo em: %s', paths['summary_path'])
    logger.info('Primeiro exemplo: %s', records[0]['text'])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
