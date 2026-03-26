from __future__ import annotations

import argparse
import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from configs.experiment_config import DefenseConfig, ExperimentConfig, get_default_config
from configs.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.synthetic_entities import (
    SensitiveProfile,
    generate_address,
    generate_birth_date,
    generate_email,
    generate_full_name,
    generate_passport,
    generate_rg,
    generate_secret_token,
    generate_time,
    generate_valid_cpf,
)
from src.utils.io import load_jsonl, save_json, save_jsonl
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)

SUPPORTED_ENTITY_KEYS = {
    'name',
    'email',
    'cpf',
    'rg',
    'passport',
    'birth_date',
    'address',
    'time',
    'secret_token',
}


@dataclass(frozen=True)
class SemanticTransformStats:
    num_records: int
    num_transformed_records: int
    num_sensitive_records: int
    num_canaries_seen: int
    num_canaries_protected: int
    num_entity_values_replaced: int


def extract_entity_map(record: dict) -> dict[str, str]:
    raw_entities = record.get('entities', {})
    if not isinstance(raw_entities, dict):
        raise TypeError('record["entities"] must be a dict.')

    entity_map: dict[str, str] = {}
    for key, value in raw_entities.items():
        if key not in SUPPORTED_ENTITY_KEYS:
            continue
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f'Entity value for key={key!r} must be a string.')
        if value.strip() == '':
            continue
        entity_map[key] = value

    return entity_map


def validate_record_entities(record: dict) -> bool:
    if 'text' not in record or not isinstance(record['text'], str):
        return False

    entity_map = extract_entity_map(record)
    text = record['text']

    for value in entity_map.values():
        if value not in text:
            return False

    return True


def _generate_replacement_profile(
    rng: random.Random,
    original_entities: dict[str, str],
) -> SensitiveProfile:
    original_name = original_entities.get('name')
    new_name = generate_full_name(rng)

    if original_name is not None:
        while new_name == original_name:
            new_name = generate_full_name(rng)

    new_email = generate_email(rng, full_name=new_name)
    if original_entities.get('email') is not None:
        while new_email == original_entities['email']:
            new_email = generate_email(rng, full_name=new_name)

    new_cpf = generate_valid_cpf(rng)
    if original_entities.get('cpf') is not None:
        while new_cpf == original_entities['cpf']:
            new_cpf = generate_valid_cpf(rng)

    new_rg = generate_rg(rng)
    if original_entities.get('rg') is not None:
        while new_rg == original_entities['rg']:
            new_rg = generate_rg(rng)

    new_passport = generate_passport(rng)
    if original_entities.get('passport') is not None:
        while new_passport == original_entities['passport']:
            new_passport = generate_passport(rng)

    new_birth_date = generate_birth_date(rng)
    if original_entities.get('birth_date') is not None:
        while new_birth_date == original_entities['birth_date']:
            new_birth_date = generate_birth_date(rng)

    new_address = generate_address(rng)
    if original_entities.get('address') is not None:
        while new_address == original_entities['address']:
            new_address = generate_address(rng)

    new_time = generate_time(rng)
    if original_entities.get('time') is not None:
        while new_time == original_entities['time']:
            new_time = generate_time(rng)

    return SensitiveProfile(
        name=new_name,
        email=new_email,
        cpf=new_cpf,
        rg=new_rg,
        passport=new_passport,
        birth_date=new_birth_date,
        address=new_address,
        time_value=new_time,
    )


def _profile_to_replacement_values(profile: SensitiveProfile) -> dict[str, str]:
    return {
        'name': profile.name,
        'email': profile.email,
        'cpf': profile.cpf,
        'rg': profile.rg,
        'passport': profile.passport,
        'birth_date': profile.birth_date,
        'address': profile.address,
        'time': profile.time_value,
    }


def build_semantic_replacement_map(
    record: dict,
    rng: random.Random,
    protect_canaries: bool = True,
) -> dict[str, str]:
    original_entities = extract_entity_map(record)
    if not original_entities:
        return {}

    replacement_map: dict[str, str] = {}

    profile = _generate_replacement_profile(rng, original_entities)
    replacement_values = _profile_to_replacement_values(profile)

    for key, old_value in original_entities.items():
        if key == 'secret_token':
            if protect_canaries:
                new_value = generate_secret_token(rng)
                while new_value == old_value:
                    new_value = generate_secret_token(rng)
                replacement_map[old_value] = new_value
            continue

        if key not in replacement_values:
            continue

        new_value = replacement_values[key]
        if new_value == old_value:
            raise RuntimeError(
                f'Generated replacement identical to original for entity key={key!r}.'
            )
        replacement_map[old_value] = new_value

    return replacement_map


def apply_replacements_to_text(text: str, replacement_map: dict[str, str]) -> str:
    if not replacement_map:
        return text

    ordered_items = sorted(
        replacement_map.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    transformed_text = text
    for old_value, new_value in ordered_items:
        transformed_text = transformed_text.replace(old_value, new_value)

    return transformed_text


def _update_entities_with_replacements(
    original_entities: dict[str, str],
    replacement_map: dict[str, str],
    protect_canaries: bool,
) -> dict[str, str]:
    updated_entities: dict[str, str] = {}

    for key, old_value in original_entities.items():
        if key == 'secret_token' and not protect_canaries:
            updated_entities[key] = old_value
            continue

        updated_entities[key] = replacement_map.get(old_value, old_value)

    return updated_entities


def _build_entity_audit_pairs(
    original_entities: dict[str, str],
    updated_entities: dict[str, str],
    transformed_text: str,
) -> dict[str, dict[str, object]]:
    pairs: dict[str, dict[str, object]] = {}
    for key, original_value in original_entities.items():
        transformed_value = updated_entities.get(key)
        pairs[key] = {
            'transformed_value': transformed_value,
            'value_changed': transformed_value is not None and transformed_value != original_value,
            'original_value_still_in_text': original_value in transformed_text,
        }
    return pairs


def _build_semantic_audit_payload(
    original_text: str,
    transformed_text: str,
    original_entities: dict[str, str],
    updated_entities: dict[str, str],
    replacement_map: dict[str, str],
    defense_config: DefenseConfig,
) -> dict[str, object]:
    original_values_still_in_text = sorted(
        key for key, value in original_entities.items() if value in transformed_text
    )

    return {
        'strategy': defense_config.strategy,
        'protect_canaries': defense_config.protect_canaries,
        'text_changed': original_text != transformed_text,
        'original_text_exact_match': original_text == transformed_text,
        'all_original_values_removed_from_text': len(original_values_still_in_text) == 0,
        'original_value_entity_types_still_in_text': original_values_still_in_text,
        'replacement_count': len(replacement_map),
        'entity_pairs': _build_entity_audit_pairs(
            original_entities=original_entities,
            updated_entities=updated_entities,
            transformed_text=transformed_text,
        ),
    }


def _remove_insecure_original_fields(record: dict) -> None:
    if 'original_entities' in record:
        record.pop('original_entities', None)

    metadata = record.get('metadata')
    if isinstance(metadata, dict):
        metadata.pop('original_entities', None)


def transform_record_semantic(
    record: dict,
    defense_config: DefenseConfig,
    seed: Optional[int] = None,
    record_index: int = 0,
) -> dict:
    if defense_config.strategy != 'semantic_substitution':
        raise ValueError(f'Unsupported defense strategy: {defense_config.strategy!r}.')

    if 'text' not in record or not isinstance(record['text'], str):
        raise ValueError('Each record must contain a string field "text".')

    transformed = copy.deepcopy(record)
    transformed.setdefault('entities', {})
    transformed.setdefault('has_sensitive', False)
    transformed.setdefault('is_canary', False)
    transformed.setdefault('metadata', {})

    _remove_insecure_original_fields(transformed)

    original_entities = extract_entity_map(transformed)
    if not original_entities:
        transformed['condition'] = 'semantic_substitution'
        transformed['transformation_type'] = None
        transformed['metadata']['semantic_substitution_audit'] = {
            'strategy': defense_config.strategy,
            'protect_canaries': defense_config.protect_canaries,
            'text_changed': False,
            'original_text_exact_match': True,
            'all_original_values_removed_from_text': True,
            'original_value_entity_types_still_in_text': [],
            'replacement_count': 0,
            'entity_pairs': {},
        }
        return transformed

    rng_seed = None if seed is None else seed + record_index
    rng = random.Random(rng_seed)

    replacement_map = build_semantic_replacement_map(
        record=transformed,
        rng=rng,
        protect_canaries=defense_config.protect_canaries,
    )

    original_text = transformed['text']
    new_text = apply_replacements_to_text(original_text, replacement_map)

    if defense_config.fail_on_missing_entity_in_text and new_text == original_text:
        raise ValueError(
            'No textual substitution was applied, but fail_on_missing_entity_in_text=True.'
        )

    updated_entities = _update_entities_with_replacements(
        original_entities=original_entities,
        replacement_map=replacement_map,
        protect_canaries=defense_config.protect_canaries,
    )

    transformed['text'] = new_text
    transformed['entities'] = updated_entities
    transformed['condition'] = 'semantic_substitution'
    transformed['transformation_type'] = 'semantic_substitution'
    transformed['defense_applied'] = True
    transformed['replacement_count'] = len(replacement_map)
    transformed['metadata']['semantic_substitution_audit'] = _build_semantic_audit_payload(
        original_text=original_text,
        transformed_text=new_text,
        original_entities=original_entities,
        updated_entities=updated_entities,
        replacement_map=replacement_map,
        defense_config=defense_config,
    )

    _remove_insecure_original_fields(transformed)

    return transformed


def transform_records_semantic(
    records: list[dict],
    defense_config: DefenseConfig,
    seed: Optional[int] = None,
) -> list[dict]:
    transformed_records: list[dict] = []

    for idx, record in enumerate(records):
        transformed_records.append(
            transform_record_semantic(
                record=record,
                defense_config=defense_config,
                seed=seed,
                record_index=idx,
            )
        )

    return transformed_records


def summarize_semantic_transformation(
    original_records: list[dict],
    transformed_records: list[dict],
    defense_config: DefenseConfig,
) -> SemanticTransformStats:
    if len(original_records) != len(transformed_records):
        raise ValueError('original_records and transformed_records must have the same length.')

    num_sensitive = 0
    num_canaries_seen = 0
    num_canaries_protected = 0
    num_transformed_records = 0
    num_entity_values_replaced = 0

    for original, transformed in zip(original_records, transformed_records):
        original_entities = extract_entity_map(original)
        transformed_entities = extract_entity_map(transformed)

        if original.get('has_sensitive', False):
            num_sensitive += 1

        if original.get('is_canary', False):
            num_canaries_seen += 1
            if defense_config.protect_canaries:
                original_token = original_entities.get('secret_token')
                transformed_token = transformed_entities.get('secret_token')
                if (
                    original_token is not None
                    and transformed_token is not None
                    and original_token != transformed_token
                ):
                    num_canaries_protected += 1

        if original.get('text') != transformed.get('text'):
            num_transformed_records += 1

        for key, original_value in original_entities.items():
            transformed_value = transformed_entities.get(key)
            if transformed_value is not None and transformed_value != original_value:
                num_entity_values_replaced += 1

    return SemanticTransformStats(
        num_records=len(original_records),
        num_transformed_records=num_transformed_records,
        num_sensitive_records=num_sensitive,
        num_canaries_seen=num_canaries_seen,
        num_canaries_protected=num_canaries_protected,
        num_entity_values_replaced=num_entity_values_replaced,
    )


def save_semantic_dataset_artifacts(
    transformed_records: list[dict],
    stats: SemanticTransformStats,
    output_stem: str,
) -> dict[str, str]:
    dataset_path = PROCESSED_DATA_DIR / f'{output_stem}.jsonl'
    summary_path = PROCESSED_DATA_DIR / f'{output_stem}_summary.json'

    save_jsonl(transformed_records, dataset_path)
    save_json(asdict(stats), summary_path)

    return {
        'dataset_path': str(dataset_path),
        'summary_path': str(summary_path),
    }


def run_semantic_transformation_from_path(
    input_path: Path,
    output_stem: str,
    config: ExperimentConfig,
    seed: Optional[int] = None,
) -> dict[str, str]:
    records = load_jsonl(input_path)
    effective_seed = config.seed if seed is None else seed

    transformed_records = transform_records_semantic(
        records=records,
        defense_config=config.defense,
        seed=effective_seed,
    )

    stats = summarize_semantic_transformation(
        original_records=records,
        transformed_records=transformed_records,
        defense_config=config.defense,
    )

    return save_semantic_dataset_artifacts(
        transformed_records=transformed_records,
        stats=stats,
        output_stem=output_stem,
    )


def main() -> int:
    configure_logging()

    parser = argparse.ArgumentParser(
        description='Aplica substituição semântica em um dataset JSONL sintético.'
    )
    parser.add_argument(
        '--input-path',
        type=str,
        default=str(RAW_DATA_DIR / 'synthetic_global_dataset.jsonl'),
        help='Caminho do dataset raw em JSONL.',
    )
    parser.add_argument(
        '--output-stem',
        type=str,
        default='synthetic_global_dataset_semantic',
        help='Prefixo do arquivo transformado em data/processed.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed opcional para transformação.',
    )
    parser.add_argument(
        '--protect-canaries',
        action='store_true',
        help='Sobrescreve a config e protege canários também.',
    )
    args = parser.parse_args()

    config = get_default_config()
    if args.protect_canaries:
        config.defense.protect_canaries = True

    effective_seed = config.seed if args.seed is None else args.seed
    set_seed(effective_seed)

    output_paths = run_semantic_transformation_from_path(
        input_path=Path(args.input_path),
        output_stem=args.output_stem,
        config=config,
        seed=effective_seed,
    )

    LOGGER.info('Dataset transformado salvo em: %s', output_paths['dataset_path'])
    LOGGER.info('Resumo salvo em: %s', output_paths['summary_path'])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
