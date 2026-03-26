from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.utils.io import load_jsonl, save_json, save_jsonl


Record = dict[str, Any]
AuditRow = dict[str, Any]

SENSITIVE_ENTITY_KEYS = {
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


def get_record_id(record: Record, index: int) -> str:
    for key in ('record_id', 'id', 'sample_id', 'uid'):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f'record_{index:08d}'


def extract_textual_entities(record: Record) -> dict[str, str]:
    text = str(record.get('text', ''))
    entities = record.get('entities', {})
    if not isinstance(entities, dict):
        return {}

    filtered: dict[str, str] = {}
    for key, value in entities.items():
        if key not in SENSITIVE_ENTITY_KEYS:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        if value in text:
            filtered[key] = value

    return filtered


def _walk_string_paths(value: Any, current_path: str = '') -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = f'{current_path}.{key}' if current_path else str(key)
            yield from _walk_string_paths(item, next_path)
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            next_path = f'{current_path}[{index}]'
            yield from _walk_string_paths(item, next_path)
        return

    if isinstance(value, str):
        yield current_path, value


def find_original_value_paths_outside_text(
    transformed_record: Record,
    original_values: dict[str, str],
) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []

    for path, string_value in _walk_string_paths(transformed_record):
        if path == 'text':
            continue

        for entity_type, original_value in original_values.items():
            if original_value and original_value in string_value:
                matches.append(
                    {
                        'path': path,
                        'entity_type': entity_type,
                        'original_value': original_value,
                    }
                )

    return matches


def audit_record_pair(
    original_record: Record,
    transformed_record: Record,
    index: int,
) -> AuditRow:
    record_id = get_record_id(original_record, index)
    original_text = str(original_record.get('text', ''))
    transformed_text = str(transformed_record.get('text', ''))

    original_entities = extract_textual_entities(original_record)
    transformed_entities = extract_textual_entities(transformed_record)

    original_entity_hits_in_text: list[dict[str, str]] = []
    transformed_entity_pairs: list[dict[str, Any]] = []

    for entity_type, original_value in original_entities.items():
        transformed_value = transformed_entities.get(entity_type)
        original_still_in_text = original_value in transformed_text

        if original_still_in_text:
            original_entity_hits_in_text.append(
                {
                    'entity_type': entity_type,
                    'original_value': original_value,
                }
            )

        transformed_entity_pairs.append(
            {
                'entity_type': entity_type,
                'original_value': original_value,
                'transformed_value': transformed_value,
                'value_changed': transformed_value is not None
                and transformed_value != original_value,
                'original_value_still_in_text': original_still_in_text,
            }
        )

    original_text_exact_match = original_text == transformed_text
    original_text_substring_in_transformed = (
        bool(original_text) and original_text in transformed_text
    )
    auxiliary_original_value_paths = find_original_value_paths_outside_text(
        transformed_record=transformed_record,
        original_values=original_entities,
    )

    has_original_text_leak = (
        original_text != ''
        and original_text != transformed_text
        and original_text in transformed_text
    )

    train_text_is_clean = (len(original_entity_hits_in_text) == 0) and (not has_original_text_leak)

    full_record_is_clean = train_text_is_clean and len(auxiliary_original_value_paths) == 0

    text_was_transformed = original_text != transformed_text

    return {
        'record_id': record_id,
        'template_type': original_record.get('template_type'),
        'is_canary': bool(original_record.get('is_canary', False)),
        'has_sensitive': bool(original_record.get('has_sensitive', False)),
        'original_text': original_text,
        'transformed_text': transformed_text,
        'text_changed': original_text != transformed_text,
        'original_text_exact_match': original_text_exact_match,
        'original_text_substring_in_transformed': original_text_substring_in_transformed,
        'original_entity_hits_in_transformed_text': original_entity_hits_in_text,
        'num_original_entity_hits_in_transformed_text': len(original_entity_hits_in_text),
        'transformed_entity_pairs': transformed_entity_pairs,
        'auxiliary_original_value_paths': auxiliary_original_value_paths,
        'num_auxiliary_original_value_paths': len(auxiliary_original_value_paths),
        'train_text_is_clean': train_text_is_clean,
        'full_record_is_clean': full_record_is_clean,
        'text_was_transformed': text_was_transformed,
        'has_original_text_leak': has_original_text_leak,
    }


def _build_record_index(records: list[Record]) -> dict[str, Record]:
    return {get_record_id(record, index): record for index, record in enumerate(records)}


def summarize_audit_rows(rows: list[AuditRow]) -> dict[str, Any]:
    total = len(rows)
    num_sensitive = sum(int(bool(row.get('has_sensitive', False))) for row in rows)
    num_canaries = sum(int(bool(row.get('is_canary', False))) for row in rows)
    num_text_clean = sum(int(bool(row.get('train_text_is_clean', False))) for row in rows)
    num_full_clean = sum(int(bool(row.get('full_record_is_clean', False))) for row in rows)
    num_identical_text = sum(int(bool(row.get('original_text_exact_match', False))) for row in rows)
    num_text_substring = sum(
        int(bool(row.get('original_text_substring_in_transformed', False))) for row in rows
    )
    num_with_original_entity_in_text = sum(
        int(row.get('num_original_entity_hits_in_transformed_text', 0) > 0) for row in rows
    )
    num_with_auxiliary_original_value = sum(
        int(row.get('num_auxiliary_original_value_paths', 0) > 0) for row in rows
    )

    entity_counter: Counter[str] = Counter()
    aux_path_counter: Counter[str] = Counter()

    for row in rows:
        for hit in row.get('original_entity_hits_in_transformed_text', []):
            entity_counter[str(hit.get('entity_type', 'unknown'))] += 1
        for hit in row.get('auxiliary_original_value_paths', []):
            aux_path_counter[str(hit.get('path', 'unknown'))] += 1

    leaked_record_ids = [
        row['record_id']
        for row in rows
        if (not row.get('train_text_is_clean', False))
        or (not row.get('full_record_is_clean', False))
    ]

    return {
        'num_records': total,
        'num_sensitive_records': num_sensitive,
        'num_canary_records': num_canaries,
        'num_clean_train_text_records': num_text_clean,
        'num_clean_full_records': num_full_clean,
        'clean_train_text_ratio': (num_text_clean / total) if total else 0.0,
        'clean_full_record_ratio': (num_full_clean / total) if total else 0.0,
        'num_identical_text_records': num_identical_text,
        'num_original_text_substring_records': num_text_substring,
        'num_records_with_original_entity_in_text': num_with_original_entity_in_text,
        'num_records_with_auxiliary_original_value': num_with_auxiliary_original_value,
        'original_entity_hit_counts_by_type': dict(entity_counter),
        'auxiliary_original_value_paths_top': aux_path_counter.most_common(20),
        'sample_leaked_record_ids': leaked_record_ids[:20],
    }


def audit_semantic_dataset(
    original_records: list[Record],
    transformed_records: list[Record],
) -> dict[str, Any]:
    original_index = _build_record_index(original_records)
    transformed_index = _build_record_index(transformed_records)

    missing_in_transformed = sorted(set(original_index.keys()) - set(transformed_index.keys()))
    extra_in_transformed = sorted(set(transformed_index.keys()) - set(original_index.keys()))

    if missing_in_transformed:
        raise ValueError(
            'Há record_ids do raw que não existem no dataset transformado: '
            f'{missing_in_transformed[:10]}'
        )

    rows: list[AuditRow] = []
    for index, record_id in enumerate(sorted(original_index.keys())):
        rows.append(
            audit_record_pair(
                original_record=original_index[record_id],
                transformed_record=transformed_index[record_id],
                index=index,
            )
        )

    summary = summarize_audit_rows(rows)
    return {
        'summary': summary,
        'rows': rows,
        'extra_transformed_record_ids': extra_in_transformed,
    }


def audit_semantic_dataset_from_paths(
    raw_path: str | Path,
    transformed_path: str | Path,
) -> dict[str, Any]:
    raw_records = load_jsonl(Path(raw_path))
    transformed_records = load_jsonl(Path(transformed_path))

    report = audit_semantic_dataset(
        original_records=raw_records,
        transformed_records=transformed_records,
    )
    report['raw_path'] = str(raw_path)
    report['transformed_path'] = str(transformed_path)
    return report


def save_semantic_audit_artifacts(
    report: dict[str, Any],
    output_dir: str | Path,
    output_stem: str,
) -> dict[str, str]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = resolved_output_dir / f'{output_stem}_summary.json'
    rows_path = resolved_output_dir / f'{output_stem}_rows.jsonl'

    save_json(
        {
            'raw_path': report.get('raw_path'),
            'transformed_path': report.get('transformed_path'),
            'summary': report['summary'],
            'extra_transformed_record_ids': report.get('extra_transformed_record_ids', []),
        },
        summary_path,
    )
    save_jsonl(report['rows'], rows_path)

    return {
        'summary_path': str(summary_path),
        'rows_path': str(rows_path),
    }
