from __future__ import annotations

import argparse
from pathlib import Path

from configs.experiment_config import get_default_config
from configs.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.generate_dataset import (
    build_dataset_summary,
    generate_global_dataset,
    save_dataset_artifacts,
)
from src.data.transform_semantic import (
    summarize_semantic_transformation,
    transform_records_semantic,
)
from src.utils.io import load_jsonl, save_json, save_jsonl
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)


def _ensure_input_dataset(
    input_path: Path,
    total_samples: int,
    output_stem: str,
    seed: int,
) -> Path:
    if input_path.exists():
        return input_path

    config = get_default_config()
    records = generate_global_dataset(
        config=config,
        total_samples=total_samples,
        seed=seed,
    )
    save_dataset_artifacts(records, config, output_stem)
    return input_path


def main() -> int:
    configure_logging()

    parser = argparse.ArgumentParser(
        description='Sanity check da defesa por substituição semântica.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed opcional.',
    )
    parser.add_argument(
        '--total-samples',
        type=int,
        default=30,
        help='Quantidade de exemplos para geração automática se o dataset não existir.',
    )
    parser.add_argument(
        '--input-stem',
        type=str,
        default='semantic_check_input',
        help='Prefixo do dataset raw em data/raw.',
    )
    parser.add_argument(
        '--output-stem',
        type=str,
        default='semantic_check_output',
        help='Prefixo do dataset transformado em data/processed.',
    )
    parser.add_argument(
        '--show-examples',
        type=int,
        default=5,
        help='Quantidade de pares original/transformado para log.',
    )
    parser.add_argument(
        '--protect-canaries',
        action='store_true',
        help='Ativa proteção explícita de canários neste sanity check.',
    )
    args = parser.parse_args()

    config = get_default_config()
    effective_seed = config.seed if args.seed is None else args.seed
    set_seed(effective_seed)

    if args.protect_canaries:
        config.defense.protect_canaries = True

    raw_input_path = RAW_DATA_DIR / f'{args.input_stem}.jsonl'
    _ensure_input_dataset(
        input_path=raw_input_path,
        total_samples=args.total_samples,
        output_stem=args.input_stem,
        seed=effective_seed,
    )

    original_records = load_jsonl(raw_input_path)
    transformed_records = transform_records_semantic(
        records=original_records,
        defense_config=config.defense,
        seed=effective_seed,
    )

    stats = summarize_semantic_transformation(
        original_records=original_records,
        transformed_records=transformed_records,
        defense_config=config.defense,
    )

    transformed_path = PROCESSED_DATA_DIR / f'{args.output_stem}.jsonl'
    transformed_summary_path = PROCESSED_DATA_DIR / f'{args.output_stem}_summary.json'
    original_summary_path = PROCESSED_DATA_DIR / f'{args.output_stem}_raw_summary.json'

    save_jsonl(transformed_records, transformed_path)
    save_json(build_dataset_summary(original_records, config), original_summary_path)
    save_json(stats.__dict__, transformed_summary_path)

    LOGGER.info('Resumo da transformação: %s', stats)
    LOGGER.info('Dataset transformado salvo em: %s', transformed_path)
    LOGGER.info('Resumo transformado salvo em: %s', transformed_summary_path)

    display_count = min(args.show_examples, len(original_records))
    for idx in range(display_count):
        original = original_records[idx]
        transformed = transformed_records[idx]

        LOGGER.info('--- EXEMPLO %d ---', idx)
        LOGGER.info('ORIGINAL TEXT: %s', original['text'])
        LOGGER.info('TRANSFORMED TEXT: %s', transformed['text'])
        LOGGER.info('ORIGINAL ENTITIES: %s', original.get('entities', {}))
        LOGGER.info('TRANSFORMED ENTITIES: %s', transformed.get('entities', {}))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
