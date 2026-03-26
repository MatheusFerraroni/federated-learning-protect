from __future__ import annotations

import argparse

from configs.experiment_config import get_default_config
from src.data.generate_dataset import (
    build_dataset_summary,
    generate_global_dataset,
    save_dataset_artifacts,
)
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    config = get_default_config()

    parser = argparse.ArgumentParser(description='Gera dataset sintético global.')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed opcional.',
    )
    parser.add_argument(
        '--total-samples',
        type=int,
        default=config.dataset.num_clients * config.dataset.samples_per_client,
        help='Número total de registros a gerar.',
    )
    parser.add_argument(
        '--output-stem',
        type=str,
        default=config.dataset.output_dataset_name,
        help='Prefixo do arquivo de saída.',
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    config = get_default_config()
    effective_seed = config.seed if args.seed is None else args.seed
    set_seed(effective_seed)

    records = generate_global_dataset(
        config=config,
        total_samples=args.total_samples,
        seed=effective_seed,
    )
    summary = build_dataset_summary(records, config)
    output_paths = save_dataset_artifacts(
        records=records,
        config=config,
        output_stem=args.output_stem,
    )

    LOGGER.info('Resumo do dataset: %s', summary)
    LOGGER.info('Arquivos salvos: %s', output_paths)

    if records:
        LOGGER.info('Exemplos:')
        for idx, record in enumerate(records[:5]):
            LOGGER.info(
                '[%d] type=%s | canary=%s | text=%s',
                idx,
                record.get('template_type'),
                record.get('is_canary'),
                record.get('text'),
            )

        sensitive_example = next(
            (record for record in records if record.get('has_sensitive', False)),
            None,
        )
        if sensitive_example is not None:
            LOGGER.info('Exemplo sensível com entidades: %s', sensitive_example)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
