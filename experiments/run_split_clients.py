from __future__ import annotations

import argparse
from pathlib import Path

from configs.experiment_config import get_default_config
from src.data.split_clients import run_partition_pipeline
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import set_seed


def main() -> int:
    configure_logging()
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(
        description='Particiona o dataset global em clientes federados e cria as condições experimentais.'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Caminho opcional para o dataset global em JSONL.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed opcional.',
    )
    parser.add_argument(
        '--num-clients',
        type=int,
        default=None,
        help='Sobrescreve o número de clientes.',
    )
    parser.add_argument(
        '--samples-per-client',
        type=int,
        default=None,
        help='Sobrescreve o número de exemplos por cliente.',
    )
    args = parser.parse_args()

    config = get_default_config()
    if args.num_clients is not None:
        config.dataset.num_clients = args.num_clients
    if args.samples_per_client is not None:
        config.dataset.samples_per_client = args.samples_per_client

    seed = config.seed if args.seed is None else args.seed
    set_seed(seed)

    dataset_path = Path(args.dataset_path) if args.dataset_path is not None else None

    result = run_partition_pipeline(
        config=config,
        dataset_path=dataset_path,
        seed=seed,
    )

    logger.info('Partition completed', result=result)
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
