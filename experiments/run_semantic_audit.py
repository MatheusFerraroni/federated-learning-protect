from __future__ import annotations

import argparse
from pathlib import Path

from configs.paths import (
    CLIENT_RAW_DATA_DIR,
    CLIENT_SEMANTIC_DATA_DIR,
    GLOBAL_SPLITS_DIR,
    RESULTS_DIR,
)
from src.data.semantic_audit import (
    audit_semantic_dataset_from_paths,
    save_semantic_audit_artifacts,
)
from src.utils.logging_utils import configure_logging, get_logger


LOGGER = get_logger(__name__)


VALID_SPLITS = {'train', 'val', 'test', 'domain_test'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audita se o dataset com substituição semântica ainda contém valores originais.'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=sorted(VALID_SPLITS),
        help='Split a auditar.',
    )
    parser.add_argument(
        '--client-id',
        type=str,
        default=None,
        help='Opcional. Se informado, audita o split do cliente em vez do split global.',
    )
    parser.add_argument(
        '--raw-path',
        type=str,
        default=None,
        help='Opcional. Caminho explícito do JSONL raw.',
    )
    parser.add_argument(
        '--transformed-path',
        type=str,
        default=None,
        help='Opcional. Caminho explícito do JSONL transformado.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Diretório de saída. Default: results/semantic_audit/',
    )
    parser.add_argument(
        '--output-stem',
        type=str,
        default=None,
        help='Prefixo dos artefatos. Se omitido, será gerado automaticamente.',
    )
    parser.add_argument(
        '--show-examples',
        type=int,
        default=5,
        help='Quantidade de exemplos com problema para log.',
    )
    return parser.parse_args()


def resolve_default_paths(split_name: str, client_id: str | None) -> tuple[Path, Path]:
    """Resolve caminhos padrão para auditoria semântica.

    Args:
        split_name: Nome do split.
        client_id: ID do cliente (opcional).

    Returns:
        Tuple com caminhos raw e transformado.

    Raises:
        FileNotFoundError: Se global_splits não existir.
    """
    if client_id is not None:
        raw_path = CLIENT_RAW_DATA_DIR / client_id / f'{split_name}.jsonl'
        transformed_path = CLIENT_SEMANTIC_DATA_DIR / client_id / f'{split_name}.jsonl'
        return raw_path, transformed_path

    raw_path = GLOBAL_SPLITS_DIR / 'raw' / f'{split_name}.jsonl'
    transformed_path = GLOBAL_SPLITS_DIR / 'semantic_substitution' / f'{split_name}.jsonl'

    if not raw_path.exists() or not transformed_path.exists():
        raise FileNotFoundError(
            '\n[ERRO] Global splits não encontrados.\n\n'
            'O semantic audit requer arquivos em:\n'
            f'  {GLOBAL_SPLITS_DIR}/raw/{split_name}.jsonl\n'
            f'  {GLOBAL_SPLITS_DIR}/semantic_substitution/{split_name}.jsonl\n\n'
            'Você provavelmente ainda não executou:\n'
            '  python -m experiments.run_split_clients\n\n'
            'Fluxo correto:\n'
            '  1. run_generate_dataset\n'
            '  2. run_split_clients   ← faltando\n'
            '  3. run_semantic_audit\n'
        )

    return raw_path, transformed_path


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.raw_path is not None and args.transformed_path is not None:
        return Path(args.raw_path), Path(args.transformed_path)

    if args.raw_path is not None or args.transformed_path is not None:
        raise ValueError('Informe os dois caminhos explícitos: --raw-path e --transformed-path.')

    return resolve_default_paths(split_name=args.split, client_id=args.client_id)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_DIR / 'semantic_audit'

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_output_stem(args: argparse.Namespace) -> str:
    if args.output_stem is not None:
        return args.output_stem

    if args.client_id is not None:
        return f'semantic_audit_{args.client_id}_{args.split}'

    return f'semantic_audit_global_{args.split}'


def main() -> int:
    configure_logging()
    args = parse_args()

    raw_path, transformed_path = resolve_paths(args)
    output_dir = resolve_output_dir(args)
    output_stem = build_output_stem(args)

    if not raw_path.exists() or not transformed_path.exists():
        raise FileNotFoundError(
            f'\n[ERRO] Arquivos não encontrados:\n'
            f'raw: {raw_path}\n'
            f'transformed: {transformed_path}\n\n'
            'Verifique se você executou:\n'
            '  python -m experiments.run_split_clients\n'
        )

    LOGGER.info(
        'Iniciando auditoria semântica: raw_path=%s transformed_path=%s output_dir=%s',
        raw_path,
        transformed_path,
        output_dir,
    )

    report = audit_semantic_dataset_from_paths(
        raw_path=raw_path,
        transformed_path=transformed_path,
    )
    artifact_paths = save_semantic_audit_artifacts(
        report=report,
        output_dir=output_dir,
        output_stem=output_stem,
    )

    summary = report['summary']
    LOGGER.info(
        (
            'Auditoria concluída: num_records=%d '
            'clean_train_text_ratio=%.6f '
            'clean_full_record_ratio=%.6f '
            'num_records_with_original_entity_in_text=%d '
            'num_records_with_auxiliary_original_value=%d '
            'summary_path=%s'
        ),
        summary['num_records'],
        summary['clean_train_text_ratio'],
        summary['clean_full_record_ratio'],
        summary['num_records_with_original_entity_in_text'],
        summary['num_records_with_auxiliary_original_value'],
        artifact_paths['summary_path'],
    )

    problematic_rows = [
        row
        for row in report['rows']
        if (not row['train_text_is_clean']) or (row['num_auxiliary_original_value_paths'] > 0)
    ]

    display_count = min(args.show_examples, len(problematic_rows))
    for index in range(display_count):
        row = problematic_rows[index]
        LOGGER.warning(
            (
                'EXEMPLO_PROBLEMA index=%d record_id=%s text_clean=%s full_clean=%s '
                'original_entity_hits=%s auxiliary_paths=%s '
                'original_text=%s transformed_text=%s'
            ),
            index,
            row['record_id'],
            row['train_text_is_clean'],
            row['full_record_is_clean'],
            row['original_entity_hits_in_transformed_text'],
            row['auxiliary_original_value_paths'],
            row['original_text'],
            row['transformed_text'],
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
