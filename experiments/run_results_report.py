"""
Gera relatório consolidado de resultados a partir dos pipelines experimentais já executados.

Responsabilidades principais:
- Localizar automaticamente os summaries produzidos por no_attacker, raw,
  semantic_substitution e baseline DP.
- Consolidar métricas de utilidade e leakage em formato tabular.
- Gerar CSVs, JSON final, TXT final e gráficos PNG.
- Servir como etapa final reprodutível de análise dos resultados experimentais.

Como este arquivo se encaixa no projeto:
- É a última camada do pipeline experimental.
- Consome artefatos já produzidos por run_no_attacker, run_attack_condition,
  run_semantic_defense e run_dp_baseline.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from configs.paths import RESULTS_DIR
from src.utils.io import load_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.plots import (
    build_comparison_summary,
    plot_leakage_metrics,
    plot_utility_metrics,
    save_comparison_csv,
    save_comparison_json,
    save_rows_csv,
    save_text_report,
    summaries_to_rows,
)

LOGGER = get_logger(__name__)

SUMMARY_FILENAME_BY_CONDITION = {
    'no_attacker': 'no_attacker_pipeline_summary.json',
    'raw': 'attack_raw_pipeline_summary.json',
    'semantic_substitution': 'attack_semantic_substitution_pipeline_summary.json',
    'dp': 'attack_raw_dp_pipeline_summary.json',
}

EXPECTED_DIRNAME_SUFFIX_BY_CONDITION = {
    'no_attacker': '_no_attacker',
    'raw': '_attack',
    'semantic_substitution': '_attack_semantic_defense',
    'dp': '_attack_dp',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Consolida resultados experimentais em tabelas, gráficos e resumo final.'
    )
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--no-attacker-dir', type=str, default=None)
    parser.add_argument('--raw-dir', type=str, default=None)
    parser.add_argument('--semantic-dir', type=str, default=None)
    parser.add_argument('--dp-dir', type=str, default=None)
    parser.add_argument('--run-prefix', type=str, default=None)
    return parser.parse_args()


def _resolve_output_dir(explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = (RESULTS_DIR / f'results_report_{timestamp}').resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _find_latest_directory_by_prefix(base_dir: Path, prefix: str) -> Path:
    candidates = [
        path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)
    ]
    if not candidates:
        raise FileNotFoundError(f"Nenhum diretório encontrado com prefixo '{prefix}' em {base_dir}")
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_latest_directory_by_exact_name(base_dir: Path, name: str) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir() and path.name == name]
    if not candidates:
        raise FileNotFoundError(f"Nenhum diretório encontrado com nome '{name}' em {base_dir}")
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_summary_in_directory(base_dir: Path, summary_filename: str) -> Path:
    base_dir = base_dir.expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f'Diretório não encontrado: {base_dir}')

    matches = list(base_dir.rglob(summary_filename))
    if not matches:
        raise FileNotFoundError(f"Arquivo '{summary_filename}' não encontrado em {base_dir}")

    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0]


def _resolve_directory_from_run_prefix(condition_name: str, run_prefix: str) -> Path:
    expected_dirname = f'{run_prefix}{EXPECTED_DIRNAME_SUFFIX_BY_CONDITION[condition_name]}'
    return _find_latest_directory_by_exact_name(RESULTS_DIR, expected_dirname)


def _resolve_condition_summary_path(condition_name, explicit_dir, run_prefix) -> Path:
    summary_filename = SUMMARY_FILENAME_BY_CONDITION[condition_name]

    if explicit_dir:
        return _find_summary_in_directory(Path(explicit_dir), summary_filename)

    if run_prefix:
        target_dir = _resolve_directory_from_run_prefix(condition_name, run_prefix)
        return _find_summary_in_directory(target_dir, summary_filename)

    raise FileNotFoundError(
        f'Não foi possível resolver automaticamente o summary para {condition_name}. '
        'Use --run-prefix ou diretórios explícitos.'
    )


def _load_condition_summaries(args: argparse.Namespace) -> list[dict[str, Any]]:
    condition_to_dir = {
        'no_attacker': args.no_attacker_dir,
        'raw': args.raw_dir,
        'semantic_substitution': args.semantic_dir,
        'dp': args.dp_dir,
    }

    summaries = []

    for condition_name, explicit_dir in condition_to_dir.items():
        summary_path = _resolve_condition_summary_path(
            condition_name, explicit_dir, args.run_prefix
        )
        summary = load_json(summary_path)

        LOGGER.info('Summary carregado: %s -> %s', condition_name, summary_path)

        # Garantir que a condição esteja explícita no summary
        summary['condition'] = condition_name

        summaries.append(summary)

    return summaries


def main() -> int:
    configure_logging()
    args = parse_args()

    output_dir = _resolve_output_dir(args.output_dir)
    tables_dir = output_dir / 'tables'
    plots_dir = output_dir / 'plots'
    reports_dir = output_dir / 'reports'

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries = _load_condition_summaries(args)

    # =========================
    # VALIDAÇÃO CRÍTICA
    # =========================
    expected_conditions = {'no_attacker', 'raw', 'semantic_substitution', 'dp'}
    loaded_conditions = {s.get('condition') for s in summaries}

    missing = expected_conditions - loaded_conditions
    if missing:
        raise ValueError(f'Condições ausentes no relatório: {missing}')

    LOGGER.info('Condições carregadas: %s', sorted(loaded_conditions))

    # =========================
    # CONVERSÃO PARA TABELA
    # =========================
    rows = summaries_to_rows(summaries)

    if not rows:
        raise ValueError('Nenhuma linha gerada a partir dos summaries.')

    # =========================
    # COMPARAÇÃO ENTRE CONDIÇÕES
    # =========================
    comparison_summary = build_comparison_summary(rows)

    # =========================
    # SALVAMENTO
    # =========================
    condition_csv_path = save_rows_csv(rows, tables_dir / 'condition_metrics.csv')

    comparison_csv_path = save_comparison_csv(
        comparison_summary,
        tables_dir / 'comparison_metrics.csv',
    )

    comparison_json_path = save_comparison_json(
        comparison_summary,
        reports_dir / 'final_results_summary.json',
    )

    text_report_path = save_text_report(
        comparison_summary,
        reports_dir / 'final_results_summary.txt',
    )

    # =========================
    # PLOTS
    # =========================
    utility_plot_paths = plot_utility_metrics(rows, plots_dir)
    leakage_plot_paths = plot_leakage_metrics(rows, plots_dir)

    # =========================
    # MANIFEST FINAL
    # =========================
    final_manifest = {
        'conditions_included': sorted(loaded_conditions),
        'condition_metrics_csv': str(condition_csv_path),
        'comparison_metrics_csv': str(comparison_csv_path),
        'final_results_summary_json': str(comparison_json_path),
        'final_results_summary_txt': str(text_report_path),
        'plots': {
            **{k: str(v) for k, v in utility_plot_paths.items()},
            **{k: str(v) for k, v in leakage_plot_paths.items()},
        },
    }

    LOGGER.info('Relatório final gerado com sucesso:')
    for key, value in final_manifest.items():
        LOGGER.info('  %s: %s', key, value)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
