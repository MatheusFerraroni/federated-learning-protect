"""Script mestre para executar as condições principais e consolidar a comparação.

Responsabilidades principais:
- Executar os pipelines ponta a ponta de:
  [A] sem atacante,
  [B] com atacante,
  [C] com atacante + defesa semântica,
  [D] com atacante + baseline DP-SGD / Opacus.
- Consolidar os summaries individuais em um resumo agregado.
- Preservar o resumo legado de três condições e anexar a baseline DP.

Como este arquivo se encaixa no projeto:
- É o orquestrador principal para rodar o experimento comparativo completo.
- Centraliza a execução reprodutível dos pipelines experimentais.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from configs.paths import RESULTS_DIR
from src.utils.experiment_runner import (
    build_three_condition_summary,
    persist_summary,
    persist_text_summary,
    run_python_module,
)
from src.utils.io import load_json
from src.utils.logging_utils import configure_logging, get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Executa todas as condições experimentais e consolida os resultados.'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--num-prompts', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=0)

    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Ativa AMP em CUDA nos scripts filhos.',
    )

    parser.add_argument('--dp-noise-multiplier', type=float, default=1.1)
    parser.add_argument('--dp-max-grad-norm', type=float, default=1.0)
    parser.add_argument('--dp-target-epsilon', type=float, default=None)
    parser.add_argument('--dp-target-delta', type=float, default=1e-5)
    parser.add_argument('--dp-max-physical-batch-size', type=int, default=None)
    parser.add_argument('--dp-num-trainable-transformer-blocks', type=int, default=2)

    parser.add_argument('--run-name', type=str, default=None)
    return parser.parse_args()


def build_base_run_name(explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    return f'pipeline_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}'


def _build_common_args(args: argparse.Namespace) -> list[str]:
    common_args = [
        '--seed',
        str(args.seed),
        '--epochs',
        str(args.epochs),
        '--rounds',
        str(args.rounds),
        '--batch-size',
        str(args.batch_size),
        '--eval-batch-size',
        str(args.eval_batch_size),
        '--max-length',
        str(args.max_length),
        '--learning-rate',
        str(args.learning_rate),
        '--num-workers',
        str(args.num_workers),
        '--num-prompts',
        str(args.num_prompts),
    ]

    if args.fp16:
        common_args.append('--fp16')

    return common_args


def _build_dp_args(args: argparse.Namespace) -> list[str]:
    dp_args = [
        '--dp-noise-multiplier',
        str(args.dp_noise_multiplier),
        '--dp-max-grad-norm',
        str(args.dp_max_grad_norm),
        '--dp-target-delta',
        str(args.dp_target_delta),
        '--dp-num-trainable-transformer-blocks',
        str(args.dp_num_trainable_transformer_blocks),
    ]

    if args.dp_target_epsilon is not None:
        dp_args.extend(['--dp-target-epsilon', str(args.dp_target_epsilon)])

    if args.dp_max_physical_batch_size is not None:
        dp_args.extend(['--dp-max-physical-batch-size', str(args.dp_max_physical_batch_size)])

    return dp_args


def _load_summary(run_dir_name: str, relative_summary_path: str) -> dict[str, Any]:
    summary_path = RESULTS_DIR / run_dir_name / relative_summary_path
    LOGGER.info('Carregando summary: %s', summary_path)
    return load_json(summary_path)


def _save_manual_text_summary(summary: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return output_path


def main() -> int:
    configure_logging()
    args = parse_args()
    LOGGER.info('Starting')

    base_run_name = build_base_run_name(args.run_name)

    output_dir = RESULTS_DIR / base_run_name
    logs_dir = output_dir / 'logs'
    comparison_dir = output_dir / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    common_args = _build_common_args(args)
    dp_args = _build_dp_args(args)

    no_attacker_run_name = f'{base_run_name}_no_attacker'
    attack_run_name = f'{base_run_name}_attack'
    defense_run_name = f'{base_run_name}_attack_semantic_defense'
    dp_run_name = f'{base_run_name}_attack_dp'

    command_results = []

    LOGGER.info('Running sub modules 1/4')
    command_results.append(
        run_python_module(
            module='experiments.run_no_attacker',
            args=[*common_args, '--run-name', no_attacker_run_name],
            logs_dir=logs_dir,
            step_name='01_pipeline_no_attacker',
        )
    )

    LOGGER.info('Running sub modules 2/4')
    command_results.append(
        run_python_module(
            module='experiments.run_attack_condition',
            args=[*common_args, '--run-name', attack_run_name],
            logs_dir=logs_dir,
            step_name='02_pipeline_attack',
        )
    )

    LOGGER.info('Running sub modules 3/4')
    command_results.append(
        run_python_module(
            module='experiments.run_semantic_defense',
            args=[*common_args, '--run-name', defense_run_name],
            logs_dir=logs_dir,
            step_name='03_pipeline_attack_semantic_defense',
        )
    )

    LOGGER.info('Running sub modules 4/4')
    command_results.append(
        run_python_module(
            module='experiments.run_dp_baseline',
            args=[*common_args, *dp_args, '--run-name', dp_run_name],
            logs_dir=logs_dir,
            step_name='04_pipeline_attack_dp',
        )
    )
    LOGGER.info('Running sub modules complete')

    no_attacker_summary = _load_summary(
        no_attacker_run_name,
        'summaries/no_attacker_pipeline_summary.json',
    )
    attack_summary = _load_summary(
        attack_run_name,
        'summaries/attack_raw_pipeline_summary.json',
    )
    defense_summary = _load_summary(
        defense_run_name,
        'summaries/attack_semantic_substitution_pipeline_summary.json',
    )
    dp_summary = _load_summary(
        dp_run_name,
        'summaries/attack_raw_dp_pipeline_summary.json',
    )

    three_condition_summary = build_three_condition_summary(
        no_attacker_summary,
        attack_summary,
        defense_summary,
    )

    four_condition_summary: dict[str, Any] = {
        'run_name': base_run_name,
        'generated_at': datetime.now().isoformat(),
        'comparison_type': 'four_condition',
        'base_three_condition_summary': three_condition_summary,
        'conditions': {
            'no_attacker': no_attacker_summary,
            'attack_raw': attack_summary,
            'attack_semantic_substitution': defense_summary,
            'attack_raw_dp': dp_summary,
        },
        'runtime_config': {
            'seed': args.seed,
            'epochs': args.epochs,
            'rounds': args.rounds,
            'batch_size': args.batch_size,
            'eval_batch_size': args.eval_batch_size,
            'max_length': args.max_length,
            'learning_rate': args.learning_rate,
            'num_prompts': args.num_prompts,
            'fp16': args.fp16,
        },
        'dp_config': {
            'noise_multiplier': args.dp_noise_multiplier,
            'max_grad_norm': args.dp_max_grad_norm,
            'target_epsilon': args.dp_target_epsilon,
            'target_delta': args.dp_target_delta,
            'max_physical_batch_size': args.dp_max_physical_batch_size,
            'num_trainable_transformer_blocks': args.dp_num_trainable_transformer_blocks,
        },
        'artifacts': {
            'no_attacker_dir': str(RESULTS_DIR / no_attacker_run_name),
            'attack_dir': str(RESULTS_DIR / attack_run_name),
            'semantic_defense_dir': str(RESULTS_DIR / defense_run_name),
            'dp_dir': str(RESULTS_DIR / dp_run_name),
        },
        'command_results': [asdict(item) for item in command_results],
    }

    legacy_json_path = comparison_dir / 'three_condition_summary.json'
    legacy_txt_path = comparison_dir / 'three_condition_summary.txt'
    persist_summary(three_condition_summary, legacy_json_path)
    persist_text_summary(three_condition_summary, legacy_txt_path)

    four_json_path = comparison_dir / 'four_condition_summary.json'
    persist_summary(four_condition_summary, four_json_path)

    four_txt_path = comparison_dir / 'four_condition_summary.txt'
    _save_manual_text_summary(four_condition_summary, four_txt_path)

    LOGGER.info(
        ('Execução agregada concluída. three_condition_summary=%s four_condition_summary=%s'),
        legacy_json_path,
        four_json_path,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
