"""Executa a condição baseline com atacante + DP-SGD / Opacus.

Responsabilidades principais:
- Executar sanity check centralizado na condição ``attack_raw`` com DP-SGD.
- Executar avaliação de vazamento no checkpoint do sanity check.
- Executar treino federado com FedAvg usando clientes ``attack_raw`` sob DP.
- Executar avaliação de vazamento no checkpoint final federado.
- Consolidar um resumo único da baseline [D].

Como este arquivo se encaixa no projeto:
- Baseline comparativa do experimento
- Condição [D] do protocolo experimental
- Permite comparar utilidade e leakage da baseline DP contra ataque sem defesa
  e contra substituição semântica.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from configs.paths import CLIENT_ATTACK_RAW_DATA_DIR, RESULTS_DIR
from src.utils.experiment_runner import (
    build_condition_summary,
    find_latest_directory_by_prefix,
    persist_summary,
    resolve_federated_run_dir,
    run_python_module,
)
from src.utils.logging_utils import configure_logging, get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Executa a baseline com atacante + DP-SGD (attack_raw).'
    )
    parser.add_argument('--seed', type=int, default=42, help='Seed global.')
    parser.add_argument('--epochs', type=int, default=1, help='Épocas locais/centrais.')
    parser.add_argument('--rounds', type=int, default=1, help='Número de rodadas federadas.')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size lógico de treino.')
    parser.add_argument('--eval-batch-size', type=int, default=4, help='Batch size de avaliação.')
    parser.add_argument('--max-length', type=int, default=64, help='Comprimento máximo.')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument(
        '--num-workers', type=int, default=0, help='Número de workers do DataLoader.'
    )
    parser.add_argument('--num-prompts', type=int, default=100, help='Número de prompts de ataque.')
    parser.add_argument(
        '--dp-noise-multiplier', type=float, default=1.1, help='Noise multiplier da baseline DP.'
    )
    parser.add_argument(
        '--dp-max-grad-norm', type=float, default=1.0, help='Max grad norm da baseline DP.'
    )
    parser.add_argument(
        '--dp-target-epsilon', type=float, default=None, help='Budget epsilon alvo opcional.'
    )
    parser.add_argument(
        '--dp-target-delta', type=float, default=1e-5, help='Delta alvo da contabilidade DP.'
    )
    parser.add_argument(
        '--dp-max-physical-batch-size',
        type=int,
        default=None,
        help='Batch físico opcional do BatchMemoryManager.',
    )
    parser.add_argument(
        '--dp-num-trainable-transformer-blocks',
        type=int,
        default=2,
        help='Quantidade de blocos finais treináveis sob DP.',
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Ativa AMP em CUDA durante o treino.',
    )
    parser.add_argument(
        '--run-name', type=str, default=None, help='Nome base opcional da execução.'
    )
    return parser.parse_args()


def build_base_run_name(explicit_name: str | None) -> str:
    """Resolve o nome base da execução."""
    if explicit_name:
        return explicit_name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'pipeline_attack_dp_{timestamp}'


def _resolve_clients_per_round() -> int:
    """Resolve dinamicamente quantos clientes podem participar por rodada."""
    if not CLIENT_ATTACK_RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f'Diretório de clientes da condição não encontrado: {CLIENT_ATTACK_RAW_DATA_DIR}'
        )

    client_dirs = sorted(path for path in CLIENT_ATTACK_RAW_DATA_DIR.iterdir() if path.is_dir())
    if not client_dirs:
        raise RuntimeError(
            f'Nenhum cliente encontrado para a condição attack_raw em {CLIENT_ATTACK_RAW_DATA_DIR}'
        )

    return max(1, min(len(client_dirs), 5))


def main() -> int:
    """Executa o pipeline completo da baseline DP."""
    configure_logging()
    args = parse_args()

    condition = 'attack_raw'
    reference_condition = 'raw'
    base_run_name = build_base_run_name(args.run_name)

    output_dir = RESULTS_DIR / base_run_name
    logs_dir = output_dir / 'logs'
    summaries_dir = output_dir / 'summaries'
    output_dir.mkdir(parents=True, exist_ok=True)

    dp_args = [
        '--dp-enabled',
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
    if args.fp16:
        LOGGER.warning(
            'Flag --fp16 recebida em run_dp_baseline, mas AMP permanece desativado '
            'quando DP-SGD/Opacus está ativo.'
        )

    command_results = []

    sanity_run_name = f'{base_run_name}_sanity'
    command_results.append(
        run_python_module(
            module='experiments.run_sanity_check',
            args=[
                '--condition',
                condition,
                '--epochs',
                str(args.epochs),
                '--batch-size',
                str(args.batch_size),
                '--eval-batch-size',
                str(args.eval_batch_size),
                '--max-length',
                str(args.max_length),
                '--learning-rate',
                str(args.learning_rate),
                '--seed',
                str(args.seed),
                '--run-name',
                sanity_run_name,
                '--num-workers',
                str(args.num_workers),
                *dp_args,
            ],
            logs_dir=logs_dir,
            step_name='01_sanity_check_dp',
        )
    )

    sanity_dir = find_latest_directory_by_prefix(RESULTS_DIR, sanity_run_name)
    sanity_checkpoint_dir = sanity_dir / 'checkpoint'

    sanity_attack_output_dir = output_dir / 'attack_sanity_attack_raw_dp'
    command_results.append(
        run_python_module(
            module='experiments.run_attack_eval',
            args=[
                '--checkpoint-dir',
                str(sanity_checkpoint_dir),
                '--condition',
                condition,
                '--reference-condition',
                reference_condition,
                '--num-prompts',
                str(args.num_prompts),
                '--seed',
                str(args.seed),
                '--output-dir',
                str(sanity_attack_output_dir),
            ],
            logs_dir=logs_dir,
            step_name='02_attack_sanity_dp',
        )
    )

    clients_per_round = _resolve_clients_per_round()

    federated_run_name = f'{base_run_name}_federated'
    command_results.append(
        run_python_module(
            module='src.fl.run_federated',
            args=[
                '--condition',
                condition,
                '--rounds',
                str(args.rounds),
                '--clients-per-round',
                str(clients_per_round),
                '--local-epochs',
                str(args.epochs),
                '--batch-size',
                str(args.batch_size),
                '--eval-batch-size',
                str(args.eval_batch_size),
                '--max-length',
                str(args.max_length),
                '--learning-rate',
                str(args.learning_rate),
                '--seed',
                str(args.seed),
                '--run-name',
                federated_run_name,
                '--num-workers',
                str(args.num_workers),
                *dp_args,
            ],
            logs_dir=logs_dir,
            step_name='03_federated_train_dp',
        )
    )

    federated_dir = resolve_federated_run_dir(
        base_dir=RESULTS_DIR, requested_run_name=federated_run_name, condition_name=condition
    )
    federated_checkpoint_dir = (
        sorted(
            path
            for path in federated_dir.iterdir()
            if path.is_dir() and path.name.startswith('round_')
        )[-1]
        / 'checkpoint'
    )

    federated_attack_output_dir = output_dir / 'attack_federated_attack_raw_dp'
    command_results.append(
        run_python_module(
            module='experiments.run_attack_eval',
            args=[
                '--checkpoint-dir',
                str(federated_checkpoint_dir),
                '--condition',
                condition,
                '--reference-condition',
                reference_condition,
                '--num-prompts',
                str(args.num_prompts),
                '--seed',
                str(args.seed),
                '--output-dir',
                str(federated_attack_output_dir),
            ],
            logs_dir=logs_dir,
            step_name='04_attack_federated_dp',
        )
    )

    summary = build_condition_summary(
        condition='attack_raw_dp',
        sanity_dir=sanity_dir,
        sanity_attack_dir=sanity_attack_output_dir,
        federated_dir=federated_dir,
        federated_attack_dir=federated_attack_output_dir,
        command_results=command_results,
    )
    summary['dp_config'] = {
        'enabled': True,
        'noise_multiplier': args.dp_noise_multiplier,
        'max_grad_norm': args.dp_max_grad_norm,
        'target_epsilon': args.dp_target_epsilon,
        'target_delta': args.dp_target_delta,
        'max_physical_batch_size': args.dp_max_physical_batch_size,
        'num_trainable_transformer_blocks': args.dp_num_trainable_transformer_blocks,
    }

    summary_path = persist_summary(summary, summaries_dir / 'attack_raw_dp_pipeline_summary.json')
    LOGGER.info('Pipeline attack_raw_dp concluído. Resumo salvo em %s', summary_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
