"""
Script de execução ponta a ponta para a condição com atacante e defesa semântica.

Responsabilidades principais:
- Executar sanity check centralizado usando attack_semantic_substitution.
- Executar avaliação de vazamento sobre o checkpoint do sanity check.
- Executar treino federado com FedAvg usando attack_semantic_substitution.
- Executar avaliação de vazamento no checkpoint final federado.
- Medir leakage contra a referência raw, que contém os segredos originais.

Como este arquivo se encaixa no projeto:
- É o ponto de entrada prático da condição [C] "com atacante + defesa".
- Permite comparar utilidade e redução de leakage contra attack_raw.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from configs.paths import (
    CLIENT_ATTACK_SEMANTIC_DATA_DIR,
    RESULTS_DIR,
)
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
        description='Executa o pipeline completo da condição attack_semantic_substitution.'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--num-prompts', type=int, default=100)
    parser.add_argument(
        '--num-workers', type=int, default=0, help='Número de workers do DataLoader.'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Ativa AMP em CUDA durante o treino.',
    )
    parser.add_argument('--run-name', type=str, default=None)
    return parser.parse_args()


def build_base_run_name(explicit_name: str | None) -> str:
    """Resolve nome base."""
    if explicit_name:
        return explicit_name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'pipeline_attack_semantic_substitution_{timestamp}'


def _resolve_clients_per_round() -> int:
    """Resolve número de clientes disponíveis dinamicamente.

    Returns:
        Número válido de clientes por rodada (>=1 e <=5).

    Raises:
        FileNotFoundError: Se diretório não existir.
        RuntimeError: Se não houver clientes.
    """
    if not CLIENT_ATTACK_SEMANTIC_DATA_DIR.exists():
        raise FileNotFoundError(f'Diretório não encontrado: {CLIENT_ATTACK_SEMANTIC_DATA_DIR}')

    client_dirs = [p for p in CLIENT_ATTACK_SEMANTIC_DATA_DIR.iterdir() if p.is_dir()]

    if not client_dirs:
        raise RuntimeError(f'Nenhum cliente encontrado em {CLIENT_ATTACK_SEMANTIC_DATA_DIR}')

    return max(1, min(len(client_dirs), 5))


def main() -> int:
    configure_logging()
    args = parse_args()

    condition = 'attack_semantic_substitution'
    reference_condition = 'raw'
    base_run_name = build_base_run_name(args.run_name)

    output_dir = RESULTS_DIR / base_run_name
    logs_dir = output_dir / 'logs'
    summaries_dir = output_dir / 'summaries'
    output_dir.mkdir(parents=True, exist_ok=True)

    command_results = []

    # ================= SANITY =================
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
                *(['--fp16'] if args.fp16 else []),
                '--num-workers',
                str(args.num_workers),
                '--run-name',
                sanity_run_name,
            ],
            logs_dir=logs_dir,
            step_name='01_sanity_check',
        )
    )

    sanity_dir = find_latest_directory_by_prefix(RESULTS_DIR, sanity_run_name)
    sanity_checkpoint_dir = sanity_dir / 'checkpoint'

    # ================= ATTACK SANITY =================
    sanity_attack_output_dir = output_dir / 'attack_sanity_attack_semantic'
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
            step_name='02_attack_sanity',
        )
    )

    # ================= FEDERATED =================
    clients_per_round = _resolve_clients_per_round()
    LOGGER.info(
        'Clients per round resolvido dinamicamente: condition=%s clients_per_round=%d',
        condition,
        clients_per_round,
    )

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
                *(['--fp16'] if args.fp16 else []),
                '--num-workers',
                str(args.num_workers),
                '--run-name',
                federated_run_name,
            ],
            logs_dir=logs_dir,
            step_name='03_federated_train',
        )
    )

    federated_dir = resolve_federated_run_dir(
        base_dir=RESULTS_DIR,
        requested_run_name=federated_run_name,
        condition_name=condition,
    )

    federated_checkpoint_dir = (
        sorted(p for p in federated_dir.iterdir() if p.is_dir() and p.name.startswith('round_'))[-1]
        / 'checkpoint'
    )

    # ================= ATTACK FINAL =================
    federated_attack_output_dir = output_dir / 'attack_federated_attack_semantic'
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
            step_name='04_attack_federated',
        )
    )

    summary = build_condition_summary(
        condition=condition,
        sanity_dir=sanity_dir,
        sanity_attack_dir=sanity_attack_output_dir,
        federated_dir=federated_dir,
        federated_attack_dir=federated_attack_output_dir,
        command_results=command_results,
    )

    summary_path = persist_summary(
        summary,
        summaries_dir / 'attack_semantic_substitution_pipeline_summary.json',
    )

    LOGGER.info('Pipeline concluído: %s', summary_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
