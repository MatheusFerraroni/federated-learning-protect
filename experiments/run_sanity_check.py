"""Executa sanity check centralizado para causal language modeling.

Responsabilidades principais:
- Carregar configuração base e aplicar overrides de CLI.
- Treinar centralizadamente em um split global de uma condição experimental.
- Avaliar o modelo em train/val/test/domain_test.
- Salvar checkpoint, relatório e amostras qualitativas geradas.
- Permitir baseline opcional com DP-SGD / Opacus no treino centralizado.

Como este arquivo se encaixa no projeto:
- Fornece um experimento mínimo de sanidade antes do loop federado.
- Agora suporta tanto condições limpas quanto condições com atacante.
- Também serve como teste rápido para a baseline DP antes do uso em FL.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from configs.experiment_config import ExperimentConfig, get_default_config
from src.model.generate_samples import (
    DEFAULT_PROMPTS,
    generate_from_prompts,
    save_generated_samples,
    save_generated_samples_txt,
)
from src.model.model_utils import describe_model, get_device, load_causal_lm_model, load_tokenizer
from src.model.train_local import (
    resolve_run_output_dir,
    save_training_run_artifacts,
    train_on_global_condition,
)
from src.utils.io import save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)

VALID_CONDITIONS = ['raw', 'semantic_substitution', 'attack_raw', 'attack_semantic_substitution']


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Executa treino centralizado de sanidade para causal LM.'
    )
    parser.add_argument(
        '--condition',
        type=str,
        default='raw',
        choices=VALID_CONDITIONS,
        help='Condição experimental a usar.',
    )
    parser.add_argument(
        '--epochs', type=int, default=None, help='Override para config.model.local_epochs.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None, help='Override para config.model.batch_size.'
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=None,
        help='Override para config.model.eval_batch_size.',
    )
    parser.add_argument(
        '--max-length', type=int, default=None, help='Override para config.model.max_length.'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Override para config.model.learning_rate.',
    )
    parser.add_argument('--fp16', action='store_true', help='Ativa AMP em CUDA durante o treino.')
    parser.add_argument(
        '--num-workers', type=int, default=None, help='Override para config.model.num_workers.'
    )
    parser.add_argument('--seed', type=int, default=None, help='Seed opcional.')
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Nome da execução. Se omitido, será gerado automaticamente.',
    )
    parser.add_argument(
        '--dp-enabled',
        action='store_true',
        help='Ativa baseline DP-SGD com Opacus no sanity check.',
    )
    parser.add_argument(
        '--dp-noise-multiplier', type=float, default=None, help='Noise multiplier da baseline DP.'
    )
    parser.add_argument(
        '--dp-max-grad-norm', type=float, default=None, help='Max grad norm da baseline DP.'
    )
    parser.add_argument(
        '--dp-target-epsilon',
        type=float,
        default=None,
        help='Budget epsilon alvo opcional para make_private_with_epsilon.',
    )
    parser.add_argument(
        '--dp-target-delta', type=float, default=None, help='Delta alvo da contabilidade DP.'
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
        default=None,
        help='Quantidade de blocos finais treináveis sob DP.',
    )
    parser.add_argument(
        '--dp-train-embeddings',
        action='store_true',
        help='Mantém embeddings treináveis na baseline DP.',
    )
    parser.add_argument(
        '--dp-disable-poisson-sampling',
        action='store_true',
        help='Desabilita amostragem de Poisson na baseline DP.',
    )
    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Aplica overrides de linha de comando na configuração."""
    updated = deepcopy(config)

    if args.epochs is not None:
        updated.model.local_epochs = args.epochs
    if args.batch_size is not None:
        updated.model.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        updated.model.eval_batch_size = args.eval_batch_size
    if args.max_length is not None:
        updated.model.max_length = args.max_length
    if args.learning_rate is not None:
        updated.model.learning_rate = args.learning_rate
    if args.seed is not None:
        updated.seed = args.seed
    if args.fp16:
        updated.model.fp16 = True
    if args.num_workers is not None:
        updated.model.num_workers = args.num_workers

    if args.dp_enabled:
        updated.dp.enabled = True
    if args.dp_noise_multiplier is not None:
        updated.dp.noise_multiplier = args.dp_noise_multiplier
    if args.dp_max_grad_norm is not None:
        updated.dp.max_grad_norm = args.dp_max_grad_norm
    if args.dp_target_epsilon is not None:
        updated.dp.target_epsilon = args.dp_target_epsilon
    if args.dp_target_delta is not None:
        updated.dp.target_delta = args.dp_target_delta
    if args.dp_max_physical_batch_size is not None:
        updated.dp.max_physical_batch_size = args.dp_max_physical_batch_size
    if args.dp_num_trainable_transformer_blocks is not None:
        updated.dp.num_trainable_transformer_blocks = args.dp_num_trainable_transformer_blocks
    if args.dp_train_embeddings:
        updated.dp.freeze_embeddings = False
    if args.dp_disable_poisson_sampling:
        updated.dp.poisson_sampling = False

    return updated


def build_run_name(condition_name: str, run_name: str | None = None) -> str:
    """Resolve o nome da execução."""
    if run_name:
        return run_name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'sanity_check_{condition_name}_{timestamp}'


def save_run_summary(
    output_dir: Path,
    config: ExperimentConfig,
    condition_name: str,
    model_description: dict[str, Any],
    run_report: dict[str, Any],
    samples_path_json: Path,
    samples_path_txt: Path,
) -> Path:
    """Salva um resumo leve da execução de sanity check."""
    summary = {
        'condition': condition_name,
        'config': {
            'seed': config.seed,
            'model_name': config.model.model_name,
            'local_epochs': config.model.local_epochs,
            'batch_size': config.model.batch_size,
            'eval_batch_size': config.model.eval_batch_size,
            'max_length': config.model.max_length,
            'learning_rate': config.model.learning_rate,
            'weight_decay': config.model.weight_decay,
            'gradient_accumulation_steps': config.model.gradient_accumulation_steps,
            'dp': vars(config.dp),
        },
        'model_description': model_description,
        'artifacts': {
            'training_report_json': str(output_dir / 'training_report.json'),
            'checkpoint_dir': str(output_dir / 'checkpoint'),
            'generated_samples_json': str(samples_path_json),
            'generated_samples_txt': str(samples_path_txt),
        },
        'final_metrics': run_report['evaluation_report']['splits'],
        'privacy_report': run_report.get('train_report', {}).get('privacy_report'),
    }

    summary_path = output_dir / 'run_summary.json'
    save_json(summary, summary_path)
    return summary_path


def main() -> int:
    """Executa o sanity check centralizado."""
    configure_logging()
    args = parse_args()

    base_config = get_default_config()
    config = apply_overrides(base_config, args)
    set_seed(config.seed)

    condition_name = args.condition
    run_name = build_run_name(condition_name=condition_name, run_name=args.run_name)
    output_dir = resolve_run_output_dir(run_name=run_name, use_persistent_dir=False)

    LOGGER.info(
        'Executando sanity check: condition=%s output_dir=%s fp16=%s dp_enabled=%s',
        condition_name,
        output_dir,
        config.model.fp16,
        config.dp.enabled,
    )

    device = get_device()
    tokenizer = load_tokenizer(config.model)
    model = load_causal_lm_model(config.model, device=device)

    model_description = describe_model(model)
    LOGGER.info('Modelo carregado: %s', model_description)

    run_report = train_on_global_condition(
        config=config,
        model=model,
        tokenizer=tokenizer,
        condition_name=condition_name,
        device=device,
    )

    save_training_run_artifacts(
        model=model,
        tokenizer=tokenizer,
        run_report=run_report,
        output_dir=output_dir,
    )

    samples = generate_from_prompts(
        config=config,
        model=model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        device=device,
    )
    samples_path_json = save_generated_samples(samples, output_dir / 'generated_samples.json')
    samples_path_txt = save_generated_samples_txt(samples, output_dir / 'generated_samples.txt')

    summary_path = save_run_summary(
        output_dir=output_dir,
        config=config,
        condition_name=condition_name,
        model_description=model_description,
        run_report=run_report,
        samples_path_json=samples_path_json,
        samples_path_txt=samples_path_txt,
    )

    LOGGER.info('Sanity check concluído. Resumo salvo em %s', summary_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
