"""Coordena o treinamento federado manual com FedAvg.

Responsabilidades principais:
- Selecionar clientes por rodada.
- Garantir suporte explícito a clientes honestos e maliciosos.
- Coletar updates locais e agregá-los via FedAvg.
- Avaliar o modelo global em uma condição limpa de referência.
- Salvar métricas, checkpoints e amostras por rodada.
- Expor um entry point CLI executável via `python -m src.fl.run_federated`.
- Permitir baseline opcional com DP-SGD / Opacus em todos os clientes.

Como este arquivo se encaixa no projeto:
- É o núcleo do treinamento federado do lado do servidor.
- Implementa o cenário de ameaça em que o servidor agrega sem acessar
  os dados brutos dos clientes.
- Serve como ponto de entrada para os scripts em `experiments/`.
"""

from __future__ import annotations

import argparse
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from configs.experiment_config import ExperimentConfig, get_default_config
from configs.paths import RESULTS_DIR
from src.fl.client import infer_client_role, list_available_client_ids, train_client_one_round
from src.fl.fedavg import fedavg_aggregate, summarize_client_weights
from src.model.evaluate import evaluate_global_condition
from src.model.generate_samples import (
    DEFAULT_PROMPTS,
    generate_from_prompts,
    save_generated_samples,
)
from src.model.model_utils import (
    get_device,
    load_causal_lm_model,
    load_tokenizer,
    save_model_checkpoint,
)
from src.utils.io import save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)

VALID_CONDITIONS = {
    'raw',
    'semantic_substitution',
    'attack_raw',
    'attack_semantic_substitution',
}


@dataclass
class FederatedRoundResult:
    """Resultado agregado de uma rodada federada."""

    round_index: int
    selected_clients: list[str]
    client_weights: list[dict[str, Any]]
    client_metrics: list[dict[str, Any]]
    global_evaluation: dict[str, Any]
    generated_samples_path: str | None
    checkpoint_dir: str


def resolve_federated_output_dir(run_name: str) -> Path:
    """Cria e resolve o diretório de saída do treino federado."""
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def split_client_ids_by_role(client_ids: list[str]) -> tuple[list[str], list[str]]:
    """Separa clientes honestos e maliciosos pelos IDs."""
    honest_ids = sorted(
        client_id for client_id in client_ids if infer_client_role(client_id) == 'honest'
    )
    malicious_ids = sorted(
        client_id for client_id in client_ids if infer_client_role(client_id) == 'malicious'
    )
    return honest_ids, malicious_ids


def _select_clients_simple(
    available_client_ids: list[str],
    clients_per_round: int,
    round_index: int,
    base_seed: int,
) -> list[str]:
    """Seleciona clientes sem distinguir papéis."""
    if clients_per_round <= 0:
        raise ValueError('clients_per_round deve ser > 0.')
    if clients_per_round > len(available_client_ids):
        raise ValueError(
            f'clients_per_round={clients_per_round} é maior que o número de clientes disponíveis={len(available_client_ids)}.'
        )

    rng = random.Random(base_seed + round_index)
    return sorted(rng.sample(available_client_ids, k=clients_per_round))


def select_clients_for_round(
    available_client_ids: list[str],
    clients_per_round: int,
    round_index: int,
    config: ExperimentConfig,
) -> list[str]:
    """Seleciona clientes para uma rodada, com suporte a atacantes."""
    honest_ids, malicious_ids = split_client_ids_by_role(available_client_ids)

    if (
        not config.federated.sample_malicious_separately
        or not malicious_ids
        or not config.malicious.enabled
    ):
        return _select_clients_simple(
            available_client_ids=available_client_ids,
            clients_per_round=clients_per_round,
            round_index=round_index,
            base_seed=config.seed,
        )

    if clients_per_round <= 0:
        raise ValueError('clients_per_round deve ser > 0.')
    if clients_per_round > len(available_client_ids):
        raise ValueError(
            f'clients_per_round={clients_per_round} é maior que o número de clientes disponíveis={len(available_client_ids)}.'
        )

    desired_by_fraction = int(
        math.ceil(clients_per_round * config.malicious.fraction_malicious_selected_per_round)
    )
    malicious_quota = max(config.federated.min_malicious_clients_per_round, desired_by_fraction)
    malicious_quota = min(malicious_quota, config.federated.max_malicious_clients_per_round)
    malicious_quota = min(malicious_quota, len(malicious_ids))
    malicious_quota = min(malicious_quota, clients_per_round)

    honest_quota = clients_per_round - malicious_quota
    if honest_quota > len(honest_ids):
        missing = honest_quota - len(honest_ids)
        malicious_quota = min(malicious_quota + missing, len(malicious_ids), clients_per_round)
        honest_quota = clients_per_round - malicious_quota

    rng = random.Random(config.seed + round_index * 97)

    selected_honest = rng.sample(honest_ids, k=honest_quota) if honest_quota > 0 else []
    selected_malicious = rng.sample(malicious_ids, k=malicious_quota) if malicious_quota > 0 else []

    selected = sorted(selected_honest + selected_malicious)
    if len(selected) != clients_per_round:
        raise ValueError(
            f'Seleção inválida de clientes. Esperado={clients_per_round}, obtido={len(selected)}.'
        )

    return selected


def _build_client_metrics_payload(client_result: Any) -> dict[str, Any]:
    """Converte resultado local em payload serializável."""
    return {
        'client_id': client_result.client_id,
        'condition_name': client_result.condition_name,
        'client_role': client_result.client_role,
        'num_examples': client_result.num_examples,
        'train_metrics': client_result.train_metrics,
        'val_metrics': client_result.val_metrics,
        'privacy_metrics': client_result.privacy_metrics,
    }


def _save_round_artifacts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    round_index: int,
    round_payload: dict[str, Any],
    save_generated_text: bool,
    generated_samples: list[dict[str, Any]] | None,
) -> dict[str, str | None]:
    """Salva métricas, checkpoint e amostras de uma rodada."""
    round_dir = output_dir / f'round_{round_index:03d}'
    round_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = round_dir / 'round_metrics.json'
    save_json(round_payload, metrics_path)

    checkpoint_dir = round_dir / 'checkpoint'
    save_model_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=checkpoint_dir,
        extra_state=round_payload,
    )

    generated_samples_path: str | None = None
    if save_generated_text and generated_samples is not None:
        samples_path = round_dir / 'generated_samples.json'
        save_generated_samples(generated_samples, samples_path)
        generated_samples_path = str(samples_path)

    return {
        'round_dir': str(round_dir),
        'metrics_path': str(metrics_path),
        'checkpoint_dir': str(checkpoint_dir),
        'generated_samples_path': generated_samples_path,
    }


def run_federated_training(
    config: ExperimentConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    condition_name: str,
    run_name: str,
    device: torch.device | None = None,
    save_generated_text_each_round: bool = True,
    evaluation_condition_name: str | None = None,
) -> dict[str, Any]:
    """Executa o loop federado manual com FedAvg."""
    resolved_device = device or get_device()
    output_dir = resolve_federated_output_dir(run_name)
    eval_condition = evaluation_condition_name or condition_name

    available_client_ids = list_available_client_ids(condition_name)
    clients_per_round = int(config.federated.clients_per_round)
    num_rounds = int(config.federated.num_rounds)

    if num_rounds <= 0:
        raise ValueError('num_rounds deve ser > 0.')
    if clients_per_round <= 0:
        raise ValueError('clients_per_round deve ser > 0.')
    if clients_per_round > len(available_client_ids):
        raise ValueError(
            f'clients_per_round={clients_per_round} é maior que clientes disponíveis={len(available_client_ids)}.'
        )

    global_model = deepcopy(model)
    global_model.to(resolved_device)

    history: list[dict[str, Any]] = []

    honest_ids, malicious_ids = split_client_ids_by_role(available_client_ids)
    LOGGER.info(
        (
            'Iniciando treino federado: train_condition=%s eval_condition=%s '
            'num_rounds=%d clients_per_round=%d available_clients=%d honest=%d '
            'malicious=%d output_dir=%s dp_enabled=%s'
        ),
        condition_name,
        eval_condition,
        num_rounds,
        clients_per_round,
        len(available_client_ids),
        len(honest_ids),
        len(malicious_ids),
        output_dir,
        config.dp.enabled,
    )

    for round_index in tqdm(range(num_rounds), desc='Rounds', position=0):
        selected_clients = select_clients_for_round(
            available_client_ids=available_client_ids,
            clients_per_round=clients_per_round,
            round_index=round_index,
            config=config,
        )

        LOGGER.info(
            'Rodada federada %d/%d | clientes selecionados=%s',
            round_index + 1,
            num_rounds,
            selected_clients,
        )

        client_results = []
        client_updates = []

        for client_id in selected_clients:
            result = train_client_one_round(
                config=config,
                global_model=global_model,
                tokenizer=tokenizer,
                condition_name=condition_name,
                client_id=client_id,
                device=resolved_device,
            )
            client_results.append(result)
            client_updates.append(
                {
                    'client_id': result.client_id,
                    'num_examples': result.num_examples,
                    'state_dict': result.state_dict,
                }
            )

        aggregated_state = fedavg_aggregate(client_updates)
        global_model.load_state_dict(aggregated_state, strict=True)
        global_model.to(resolved_device)

        global_evaluation = evaluate_global_condition(
            config=config,
            model=global_model,
            tokenizer=tokenizer,
            condition_name=eval_condition,
            split_names=['train', 'val', 'test', 'domain_test'],
            device=resolved_device,
        )

        generated_samples = None
        if save_generated_text_each_round:
            generated_samples = generate_from_prompts(
                config=config,
                model=global_model,
                tokenizer=tokenizer,
                prompts=DEFAULT_PROMPTS,
                device=resolved_device,
            )

        round_payload = {
            'round_index': round_index,
            'train_condition': condition_name,
            'evaluation_condition': eval_condition,
            'selected_clients': selected_clients,
            'client_weights': summarize_client_weights(client_updates),
            'client_metrics': [_build_client_metrics_payload(item) for item in client_results],
            'global_evaluation': global_evaluation,
            'dp_config': asdict(config.dp),
        }

        artifact_paths = _save_round_artifacts(
            model=global_model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            round_index=round_index,
            round_payload=round_payload,
            save_generated_text=save_generated_text_each_round,
            generated_samples=generated_samples,
        )

        round_result = FederatedRoundResult(
            round_index=round_index,
            selected_clients=selected_clients,
            client_weights=round_payload['client_weights'],
            client_metrics=round_payload['client_metrics'],
            global_evaluation=global_evaluation,
            generated_samples_path=artifact_paths['generated_samples_path'],
            checkpoint_dir=str(artifact_paths['checkpoint_dir']),
        )
        history.append(
            {
                'round_index': round_result.round_index,
                'selected_clients': round_result.selected_clients,
                'client_weights': round_result.client_weights,
                'client_metrics': round_result.client_metrics,
                'global_evaluation': round_result.global_evaluation,
                'generated_samples_path': round_result.generated_samples_path,
                'checkpoint_dir': round_result.checkpoint_dir,
            }
        )

    final_report = {
        'run_name': run_name,
        'train_condition': condition_name,
        'evaluation_condition': eval_condition,
        'num_rounds': num_rounds,
        'clients_per_round': clients_per_round,
        'available_clients': available_client_ids,
        'dp_config': asdict(config.dp),
        'history': history,
    }

    save_json(final_report, output_dir / 'federated_report.json')
    LOGGER.info(
        'Treino federado concluído. Relatório salvo em %s', output_dir / 'federated_report.json'
    )
    return final_report


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando do treino federado."""
    parser = argparse.ArgumentParser(description='Executa treino federado manual com FedAvg.')
    parser.add_argument(
        '--condition',
        type=str,
        default='raw',
        choices=sorted(VALID_CONDITIONS),
        help='Condição federada usada para os clientes locais.',
    )
    parser.add_argument(
        '--evaluation-condition',
        type=str,
        default=None,
        choices=sorted(VALID_CONDITIONS),
        help='Condição usada na avaliação global. Se omitida, será inferida.',
    )
    parser.add_argument(
        '--rounds', type=int, default=None, help='Override para config.federated.num_rounds.'
    )
    parser.add_argument(
        '--clients-per-round',
        type=int,
        default=None,
        help='Override para config.federated.clients_per_round.',
    )
    parser.add_argument(
        '--local-epochs',
        type=int,
        default=None,
        help='Override para config.federated.local_epochs.',
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
    parser.add_argument(
        '--fp16', action='store_true', help='Ativa AMP em CUDA durante o treino local.'
    )
    parser.add_argument(
        '--num-workers', type=int, default=None, help='Override para config.model.num_workers.'
    )
    parser.add_argument('--seed', type=int, default=None, help='Seed global.')
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Nome explícito da execução. Se omitido, será gerado automaticamente.',
    )
    parser.add_argument(
        '--disable-generated-samples',
        action='store_true',
        help='Se definido, não salva amostras geradas por rodada.',
    )
    parser.add_argument(
        '--dp-enabled',
        action='store_true',
        help='Ativa baseline DP-SGD com Opacus nos clientes selecionados.',
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
    """Aplica overrides de CLI na configuração do experimento."""
    updated = deepcopy(config)

    if args.rounds is not None:
        updated.federated.num_rounds = int(args.rounds)
    if args.clients_per_round is not None:
        updated.federated.clients_per_round = int(args.clients_per_round)
    if args.local_epochs is not None:
        updated.federated.local_epochs = int(args.local_epochs)

    if args.batch_size is not None:
        updated.model.batch_size = int(args.batch_size)
    if args.eval_batch_size is not None:
        updated.model.eval_batch_size = int(args.eval_batch_size)
    if args.max_length is not None:
        updated.model.max_length = int(args.max_length)
    if args.learning_rate is not None:
        updated.model.learning_rate = float(args.learning_rate)
    if args.fp16:
        updated.model.fp16 = True
    if args.num_workers is not None:
        updated.model.num_workers = int(args.num_workers)

    if args.seed is not None:
        updated.seed = int(args.seed)
        updated.attack.prompt_seed = int(args.seed)

    if args.dp_enabled:
        updated.dp.enabled = True
    if args.dp_noise_multiplier is not None:
        updated.dp.noise_multiplier = float(args.dp_noise_multiplier)
    if args.dp_max_grad_norm is not None:
        updated.dp.max_grad_norm = float(args.dp_max_grad_norm)
    if args.dp_target_epsilon is not None:
        updated.dp.target_epsilon = float(args.dp_target_epsilon)
    if args.dp_target_delta is not None:
        updated.dp.target_delta = float(args.dp_target_delta)
    if args.dp_max_physical_batch_size is not None:
        updated.dp.max_physical_batch_size = int(args.dp_max_physical_batch_size)
    if args.dp_num_trainable_transformer_blocks is not None:
        updated.dp.num_trainable_transformer_blocks = int(args.dp_num_trainable_transformer_blocks)
    if args.dp_train_embeddings:
        updated.dp.freeze_embeddings = False
    if args.dp_disable_poisson_sampling:
        updated.dp.poisson_sampling = False

    return updated


def resolve_default_evaluation_condition(condition_name: str) -> str:
    """Infere a condição padrão de avaliação global."""
    if condition_name == 'raw':
        return 'raw'
    if condition_name == 'semantic_substitution':
        return 'semantic_substitution'
    if condition_name == 'attack_raw':
        return 'raw'
    if condition_name == 'attack_semantic_substitution':
        return 'raw'
    raise ValueError(f'Condição inválida: {condition_name}')


def build_run_name(condition_name: str, explicit_run_name: str | None = None) -> str:
    """Resolve o nome da execução federada."""
    if explicit_run_name:
        return explicit_run_name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'federated_{condition_name}_{timestamp}'


def main() -> int:
    """Executa o treino federado como entry point CLI."""
    configure_logging()
    args = parse_args()

    config = apply_overrides(get_default_config(), args)
    set_seed(config.seed)

    condition_name = args.condition
    evaluation_condition_name = (
        args.evaluation_condition
        if args.evaluation_condition is not None
        else resolve_default_evaluation_condition(condition_name)
    )
    run_name = build_run_name(condition_name=condition_name, explicit_run_name=args.run_name)

    LOGGER.info(
        (
            'Preparando treino federado: run_name=%s train_condition=%s '
            'evaluation_condition=%s rounds=%d clients_per_round=%d '
            'local_epochs=%d batch_size=%d eval_batch_size=%d max_length=%d '
            'learning_rate=%s num_workers=%d seed=%d fp16=%s dp_enabled=%s results_dir=%s'
        ),
        run_name,
        condition_name,
        evaluation_condition_name,
        config.federated.num_rounds,
        config.federated.clients_per_round,
        config.federated.local_epochs,
        config.model.batch_size,
        config.model.eval_batch_size,
        config.model.max_length,
        config.model.learning_rate,
        config.model.num_workers,
        config.seed,
        config.model.fp16,
        config.dp.enabled,
        RESULTS_DIR,
    )

    device = get_device()
    tokenizer = load_tokenizer(config.model)
    model = load_causal_lm_model(config.model, device=device)

    report = run_federated_training(
        config=config,
        model=model,
        tokenizer=tokenizer,
        condition_name=condition_name,
        run_name=run_name,
        device=device,
        save_generated_text_each_round=not args.disable_generated_samples,
        evaluation_condition_name=evaluation_condition_name,
    )

    LOGGER.info(
        'Execução federada finalizada com sucesso. run_name=%s report_path=%s',
        run_name,
        RESULTS_DIR / run_name / 'federated_report.json',
    )
    LOGGER.debug('Resumo do relatório federado: %s', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
