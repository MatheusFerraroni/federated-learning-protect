"""
Script de avaliação de vazamento por prompting em checkpoints treinados.

Responsabilidades principais:
- Carregar checkpoint salvo de sanity check ou treino federado.
- Resolver split de treino da condição avaliada.
- Resolver split de referência para medir leakage contra segredos originais.
- Executar ataque por prompting e salvar relatório estruturado.
- Normalizar e persistir métricas com foco em exposição real de entidades
  sensíveis, incluindo tamanho e cobertura do catálogo sensível.

Como este arquivo se encaixa no projeto:
- É o ponto de entrada de avaliação de leakage do experimento.
- Suporta condições limpas e condições com atacante.
- Consolida artefatos prontos para análise comparativa entre cenários.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from configs.experiment_config import ExperimentConfig, get_default_config
from configs.paths import GLOBAL_SPLITS_DIR, RESULTS_DIR
from src.model.attack import run_attack_evaluation
from src.model.model_utils import get_device
from src.utils.io import save_json
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.seed import set_seed

LOGGER = get_logger(__name__)

VALID_CONDITIONS = [
    'raw',
    'semantic_substitution',
    'attack_raw',
    'attack_semantic_substitution',
]


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando.

    Args:
        Nenhum.

    Returns:
        Namespace com parâmetros de execução.

    Raises:
        Não se aplica.
    """
    parser = argparse.ArgumentParser(
        description='Executa avaliação de vazamento por prompting sobre um checkpoint treinado.'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help='Diretório do checkpoint salvo pelo sanity check ou FL.',
    )
    parser.add_argument(
        '--condition',
        type=str,
        default='raw',
        choices=VALID_CONDITIONS,
        help='Condição experimental usada para resolver o train split padrão.',
    )
    parser.add_argument(
        '--train-path',
        type=str,
        default=None,
        help='Opcional. Caminho explícito para o JSONL de treino usado para montar os prompts.',
    )
    parser.add_argument(
        '--reference-train-path',
        type=str,
        default=None,
        help='Opcional. Caminho explícito para o JSONL de referência contra o qual o leakage será medido.',
    )
    parser.add_argument(
        '--reference-condition',
        type=str,
        default=None,
        choices=VALID_CONDITIONS,
        help=(
            'Condição de referência. Se omitido: raw para semantic_substitution e '
            'attack_semantic_substitution; senão a própria condição.'
        ),
    )
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=None,
        help='Override para config.attack.num_prompts.',
    )
    parser.add_argument(
        '--max-generation-tokens',
        type=int,
        default=None,
        help='Override para config.attack.max_generation_tokens.',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Override para config.attack.generation_temperature.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Override para config.attack.generation_top_k.',
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        help='Override para config.attack.generation_top_p.',
    )
    parser.add_argument(
        '--num-return-sequences',
        type=int,
        default=None,
        help='Override para config.attack.generation_num_return_sequences.',
    )
    parser.add_argument(
        '--partial-match-min-ratio',
        type=float,
        default=None,
        help='Override para config.attack.partial_match_min_ratio.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Diretório de saída. Se omitido, será criado dentro de results/.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed opcional.',
    )
    return parser.parse_args()


def apply_overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    """Aplica overrides de CLI na configuração do experimento.

    Args:
        config: Configuração base.
        args: Argumentos parseados.

    Returns:
        Configuração atualizada.

    Raises:
        Não se aplica.
    """
    updated = deepcopy(config)

    if args.num_prompts is not None:
        updated.attack.num_prompts = args.num_prompts
    if args.max_generation_tokens is not None:
        updated.attack.max_generation_tokens = args.max_generation_tokens
    if args.temperature is not None:
        updated.attack.generation_temperature = args.temperature
    if args.top_k is not None:
        updated.attack.generation_top_k = args.top_k
    if args.top_p is not None:
        updated.attack.generation_top_p = args.top_p
    if args.num_return_sequences is not None:
        updated.attack.generation_num_return_sequences = args.num_return_sequences
    if args.partial_match_min_ratio is not None:
        updated.attack.partial_match_min_ratio = args.partial_match_min_ratio
    if args.seed is not None:
        updated.seed = args.seed
        updated.attack.prompt_seed = args.seed

    return updated


def resolve_default_train_path(condition_name: str) -> Path:
    """Resolve o train split padrão de uma condição.

    Args:
        condition_name: Nome da condição experimental.

    Returns:
        Caminho do JSONL de treino agregado global.

    Raises:
        FileNotFoundError: Se o arquivo esperado não existir.
    """
    path = GLOBAL_SPLITS_DIR / condition_name / 'train.jsonl'
    if not path.exists():
        raise FileNotFoundError(
            f'Train split não encontrado para condição {condition_name}: {path}'
        )
    return path


def resolve_reference_path(
    condition_name: str,
    explicit_reference_condition: str | None,
    explicit_reference_train_path: str | None,
) -> tuple[Path, str]:
    """Resolve o split de referência usado para medir leakage.

    Args:
        condition_name: Condição sendo avaliada.
        explicit_reference_condition: Condição de referência explícita.
        explicit_reference_train_path: Caminho explícito do split de referência.

    Returns:
        Tupla com caminho do split e rótulo textual da referência.

    Raises:
        FileNotFoundError: Se o caminho explícito ou default não existir.
    """
    if explicit_reference_train_path is not None:
        reference_path = Path(explicit_reference_train_path)
        if not reference_path.exists():
            raise FileNotFoundError(f'Reference train split não encontrado: {reference_path}')
        reference_label = explicit_reference_condition or 'custom_reference'
        return reference_path, reference_label

    if explicit_reference_condition is not None:
        reference_condition = explicit_reference_condition
    elif condition_name in {'semantic_substitution', 'attack_semantic_substitution'}:
        reference_condition = 'raw'
    else:
        reference_condition = condition_name

    reference_path = resolve_default_train_path(reference_condition)
    return reference_path, reference_condition


def resolve_output_dir(
    checkpoint_dir: Path,
    output_dir: str | None,
    condition_name: str,
    reference_label: str,
) -> Path:
    """Resolve o diretório de saída da avaliação.

    Args:
        checkpoint_dir: Diretório do checkpoint carregado.
        output_dir: Diretório explícito opcional.
        condition_name: Condição avaliada.
        reference_label: Rótulo textual da referência.

    Returns:
        Diretório de saída criado.

    Raises:
        OSError: Se a criação falhar.
    """
    if output_dir:
        resolved = Path(output_dir)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    checkpoint_parent = checkpoint_dir.parent.name
    checkpoint_name = checkpoint_dir.name
    checkpoint_grandparent = (
        checkpoint_dir.parent.parent.name
        if checkpoint_dir.parent.parent != checkpoint_dir.parent
        else 'root'
    )

    run_name = (
        f'attack_eval_{condition_name}_ref_{reference_label}_'
        f'{checkpoint_grandparent}_{checkpoint_parent}_{checkpoint_name}'
    )
    resolved = RESULTS_DIR / run_name
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_model_and_tokenizer(
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Carrega tokenizer e modelo a partir de um checkpoint local.

    Args:
        checkpoint_dir: Diretório do checkpoint.
        device: Device alvo.

    Returns:
        Tupla ``(model, tokenizer)`` pronta para inferência.

    Raises:
        FileNotFoundError: Se o checkpoint não existir.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f'Checkpoint não encontrado: {checkpoint_dir}')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    return model, tokenizer


def _safe_dict(value: Any) -> dict[str, Any]:
    """Retorna um dicionário seguro.

    Args:
        value: Valor arbitrário.

    Returns:
        Dicionário se possível, ou vazio.

    Raises:
        Não se aplica.
    """
    if isinstance(value, dict):
        return value
    return {}


def _safe_list(value: Any) -> list[Any]:
    """Retorna uma lista segura.

    Args:
        value: Valor arbitrário.

    Returns:
        Lista se possível, ou lista vazia.

    Raises:
        Não se aplica.
    """
    if isinstance(value, list):
        return value
    return []


def _pick_first(source: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    """Seleciona o primeiro valor não nulo entre várias chaves.

    Args:
        source: Dicionário-fonte.
        keys: Chaves candidatas em ordem de prioridade.

    Returns:
        Primeiro valor não nulo encontrado, ou ``None``.

    Raises:
        Não se aplica.
    """
    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return None


def _normalize_float_or_none(value: Any) -> float | None:
    """Converte um valor para float quando possível.

    Args:
        value: Valor arbitrário.

    Returns:
        Float convertido, ou ``None`` quando não conversível.

    Raises:
        Não se aplica.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_int_or_none(value: Any) -> int | None:
    """Converte um valor para int quando possível.

    Args:
        value: Valor arbitrário.

    Returns:
        Int convertido, ou ``None`` quando não conversível.

    Raises:
        Não se aplica.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compute_catalog_coverage(
    unique_exposed_count: int | None,
    catalog_size: int | None,
) -> float | None:
    """Calcula cobertura do catálogo de forma segura.

    Args:
        unique_exposed_count: Número de entidades únicas expostas.
        catalog_size: Tamanho do catálogo sensível.

    Returns:
        Cobertura entre 0 e 1, ou ``None`` quando não calculável.

    Raises:
        Não se aplica.
    """
    if unique_exposed_count is None or catalog_size is None or catalog_size <= 0:
        return None
    return float(unique_exposed_count) / float(catalog_size)


def _normalize_attack_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    """Normaliza métricas de ataque a partir do relatório completo.

    Args:
        report: Relatório completo retornado por ``run_attack_evaluation``.

    Returns:
        Dicionário com métricas normalizadas, mantendo foco em exposição real
        de entidades sensíveis.

    Raises:
        Não se aplica.
    """
    payload = _safe_dict(report)
    summary = _safe_dict(payload.get('summary'))
    metrics = _safe_dict(payload.get('metrics'))
    aggregate = _safe_dict(payload.get('aggregate'))
    results = _safe_dict(payload.get('results'))
    attack_config = _safe_dict(payload.get('attack_config'))

    candidate_sources: list[Mapping[str, Any]] = [
        summary,
        metrics,
        aggregate,
        results,
        payload,
    ]

    normalized: dict[str, Any] = {}

    float_field_aliases: dict[str, tuple[str, ...]] = {
        # =========================
        # MÉTRICAS PRINCIPAIS
        # =========================
        'reference_sensitive_entity_exposure_rate': ('reference_sensitive_entity_exposure_rate',),
        'train_sensitive_entity_exposure_rate': ('train_sensitive_entity_exposure_rate',),
        'reference_secret_exact_match_rate': ('reference_secret_exact_match_rate',),
        'train_secret_exact_match_rate': ('train_secret_exact_match_rate',),
        'reference_secret_partial_match_rate': ('reference_secret_partial_match_rate',),
        'train_secret_partial_match_rate': ('train_secret_partial_match_rate',),
        # =========================
        # DERIVADAS / AUXILIARES
        # =========================
        'reference_entities_exposed_per_generation': ('reference_entities_exposed_per_generation',),
        'train_entities_exposed_per_generation': ('train_entities_exposed_per_generation',),
        'exact_target_match_rate': (
            'exact_target_match_rate',
            'target_exact_match_rate',
        ),
        'partial_target_match_rate': (
            'partial_target_match_rate',
            'target_partial_match_rate',
        ),
        'structured_entity_generation_rate': ('structured_entity_generation_rate',),
        'canary_recovery_rate': (
            'canary_recovery_rate',
            'canary_recovery',
        ),
        'reference_catalog_coverage_rate': ('reference_catalog_coverage_rate',),
        'train_catalog_coverage_rate': ('train_catalog_coverage_rate',),
        # Aliases legados adicionais úteis
        'exact_match_rate': ('exact_match_rate',),
        'partial_match_rate': ('partial_match_rate',),
    }

    int_field_aliases: dict[str, tuple[str, ...]] = {
        'num_attack_generations': (
            'num_attack_generations',
            'num_generations',
            'generation_count',
            'num_records',
        ),
        'num_prompts': (
            'num_prompts',
            'prompt_count',
        ),
        'reference_exposed_sensitive_entity_count': ('reference_exposed_sensitive_entity_count',),
        'train_exposed_sensitive_entity_count': ('train_exposed_sensitive_entity_count',),
        'reference_unique_exposed_sensitive_entity_count': (
            'reference_unique_exposed_sensitive_entity_count',
        ),
        'train_unique_exposed_sensitive_entity_count': (
            'train_unique_exposed_sensitive_entity_count',
        ),
        'reference_exposed_canary_entity_count': ('reference_exposed_canary_entity_count',),
        'reference_unique_exposed_canary_entity_count': (
            'reference_unique_exposed_canary_entity_count',
        ),
        'reference_sensitive_catalog_size': (
            'reference_sensitive_catalog_size',
            'num_reference_sensitive_catalog_items',
        ),
        'train_sensitive_catalog_size': (
            'train_sensitive_catalog_size',
            'num_train_sensitive_catalog_items',
        ),
        'num_train_sensitive_catalog_items': (
            'num_train_sensitive_catalog_items',
            'train_sensitive_catalog_size',
        ),
        'num_reference_sensitive_catalog_items': (
            'num_reference_sensitive_catalog_items',
            'reference_sensitive_catalog_size',
        ),
        'num_train_records': ('num_train_records',),
        'num_reference_records': ('num_reference_records',),
    }

    for normalized_key, aliases in float_field_aliases.items():
        value: Any = None
        for source in candidate_sources:
            value = _pick_first(source, aliases)
            if value is not None:
                break
        normalized[normalized_key] = _normalize_float_or_none(value)

    for normalized_key, aliases in int_field_aliases.items():
        value = None
        for source in candidate_sources:
            value = _pick_first(source, aliases)
            if value is not None:
                break
        normalized[normalized_key] = _normalize_int_or_none(value)

    if normalized['num_prompts'] is None:
        normalized['num_prompts'] = _normalize_int_or_none(attack_config.get('num_prompts'))

    # =========================
    # FALLBACKS CRÍTICOS
    # =========================
    if normalized.get('reference_sensitive_entity_exposure_rate') is None:
        normalized['reference_sensitive_entity_exposure_rate'] = normalized.get(
            'reference_secret_exact_match_rate'
        )

    if normalized.get('train_sensitive_entity_exposure_rate') is None:
        normalized['train_sensitive_entity_exposure_rate'] = normalized.get(
            'train_secret_exact_match_rate'
        )

    if normalized.get('reference_secret_exact_match_rate') is None:
        normalized['reference_secret_exact_match_rate'] = normalized.get(
            'reference_sensitive_entity_exposure_rate'
        )

    if normalized.get('train_secret_exact_match_rate') is None:
        normalized['train_secret_exact_match_rate'] = normalized.get(
            'train_sensitive_entity_exposure_rate'
        )

    if normalized.get('exact_match_rate') is None:
        normalized['exact_match_rate'] = normalized.get('reference_secret_exact_match_rate')

    if normalized.get('partial_match_rate') is None:
        normalized['partial_match_rate'] = normalized.get('reference_secret_partial_match_rate')

    if normalized['reference_catalog_coverage_rate'] is None:
        normalized['reference_catalog_coverage_rate'] = _compute_catalog_coverage(
            unique_exposed_count=normalized['reference_unique_exposed_sensitive_entity_count'],
            catalog_size=normalized['reference_sensitive_catalog_size'],
        )

    if normalized['train_catalog_coverage_rate'] is None:
        normalized['train_catalog_coverage_rate'] = _compute_catalog_coverage(
            unique_exposed_count=normalized['train_unique_exposed_sensitive_entity_count'],
            catalog_size=normalized['train_sensitive_catalog_size'],
        )

    normalized['legacy_metrics'] = {
        'exact_target_match_rate': normalized['exact_target_match_rate'],
        'partial_target_match_rate': normalized['partial_target_match_rate'],
        'structured_entity_generation_rate': normalized['structured_entity_generation_rate'],
        'reference_secret_exact_match_rate': normalized['reference_secret_exact_match_rate'],
        'reference_secret_partial_match_rate': normalized['reference_secret_partial_match_rate'],
        'train_secret_exact_match_rate': normalized['train_secret_exact_match_rate'],
        'train_secret_partial_match_rate': normalized['train_secret_partial_match_rate'],
        'exact_match_rate': normalized['exact_match_rate'],
        'partial_match_rate': normalized['partial_match_rate'],
    }

    normalized['headline_metrics'] = {
        'reference_sensitive_entity_exposure_rate': normalized[
            'reference_sensitive_entity_exposure_rate'
        ],
        'reference_exposed_sensitive_entity_count': normalized[
            'reference_exposed_sensitive_entity_count'
        ],
        'reference_unique_exposed_sensitive_entity_count': normalized[
            'reference_unique_exposed_sensitive_entity_count'
        ],
        'reference_entities_exposed_per_generation': normalized[
            'reference_entities_exposed_per_generation'
        ],
        'reference_catalog_coverage_rate': normalized['reference_catalog_coverage_rate'],
        'canary_recovery_rate': normalized['canary_recovery_rate'],
    }

    return normalized


def save_execution_summary(
    output_dir: Path,
    checkpoint_dir: Path,
    train_path: Path,
    reference_train_path: Path,
    condition_name: str,
    reference_label: str,
    report: dict[str, Any],
) -> Path:
    """Salva resumo estruturado da avaliação de ataque.

    Args:
        output_dir: Diretório de saída.
        checkpoint_dir: Diretório do checkpoint avaliado.
        train_path: Split usado para montar prompts.
        reference_train_path: Split usado como referência de leakage.
        condition_name: Condição avaliada.
        reference_label: Nome textual da referência.
        report: Relatório retornado por ``run_attack_evaluation``.

    Returns:
        Caminho do summary salvo.

    Raises:
        OSError: Se a gravação falhar.
    """
    normalized_metrics = _normalize_attack_metrics(report)
    report_payload = _safe_dict(report)

    train_secret_catalog = _safe_list(report_payload.get('train_secret_catalog'))
    reference_secret_catalog = _safe_list(report_payload.get('reference_secret_catalog'))

    if normalized_metrics.get('train_sensitive_catalog_size') is None:
        normalized_metrics['train_sensitive_catalog_size'] = len(train_secret_catalog)
    if normalized_metrics.get('reference_sensitive_catalog_size') is None:
        normalized_metrics['reference_sensitive_catalog_size'] = len(reference_secret_catalog)

    if normalized_metrics.get('num_train_sensitive_catalog_items') is None:
        normalized_metrics['num_train_sensitive_catalog_items'] = normalized_metrics.get(
            'train_sensitive_catalog_size'
        )
    if normalized_metrics.get('num_reference_sensitive_catalog_items') is None:
        normalized_metrics['num_reference_sensitive_catalog_items'] = normalized_metrics.get(
            'reference_sensitive_catalog_size'
        )

    normalized_metrics['train_catalog_coverage_rate'] = (
        normalized_metrics['train_catalog_coverage_rate']
        if normalized_metrics['train_catalog_coverage_rate'] is not None
        else _compute_catalog_coverage(
            unique_exposed_count=normalized_metrics.get(
                'train_unique_exposed_sensitive_entity_count'
            ),
            catalog_size=normalized_metrics.get('train_sensitive_catalog_size'),
        )
    )
    normalized_metrics['reference_catalog_coverage_rate'] = (
        normalized_metrics['reference_catalog_coverage_rate']
        if normalized_metrics['reference_catalog_coverage_rate'] is not None
        else _compute_catalog_coverage(
            unique_exposed_count=normalized_metrics.get(
                'reference_unique_exposed_sensitive_entity_count'
            ),
            catalog_size=normalized_metrics.get('reference_sensitive_catalog_size'),
        )
    )

    normalized_metrics['headline_metrics'] = {
        'reference_sensitive_entity_exposure_rate': normalized_metrics.get(
            'reference_sensitive_entity_exposure_rate'
        ),
        'reference_exposed_sensitive_entity_count': normalized_metrics.get(
            'reference_exposed_sensitive_entity_count'
        ),
        'reference_unique_exposed_sensitive_entity_count': normalized_metrics.get(
            'reference_unique_exposed_sensitive_entity_count'
        ),
        'reference_entities_exposed_per_generation': normalized_metrics.get(
            'reference_entities_exposed_per_generation'
        ),
        'reference_catalog_coverage_rate': normalized_metrics.get(
            'reference_catalog_coverage_rate'
        ),
        'canary_recovery_rate': normalized_metrics.get('canary_recovery_rate'),
    }

    summary = {
        'checkpoint_dir': str(checkpoint_dir),
        'train_path': str(train_path),
        'reference_train_path': str(reference_train_path),
        'condition': condition_name,
        'reference_label': reference_label,
        'summary': normalized_metrics,
        'metrics': normalized_metrics,
        'headline_metrics': normalized_metrics['headline_metrics'],
        'catalog': {
            'train_sensitive_catalog_size': normalized_metrics.get('train_sensitive_catalog_size'),
            'reference_sensitive_catalog_size': normalized_metrics.get(
                'reference_sensitive_catalog_size'
            ),
            'train_secret_catalog': train_secret_catalog,
            'reference_secret_catalog': reference_secret_catalog,
        },
        'artifacts': {
            'attack_report_json': str(output_dir / 'attack_report.json'),
            'attack_generations_jsonl': str(output_dir / 'attack_generations.jsonl'),
            'attack_secret_index_json': str(output_dir / 'attack_secret_index.json'),
        },
        'full_report': report_payload,
    }

    summary_path = output_dir / 'attack_summary.json'
    save_json(summary, summary_path)
    return summary_path


def main() -> int:
    """Executa a avaliação de vazamento por prompting.

    Args:
        Nenhum.

    Returns:
        Código de saída do processo.

    Raises:
        FileNotFoundError: Se checkpoints ou splits não forem encontrados.
        RuntimeError: Se a avaliação falhar.
    """
    configure_logging()
    args = parse_args()

    config = apply_overrides(get_default_config(), args)
    set_seed(config.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    train_path = (
        Path(args.train_path)
        if args.train_path is not None
        else resolve_default_train_path(args.condition)
    )
    reference_train_path, reference_label = resolve_reference_path(
        condition_name=args.condition,
        explicit_reference_condition=args.reference_condition,
        explicit_reference_train_path=args.reference_train_path,
    )
    output_dir = resolve_output_dir(
        checkpoint_dir=checkpoint_dir,
        output_dir=args.output_dir,
        condition_name=args.condition,
        reference_label=reference_label,
    )

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir=checkpoint_dir, device=device)

    LOGGER.info(
        'Executando attack eval: condition=%s reference=%s checkpoint=%s output_dir=%s',
        args.condition,
        reference_label,
        checkpoint_dir,
        output_dir,
    )

    report = run_attack_evaluation(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_records_or_path=train_path,
        reference_records_or_path=reference_train_path,
        reference_label=reference_label,
        output_dir=output_dir,
        device=device,
    )

    summary_path = save_execution_summary(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        train_path=train_path,
        reference_train_path=reference_train_path,
        condition_name=args.condition,
        reference_label=reference_label,
        report=report,
    )

    LOGGER.info('Attack evaluation concluída. Summary salvo em %s', summary_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
