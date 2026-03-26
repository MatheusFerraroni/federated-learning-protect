"""Consolida, normaliza e plota métricas finais dos pipelines experimentais.

Responsabilidades principais:
- Extrair métricas de utilidade e leakage a partir dos summaries consolidados.
- Compatibilizar nomes legados e novos das métricas após mudanças no pipeline.
- Gerar CSVs, JSON/TXT de comparação e gráficos PNG consistentes.
- Servir de camada robusta para o relatório final da tese.

Como este arquivo se encaixa no projeto:
- É consumido por ``experiments.run_results_report``.
- Opera sobre os summaries produzidos pelos pipelines por condição.
- Garante que mudanças no schema de métricas de ataque não quebrem o relatório.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

from src.utils.io import save_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

DEFAULT_UTILITY_METRICS = (
    'sanity_test_loss',
    'sanity_test_perplexity',
    'federated_test_loss',
    'federated_test_perplexity',
)

DEFAULT_LEAKAGE_METRICS = (
    'sanity_reference_secret_exact_match_rate',
    'sanity_reference_secret_partial_match_rate',
    'sanity_structured_entity_generation_rate',
    'sanity_canary_recovery_rate',
    'federated_reference_secret_exact_match_rate',
    'federated_reference_secret_partial_match_rate',
    'federated_structured_entity_generation_rate',
    'federated_canary_recovery_rate',
)

DISPLAY_NAME_BY_CONDITION = {
    'no_attacker': 'Sem atacante, sem defesa',
    'raw': 'Com atacante, sem defesa',
    'attack_raw': 'Com atacante, sem defesa',
    'semantic_substitution': 'Com atacante + substituição semântica',
    'attack_semantic_substitution': 'Com atacante + substituição semântica',
    'dp': 'Com atacante + DP-SGD / Opacus',
    'attack_raw_dp': 'Com atacante + DP-SGD / Opacus',
}

PLOT_LABEL_BY_METRIC = {
    'sanity_test_loss': 'Sanity',
    'federated_test_loss': 'Federado',
    'sanity_test_perplexity': 'Sanity',
    'federated_test_perplexity': 'Federado',
    'sanity_exact_match_rate': 'Sanity',
    'federated_exact_match_rate': 'Federado',
    'sanity_partial_match_rate': 'Sanity',
    'federated_partial_match_rate': 'Federado',
    'sanity_reference_secret_exact_match_rate': 'Sanity',
    'federated_reference_secret_exact_match_rate': 'Federado',
    'sanity_reference_secret_partial_match_rate': 'Sanity',
    'federated_reference_secret_partial_match_rate': 'Federado',
    'sanity_target_exact_match_rate': 'Sanity',
    'federated_target_exact_match_rate': 'Federado',
    'sanity_structured_entity_generation_rate': 'Sanity',
    'federated_structured_entity_generation_rate': 'Federado',
    'sanity_canary_recovery_rate': 'Sanity',
    'federated_canary_recovery_rate': 'Federado',
}


def ensure_directory(path: str | Path) -> Path:
    """Cria um diretório quando necessário.

    Args:
        path: Caminho a garantir.

    Returns:
        Caminho resolvido como ``Path``.

    Raises:
        OSError: Se a criação do diretório falhar.
    """
    resolved_path = Path(path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def _safe_float(value: Any) -> float | None:
    """Converte valores arbitrários para ``float`` seguro.

    Args:
        value: Valor arbitrário.

    Returns:
        ``float`` válido ou ``None``.

    Raises:
        Não se aplica.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted) or math.isinf(converted):
        return None
    return converted


def _safe_dict(value: Any) -> dict[str, Any]:
    """Garante um dicionário seguro.

    Args:
        value: Valor arbitrário.

    Returns:
        Dicionário se ``value`` já for ``dict``; caso contrário, ``{}``.

    Raises:
        Não se aplica.
    """
    if isinstance(value, dict):
        return value
    return {}


def _get_nested(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    """Recupera um valor em profundidade com tolerância a tipos inválidos.

    Args:
        mapping: Mapeamento de origem.
        *keys: Caminho de chaves.

    Returns:
        Valor encontrado ou ``None``.

    Raises:
        Não se aplica.
    """
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _pick_first_float(
    mapping: Mapping[str, Any] | None,
    candidate_keys: Sequence[str],
) -> float | None:
    """Seleciona o primeiro ``float`` válido entre chaves candidatas.

    Args:
        mapping: Dicionário de origem.
        candidate_keys: Chaves em ordem de prioridade.

    Returns:
        Primeiro valor numérico válido encontrado, ou ``None``.

    Raises:
        Não se aplica.
    """
    payload = _safe_dict(mapping)
    for key in candidate_keys:
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    return None


def _extract_split_metrics_from_payload(
    payload: Mapping[str, Any] | None,
) -> dict[str, float | None]:
    """Extrai loss e perplexity de um payload de avaliação.

    Args:
        payload: Payload candidato.

    Returns:
        Dicionário com ``loss`` e ``perplexity``.

    Raises:
        Não se aplica.
    """
    safe_payload = _safe_dict(payload)
    return {
        'loss': _pick_first_float(
            safe_payload,
            ('loss', 'test_loss', 'eval_loss'),
        ),
        'perplexity': _pick_first_float(
            safe_payload,
            ('perplexity', 'test_perplexity', 'eval_perplexity', 'ppl'),
        ),
    }


def _extract_split_metrics(
    section: Mapping[str, Any] | None,
    split_name: str = 'test',
) -> dict[str, float | None]:
    """Extrai métricas de um split considerando diferentes layouts.

    Args:
        section: Bloco do summary.
        split_name: Nome do split desejado.

    Returns:
        Dicionário com ``loss`` e ``perplexity``.

    Raises:
        Não se aplica.
    """
    payload = _safe_dict(section)

    candidate_locations: list[Mapping[str, Any]] = [
        _safe_dict(payload.get(split_name)),
        _safe_dict(_get_nested(payload, 'evaluation_report', split_name)),
        _safe_dict(_get_nested(payload, 'final_global_evaluation', split_name)),
        payload,
    ]

    for candidate in candidate_locations:
        metrics = _extract_split_metrics_from_payload(candidate)
        if metrics['loss'] is not None or metrics['perplexity'] is not None:
            return metrics

    return {'loss': None, 'perplexity': None}


def _extract_attack_summary(attack_report: Mapping[str, Any] | None) -> dict[str, float | None]:
    """Extrai métricas de ataque com compatibilidade entre schemas.

    Args:
        attack_report: Resumo do ataque em qualquer uma das versões persistidas.

    Returns:
        Dicionário normalizado contendo métricas explícitas de target, treino
        e referência, além de aliases legados usados pelo relatório.

    Raises:
        Não se aplica.
    """
    payload = _safe_dict(attack_report)
    summary = _safe_dict(payload.get('summary'))
    metrics = _safe_dict(payload.get('metrics'))
    full_report = _safe_dict(payload.get('full_report'))
    full_report_summary = _safe_dict(full_report.get('summary'))

    candidate_sources: list[Mapping[str, Any]] = [
        summary,
        metrics,
        full_report_summary,
        full_report,
        payload,
    ]

    def pick(*aliases: str) -> float | None:
        for source in candidate_sources:
            value = _pick_first_float(source, aliases)
            if value is not None:
                return value
        return None

    target_exact_match_rate = pick(
        'exact_target_match_rate',
        'target_exact_match_rate',
    )
    target_partial_match_rate = pick(
        'partial_target_match_rate',
        'target_partial_match_rate',
    )
    train_secret_exact_match_rate = pick(
        'train_secret_exact_match_rate',
        'train_exact_match_rate',
    )
    train_secret_partial_match_rate = pick(
        'train_secret_partial_match_rate',
        'train_partial_match_rate',
    )
    reference_secret_exact_match_rate = pick(
        'reference_secret_exact_match_rate',
        'reference_exact_match_rate',
        'secret_exact_match_rate',
        'exact_match_rate',
    )
    reference_secret_partial_match_rate = pick(
        'reference_secret_partial_match_rate',
        'reference_partial_match_rate',
        'secret_partial_match_rate',
        'partial_match_rate',
    )
    structured_entity_generation_rate = pick('structured_entity_generation_rate')
    canary_recovery_rate = pick('canary_recovery_rate', 'canary_recovery')
    num_records = pick(
        'num_records',
        'num_attack_generations',
        'num_generations',
        'total_records',
        'n_records',
        'count',
    )
    num_prompts = pick('num_prompts', 'prompt_count')
    num_generations = pick('num_generations', 'num_attack_generations', 'generation_count')

    return {
        'target_exact_match_rate': target_exact_match_rate,
        'target_partial_match_rate': target_partial_match_rate,
        'train_secret_exact_match_rate': train_secret_exact_match_rate,
        'train_secret_partial_match_rate': train_secret_partial_match_rate,
        'reference_secret_exact_match_rate': reference_secret_exact_match_rate,
        'reference_secret_partial_match_rate': reference_secret_partial_match_rate,
        'exact_match_rate': reference_secret_exact_match_rate,
        'partial_match_rate': reference_secret_partial_match_rate,
        'structured_entity_generation_rate': structured_entity_generation_rate,
        'canary_recovery_rate': canary_recovery_rate,
        'num_records': num_records,
        'num_prompts': num_prompts,
        'num_generations': num_generations,
    }


def _compute_pct_change(
    baseline_value: float | None,
    comparison_value: float | None,
) -> float | None:
    """Calcula variação percentual relativa.

    Args:
        baseline_value: Valor de referência.
        comparison_value: Valor comparado.

    Returns:
        Percentual relativo ou ``None``.

    Raises:
        Não se aplica.
    """
    if baseline_value is None or comparison_value is None:
        return None
    if baseline_value == 0:
        return None
    return ((comparison_value - baseline_value) / baseline_value) * 100.0


def _compute_absolute_delta(
    baseline_value: float | None,
    comparison_value: float | None,
) -> float | None:
    """Calcula diferença absoluta entre dois valores.

    Args:
        baseline_value: Valor de referência.
        comparison_value: Valor comparado.

    Returns:
        Diferença absoluta ou ``None``.

    Raises:
        Não se aplica.
    """
    if baseline_value is None or comparison_value is None:
        return None
    return comparison_value - baseline_value


def _fmt_pct(value: float | None) -> str:
    """Formata percentuais humanos.

    Args:
        value: Valor percentual.

    Returns:
        String formatada.

    Raises:
        Não se aplica.
    """
    return 'N/A' if value is None else f'{value:.2f}%'


def _fmt_float(value: float | None, decimals: int = 4) -> str:
    """Formata floats com tolerância a ``None``.

    Args:
        value: Valor numérico.
        decimals: Casas decimais.

    Returns:
        String formatada.

    Raises:
        Não se aplica.
    """
    return 'N/A' if value is None else f'{value:.{decimals}f}'


def _describe_signed_change(metric_name: str, value: float | None) -> str:
    """Produz descrição textual curta para uma mudança percentual.

    Args:
        metric_name: Nome da métrica.
        value: Mudança percentual.

    Returns:
        Texto resumido.

    Raises:
        Não se aplica.
    """
    if value is None:
        return f'{metric_name}: N/A.'

    if value > 0:
        return f'{metric_name}: aumento de {_fmt_pct(abs(value))}.'
    if value < 0:
        return f'{metric_name}: redução de {_fmt_pct(abs(value))}.'
    return f'{metric_name}: sem variação (0.00%).'


def _describe_reduction_vs_attack(
    metric_name: str,
    attack_value: float | None,
    defended_value: float | None,
) -> str:
    """Descreve redução ou aumento relativo frente à condição com ataque.

    Args:
        metric_name: Nome da métrica.
        attack_value: Valor da condição com ataque.
        defended_value: Valor da condição defendida.

    Returns:
        Texto resumido.

    Raises:
        Não se aplica.
    """
    if attack_value is None or defended_value is None:
        return f'{metric_name} vs ataque: N/A.'

    if attack_value == 0 and defended_value == 0:
        return f'{metric_name} vs ataque: sem variação (ambos 0.00%).'

    if attack_value == 0 and defended_value != 0:
        delta_abs = defended_value - attack_value
        return (
            f'{metric_name} vs ataque: aumento absoluto de '
            f'{_fmt_float(delta_abs)} (baseline de ataque = 0.00).'
        )

    pct_change = _compute_pct_change(attack_value, defended_value)
    if pct_change is None:
        return f'{metric_name} vs ataque: N/A.'

    if pct_change < 0:
        return f'{metric_name} vs ataque: redução de {_fmt_pct(abs(pct_change))}.'
    if pct_change > 0:
        return f'{metric_name} vs ataque: aumento de {_fmt_pct(abs(pct_change))}.'
    return f'{metric_name} vs ataque: sem variação (0.00%).'


def normalize_condition_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Normaliza um pipeline summary para uma linha tabular estável.

    Args:
        summary: Summary consolidado de uma condição experimental.

    Returns:
        Linha normalizada com métricas explícitas e aliases legados.

    Raises:
        ValueError: Se o campo ``condition`` estiver ausente ou inválido.
    """
    raw_condition = summary.get('condition')
    if not isinstance(raw_condition, str) or not raw_condition.strip():
        raise ValueError("O summary não contém o campo 'condition' válido.")
    condition = raw_condition.strip()

    sanity_section = _safe_dict(summary.get('sanity'))
    federated_section = _safe_dict(summary.get('federated'))

    sanity_attack = _safe_dict(summary.get('sanity_attack'))
    federated_attack = _safe_dict(summary.get('federated_attack'))

    dp_config = _safe_dict(summary.get('dp_config'))
    privacy_report = _safe_dict(summary.get('privacy_report'))

    sanity_split_metrics = _extract_split_metrics(sanity_section, split_name='test')
    federated_split_metrics = _extract_split_metrics(federated_section, split_name='test')
    sanity_attack_metrics = _extract_attack_summary(sanity_attack)
    federated_attack_metrics = _extract_attack_summary(federated_attack)

    epsilon = (
        _safe_float(privacy_report.get('epsilon'))
        or _safe_float(dp_config.get('epsilon'))
        or _safe_float(dp_config.get('target_epsilon'))
    )
    delta = _safe_float(privacy_report.get('delta')) or _safe_float(dp_config.get('target_delta'))
    noise_multiplier = _safe_float(privacy_report.get('noise_multiplier')) or _safe_float(
        dp_config.get('noise_multiplier')
    )
    max_grad_norm = _safe_float(privacy_report.get('max_grad_norm')) or _safe_float(
        dp_config.get('max_grad_norm')
    )

    return {
        'condition': condition,
        'condition_display_name': DISPLAY_NAME_BY_CONDITION.get(condition, condition),
        'sanity_test_loss': sanity_split_metrics['loss'],
        'sanity_test_perplexity': sanity_split_metrics['perplexity'],
        'federated_test_loss': federated_split_metrics['loss'],
        'federated_test_perplexity': federated_split_metrics['perplexity'],
        'sanity_target_exact_match_rate': sanity_attack_metrics['target_exact_match_rate'],
        'sanity_target_partial_match_rate': sanity_attack_metrics['target_partial_match_rate'],
        'sanity_train_secret_exact_match_rate': sanity_attack_metrics[
            'train_secret_exact_match_rate'
        ],
        'sanity_train_secret_partial_match_rate': sanity_attack_metrics[
            'train_secret_partial_match_rate'
        ],
        'sanity_reference_secret_exact_match_rate': sanity_attack_metrics[
            'reference_secret_exact_match_rate'
        ],
        'sanity_reference_secret_partial_match_rate': sanity_attack_metrics[
            'reference_secret_partial_match_rate'
        ],
        'sanity_exact_match_rate': sanity_attack_metrics['reference_secret_exact_match_rate'],
        'sanity_partial_match_rate': sanity_attack_metrics['reference_secret_partial_match_rate'],
        'sanity_structured_entity_generation_rate': sanity_attack_metrics[
            'structured_entity_generation_rate'
        ],
        'sanity_canary_recovery_rate': sanity_attack_metrics['canary_recovery_rate'],
        'sanity_num_attack_records': sanity_attack_metrics['num_records'],
        'sanity_num_attack_prompts': sanity_attack_metrics['num_prompts'],
        'sanity_num_attack_generations': sanity_attack_metrics['num_generations'],
        'federated_target_exact_match_rate': federated_attack_metrics['target_exact_match_rate'],
        'federated_target_partial_match_rate': federated_attack_metrics[
            'target_partial_match_rate'
        ],
        'federated_train_secret_exact_match_rate': federated_attack_metrics[
            'train_secret_exact_match_rate'
        ],
        'federated_train_secret_partial_match_rate': federated_attack_metrics[
            'train_secret_partial_match_rate'
        ],
        'federated_reference_secret_exact_match_rate': federated_attack_metrics[
            'reference_secret_exact_match_rate'
        ],
        'federated_reference_secret_partial_match_rate': federated_attack_metrics[
            'reference_secret_partial_match_rate'
        ],
        'federated_exact_match_rate': federated_attack_metrics['reference_secret_exact_match_rate'],
        'federated_partial_match_rate': federated_attack_metrics[
            'reference_secret_partial_match_rate'
        ],
        'federated_structured_entity_generation_rate': federated_attack_metrics[
            'structured_entity_generation_rate'
        ],
        'federated_canary_recovery_rate': federated_attack_metrics['canary_recovery_rate'],
        'federated_num_attack_records': federated_attack_metrics['num_records'],
        'federated_num_attack_prompts': federated_attack_metrics['num_prompts'],
        'federated_num_attack_generations': federated_attack_metrics['num_generations'],
        'dp_noise_multiplier': noise_multiplier,
        'dp_max_grad_norm': max_grad_norm,
        'dp_target_delta': delta,
        'dp_epsilon': epsilon,
        'raw_summary': dict(summary),
    }


def summaries_to_rows(summaries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Converte vários summaries em linhas ordenadas.

    Args:
        summaries: Iterable de pipeline summaries.

    Returns:
        Lista de linhas normalizadas em ordem preferencial.

    Raises:
        ValueError: Se algum summary estiver inválido.
    """
    rows = [normalize_condition_summary(summary) for summary in summaries]

    preferred_order = {
        'no_attacker': 0,
        'raw': 1,
        'attack_raw': 1,
        'semantic_substitution': 2,
        'attack_semantic_substitution': 2,
        'dp': 3,
        'attack_raw_dp': 3,
    }
    rows.sort(key=lambda item: (preferred_order.get(item['condition'], 999), item['condition']))
    return rows


def build_comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Monta comparações agregadas entre as quatro condições principais.

    Args:
        rows: Linhas normalizadas por condição.

    Returns:
        Dicionário com linhas, comparações numéricas e interpretação textual.

    Raises:
        Não se aplica.
    """
    index = {row['condition']: row for row in rows}

    no_attacker = index.get('no_attacker')
    raw = index.get('raw') or index.get('attack_raw')
    semantic = index.get('semantic_substitution') or index.get('attack_semantic_substitution')
    dp = index.get('dp') or index.get('attack_raw_dp')

    comparisons: dict[str, float | None] = {}
    interpretation: list[str] = []

    no_attacker_exact = (
        _safe_float(no_attacker.get('federated_reference_secret_exact_match_rate'))
        if no_attacker
        else None
    )
    raw_exact = _safe_float(raw.get('federated_reference_secret_exact_match_rate')) if raw else None
    semantic_exact = (
        _safe_float(semantic.get('federated_reference_secret_exact_match_rate'))
        if semantic
        else None
    )
    dp_exact = _safe_float(dp.get('federated_reference_secret_exact_match_rate')) if dp else None

    no_attacker_partial = (
        _safe_float(no_attacker.get('federated_reference_secret_partial_match_rate'))
        if no_attacker
        else None
    )
    raw_partial = (
        _safe_float(raw.get('federated_reference_secret_partial_match_rate')) if raw else None
    )
    semantic_partial = (
        _safe_float(semantic.get('federated_reference_secret_partial_match_rate'))
        if semantic
        else None
    )
    dp_partial = (
        _safe_float(dp.get('federated_reference_secret_partial_match_rate')) if dp else None
    )

    no_attacker_entity = (
        _safe_float(no_attacker.get('federated_structured_entity_generation_rate'))
        if no_attacker
        else None
    )
    raw_entity = (
        _safe_float(raw.get('federated_structured_entity_generation_rate')) if raw else None
    )
    semantic_entity = (
        _safe_float(semantic.get('federated_structured_entity_generation_rate'))
        if semantic
        else None
    )
    dp_entity = _safe_float(dp.get('federated_structured_entity_generation_rate')) if dp else None

    no_attacker_canary = (
        _safe_float(no_attacker.get('federated_canary_recovery_rate')) if no_attacker else None
    )
    raw_canary = _safe_float(raw.get('federated_canary_recovery_rate')) if raw else None
    semantic_canary = (
        _safe_float(semantic.get('federated_canary_recovery_rate')) if semantic else None
    )
    dp_canary = _safe_float(dp.get('federated_canary_recovery_rate')) if dp else None

    raw_ppl = _safe_float(raw.get('federated_test_perplexity')) if raw else None
    semantic_ppl = _safe_float(semantic.get('federated_test_perplexity')) if semantic else None
    dp_ppl = _safe_float(dp.get('federated_test_perplexity')) if dp else None

    if no_attacker and raw:
        comparisons['attack_gain_reference_exact_match_pct'] = _compute_pct_change(
            no_attacker_exact,
            raw_exact,
        )
        comparisons['attack_gain_reference_partial_match_pct'] = _compute_pct_change(
            no_attacker_partial,
            raw_partial,
        )
        comparisons['attack_gain_entity_pct'] = _compute_pct_change(
            no_attacker_entity,
            raw_entity,
        )
        comparisons['attack_gain_canary_pct'] = _compute_pct_change(
            no_attacker_canary,
            raw_canary,
        )

    if raw and semantic:
        comparisons['semantic_reference_exact_match_delta_pct_vs_attack'] = _compute_pct_change(
            raw_exact,
            semantic_exact,
        )
        comparisons['semantic_reference_partial_match_delta_pct_vs_attack'] = _compute_pct_change(
            raw_partial,
            semantic_partial,
        )
        comparisons['semantic_entity_delta_pct_vs_attack'] = _compute_pct_change(
            raw_entity,
            semantic_entity,
        )
        comparisons['semantic_canary_delta_pct_vs_attack'] = _compute_pct_change(
            raw_canary,
            semantic_canary,
        )
        comparisons['semantic_perplexity_delta_vs_attack'] = _compute_absolute_delta(
            raw_ppl,
            semantic_ppl,
        )

    if raw and dp:
        comparisons['dp_reference_exact_match_delta_pct_vs_attack'] = _compute_pct_change(
            raw_exact,
            dp_exact,
        )
        comparisons['dp_reference_partial_match_delta_pct_vs_attack'] = _compute_pct_change(
            raw_partial,
            dp_partial,
        )
        comparisons['dp_entity_delta_pct_vs_attack'] = _compute_pct_change(
            raw_entity,
            dp_entity,
        )
        comparisons['dp_canary_delta_pct_vs_attack'] = _compute_pct_change(
            raw_canary,
            dp_canary,
        )
        comparisons['dp_perplexity_delta_vs_attack'] = _compute_absolute_delta(
            raw_ppl,
            dp_ppl,
        )

    if semantic and dp:
        comparisons['dp_vs_semantic_reference_exact_match_delta_pct'] = _compute_pct_change(
            semantic_exact,
            dp_exact,
        )
        comparisons['dp_vs_semantic_reference_partial_match_delta_pct'] = _compute_pct_change(
            semantic_partial,
            dp_partial,
        )
        comparisons['dp_vs_semantic_entity_delta_pct'] = _compute_pct_change(
            semantic_entity,
            dp_entity,
        )
        comparisons['dp_vs_semantic_canary_delta_pct'] = _compute_pct_change(
            semantic_canary,
            dp_canary,
        )
        comparisons['dp_vs_semantic_perplexity_delta'] = _compute_absolute_delta(
            semantic_ppl,
            dp_ppl,
        )

    if 'attack_gain_reference_exact_match_pct' in comparisons:
        interpretation.append(
            'Impacto do atacante na reference_secret_exact_match_rate: '
            f'{_describe_signed_change("federated", comparisons["attack_gain_reference_exact_match_pct"])}'
        )
    if 'attack_gain_reference_partial_match_pct' in comparisons:
        interpretation.append(
            'Impacto do atacante na reference_secret_partial_match_rate: '
            f'{_describe_signed_change("federated", comparisons["attack_gain_reference_partial_match_pct"])}'
        )
    if 'attack_gain_entity_pct' in comparisons:
        interpretation.append(
            'Impacto do atacante na structured_entity_generation_rate: '
            f'{_describe_signed_change("federated", comparisons["attack_gain_entity_pct"])}'
        )
    if 'attack_gain_canary_pct' in comparisons:
        interpretation.append(
            'Impacto do atacante na canary_recovery_rate: '
            f'{_describe_signed_change("federated", comparisons["attack_gain_canary_pct"])}'
        )

    if raw and semantic:
        interpretation.append(
            _describe_reduction_vs_attack(
                'Defesa semântica na reference_secret_exact_match_rate',
                raw_exact,
                semantic_exact,
            )
        )
        interpretation.append(
            _describe_reduction_vs_attack(
                'Defesa semântica na reference_secret_partial_match_rate',
                raw_partial,
                semantic_partial,
            )
        )
        interpretation.append(
            _describe_reduction_vs_attack(
                'Defesa semântica na structured_entity_generation_rate',
                raw_entity,
                semantic_entity,
            )
        )
        interpretation.append(
            _describe_reduction_vs_attack(
                'Defesa semântica na canary_recovery_rate',
                raw_canary,
                semantic_canary,
            )
        )
        interpretation.append(
            'Custo de utilidade da defesa semântica (vs ataque): '
            f'Δperplexity={_fmt_float(comparisons.get("semantic_perplexity_delta_vs_attack"))}.'
        )

    if raw and dp:
        interpretation.append(
            _describe_reduction_vs_attack(
                'Baseline DP na reference_secret_exact_match_rate',
                raw_exact,
                dp_exact,
            )
        )
        interpretation.append(
            _describe_reduction_vs_attack(
                'Baseline DP na reference_secret_partial_match_rate',
                raw_partial,
                dp_partial,
            )
        )
        interpretation.append(
            _describe_reduction_vs_attack(
                'Baseline DP na structured_entity_generation_rate',
                raw_entity,
                dp_entity,
            )
        )
        interpretation.append(
            _describe_reduction_vs_attack(
                'Baseline DP na canary_recovery_rate',
                raw_canary,
                dp_canary,
            )
        )
        interpretation.append(
            'Custo de utilidade do DP (vs ataque): '
            f'Δperplexity={_fmt_float(comparisons.get("dp_perplexity_delta_vs_attack"))}.'
        )

    if semantic and dp:
        interpretation.append(
            'DP vs semântica na reference_secret_exact_match_rate: '
            f'{_describe_signed_change("federated", comparisons.get("dp_vs_semantic_reference_exact_match_delta_pct"))}'
        )
        interpretation.append(
            'DP vs semântica na reference_secret_partial_match_rate: '
            f'{_describe_signed_change("federated", comparisons.get("dp_vs_semantic_reference_partial_match_delta_pct"))}'
        )
        interpretation.append(
            'DP vs semântica na structured_entity_generation_rate: '
            f'{_describe_signed_change("federated", comparisons.get("dp_vs_semantic_entity_delta_pct"))}'
        )
        interpretation.append(
            'DP vs semântica na canary_recovery_rate: '
            f'{_describe_signed_change("federated", comparisons.get("dp_vs_semantic_canary_delta_pct"))}'
        )
        interpretation.append(
            'Custo de utilidade do DP vs semântica: '
            f'Δperplexity={_fmt_float(comparisons.get("dp_vs_semantic_perplexity_delta"))}.'
        )

    if dp:
        epsilon = _safe_float(dp.get('dp_epsilon'))
        noise = _safe_float(dp.get('dp_noise_multiplier'))
        delta = _safe_float(dp.get('dp_target_delta'))
        max_grad_norm = _safe_float(dp.get('dp_max_grad_norm'))

        if noise is not None:
            interpretation.append(f'O baseline DP utilizou noise_multiplier={_fmt_float(noise)}.')
        if delta is not None:
            interpretation.append(f'O baseline DP utilizou target_delta={delta}.')
        if max_grad_norm is not None:
            interpretation.append(
                f'O baseline DP utilizou max_grad_norm={_fmt_float(max_grad_norm)}.'
            )
        if epsilon is not None:
            interpretation.append(f'O modelo DP atingiu epsilon={_fmt_float(epsilon)}.')

    return {
        'rows': rows,
        'comparisons': comparisons,
        'interpretation': interpretation,
    }


def save_rows_csv(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    exclude_fields: tuple[str, ...] = ('raw_summary',),
) -> Path:
    """Salva tabela por condição em CSV.

    Args:
        rows: Linhas normalizadas.
        output_path: Caminho do arquivo.
        exclude_fields: Campos a excluir.

    Returns:
        Caminho salvo.

    Raises:
        OSError: Se a gravação falhar.
    """
    resolved_path = Path(output_path)
    ensure_directory(resolved_path.parent)

    filtered_rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []

    for row in rows:
        filtered = {key: value for key, value in row.items() if key not in exclude_fields}
        filtered_rows.append(filtered)
        for key in filtered.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with resolved_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in filtered_rows:
            writer.writerow(row)

    LOGGER.info('Tabela CSV salva em %s', resolved_path)
    return resolved_path


def save_comparison_csv(comparison_summary: Mapping[str, Any], output_path: str | Path) -> Path:
    """Salva as comparações agregadas em CSV.

    Args:
        comparison_summary: Estrutura de comparação.
        output_path: Caminho do arquivo.

    Returns:
        Caminho salvo.

    Raises:
        OSError: Se a gravação falhar.
    """
    resolved_path = Path(output_path)
    ensure_directory(resolved_path.parent)

    comparisons = _safe_dict(comparison_summary.get('comparisons'))
    rows = [{'metric': key, 'value': value} for key, value in comparisons.items()]

    with resolved_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['metric', 'value'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    LOGGER.info('CSV comparativo salvo em %s', resolved_path)
    return resolved_path


def save_comparison_json(comparison_summary: Mapping[str, Any], output_path: str | Path) -> Path:
    """Salva o resumo consolidado em JSON.

    Args:
        comparison_summary: Estrutura consolidada.
        output_path: Caminho do arquivo.

    Returns:
        Caminho salvo.

    Raises:
        OSError: Se a gravação falhar.
    """
    resolved_path = Path(output_path)
    ensure_directory(resolved_path.parent)
    save_json(dict(comparison_summary), resolved_path)
    LOGGER.info('Resumo JSON salvo em %s', resolved_path)
    return resolved_path


def save_text_report(comparison_summary: Mapping[str, Any], output_path: str | Path) -> Path:
    """Salva um relatório textual de leitura humana.

    Args:
        comparison_summary: Estrutura consolidada.
        output_path: Caminho do arquivo.

    Returns:
        Caminho salvo.

    Raises:
        OSError: Se a gravação falhar.
    """
    resolved_path = Path(output_path)
    ensure_directory(resolved_path.parent)

    rows = list(comparison_summary.get('rows', []))
    comparisons = _safe_dict(comparison_summary.get('comparisons'))
    interpretation = list(comparison_summary.get('interpretation', []))

    lines: list[str] = []
    lines.append('Relatório consolidado de resultados')
    lines.append('=' * 72)
    lines.append('')

    lines.append('Condições avaliadas:')
    for row in rows:
        lines.append(
            f'- {row["condition"]} | '
            f'federated_test_perplexity={row.get("federated_test_perplexity")} | '
            f'federated_reference_secret_exact_match_rate={row.get("federated_reference_secret_exact_match_rate")} | '
            f'federated_reference_secret_partial_match_rate={row.get("federated_reference_secret_partial_match_rate")} | '
            f'federated_structured_entity_generation_rate={row.get("federated_structured_entity_generation_rate")} | '
            f'federated_canary_recovery_rate={row.get("federated_canary_recovery_rate")} | '
            f'dp_noise_multiplier={row.get("dp_noise_multiplier")} | '
            f'dp_target_delta={row.get("dp_target_delta")} | '
            f'dp_epsilon={row.get("dp_epsilon")}'
        )

    lines.append('')
    lines.append('Comparações agregadas:')
    for key, value in comparisons.items():
        lines.append(f'- {key}: {value}')

    lines.append('')
    lines.append('Interpretação automática:')
    if interpretation:
        for item in interpretation:
            lines.append(f'- {item}')
    else:
        lines.append('- Nenhuma interpretação pôde ser derivada automaticamente.')

    resolved_path.write_text('\n'.join(lines), encoding='utf-8')
    LOGGER.info('Relatório textual salvo em %s', resolved_path)
    return resolved_path


def _has_any_metric(rows: list[dict[str, Any]], metric_names: tuple[str, ...]) -> bool:
    """Verifica se há ao menos uma métrica válida entre várias colunas.

    Args:
        rows: Linhas tabulares.
        metric_names: Colunas a inspecionar.

    Returns:
        ``True`` se houver algum valor numérico válido.

    Raises:
        Não se aplica.
    """
    for row in rows:
        for metric_name in metric_names:
            if _safe_float(row.get(metric_name)) is not None:
                return True
    return False


def _resolve_metric_label(metric_name: str) -> str:
    """Resolve o rótulo amigável exibido na legenda do gráfico.

    Args:
        metric_name: Nome interno da coluna.

    Returns:
        Rótulo amigável.

    Raises:
        Não se aplica.
    """
    return PLOT_LABEL_BY_METRIC.get(metric_name, metric_name)


def _plot_bar_chart(
    rows: list[dict[str, Any]],
    metric_names: tuple[str, ...],
    title: str,
    output_path: str | Path,
    ylabel: str,
) -> Path:
    """Gera um gráfico de barras agrupadas.

    Args:
        rows: Linhas tabulares por condição.
        metric_names: Colunas a plotar.
        title: Título do gráfico.
        output_path: Caminho do PNG.
        ylabel: Rótulo do eixo Y.

    Returns:
        Caminho do gráfico salvo.

    Raises:
        OSError: Se a gravação falhar.
    """
    resolved_path = Path(output_path)
    ensure_directory(resolved_path.parent)

    labels = [row['condition_display_name'] for row in rows]
    x_positions = list(range(len(rows)))

    plt.figure(figsize=(12, 6))

    if not _has_any_metric(rows, metric_names):
        plt.text(
            0.5,
            0.5,
            'Nenhuma métrica válida encontrada para este gráfico.',
            ha='center',
            va='center',
            fontsize=12,
            transform=plt.gca().transAxes,
        )
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(resolved_path, dpi=200)
        plt.close()
        LOGGER.warning(
            'Gráfico salvo sem dados válidos: %s | métricas=%s',
            resolved_path,
            metric_names,
        )
        return resolved_path

    width = 0.8 / max(len(metric_names), 1)

    for metric_index, metric_name in enumerate(metric_names):
        values = []
        positions = []

        for idx, row in enumerate(rows):
            metric_value = _safe_float(row.get(metric_name))

            if metric_value is None:
                continue

            values.append(metric_value)
            positions.append(idx)

        offset = (metric_index - (len(metric_names) - 1) / 2.0) * width
        bar_positions = [position + offset for position in positions]
        plt.bar(
            bar_positions,
            values,
            width=width,
            label=_resolve_metric_label(metric_name),
        )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x_positions, labels, rotation=15, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(resolved_path, dpi=200)
    plt.close()

    LOGGER.info('Gráfico salvo em %s', resolved_path)
    return resolved_path


def plot_utility_metrics(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    """Gera gráficos de utilidade final.

    Args:
        rows: Linhas por condição.
        output_dir: Diretório de saída.

    Returns:
        Dicionário ``nome -> caminho``.

    Raises:
        OSError: Se alguma gravação falhar.
    """
    resolved_dir = ensure_directory(output_dir)

    loss_path = _plot_bar_chart(
        rows=rows,
        metric_names=('sanity_test_loss', 'federated_test_loss'),
        title='Loss por condição',
        output_path=resolved_dir / 'utility_loss.png',
        ylabel='Loss',
    )
    perplexity_path = _plot_bar_chart(
        rows=rows,
        metric_names=('sanity_test_perplexity', 'federated_test_perplexity'),
        title='Perplexity por condição',
        output_path=resolved_dir / 'utility_perplexity.png',
        ylabel='Perplexity',
    )

    return {
        'utility_loss': loss_path,
        'utility_perplexity': perplexity_path,
    }


def plot_leakage_metrics(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    """Gera gráficos de leakage final.

    Args:
        rows: Linhas por condição.
        output_dir: Diretório de saída.

    Returns:
        Dicionário ``nome -> caminho``.

    Raises:
        OSError: Se alguma gravação falhar.
    """
    resolved_dir = ensure_directory(output_dir)

    exact_path = _plot_bar_chart(
        rows=rows,
        metric_names=(
            'sanity_reference_secret_exact_match_rate',
            'federated_reference_secret_exact_match_rate',
        ),
        title='Reference secret exact match rate por condição',
        output_path=resolved_dir / 'leakage_exact_match.png',
        ylabel='Taxa',
    )
    partial_path = _plot_bar_chart(
        rows=rows,
        metric_names=(
            'sanity_reference_secret_partial_match_rate',
            'federated_reference_secret_partial_match_rate',
        ),
        title='Reference secret partial match rate por condição',
        output_path=resolved_dir / 'leakage_partial_match.png',
        ylabel='Taxa',
    )
    structured_entity_path = _plot_bar_chart(
        rows=rows,
        metric_names=(
            'sanity_structured_entity_generation_rate',
            'federated_structured_entity_generation_rate',
        ),
        title='Structured entity generation rate por condição',
        output_path=resolved_dir / 'leakage_structured_entity_generation.png',
        ylabel='Taxa',
    )
    canary_path = _plot_bar_chart(
        rows=rows,
        metric_names=('sanity_canary_recovery_rate', 'federated_canary_recovery_rate'),
        title='Canary recovery rate por condição',
        output_path=resolved_dir / 'leakage_canary_rate.png',
        ylabel='Taxa',
    )

    return {
        'leakage_exact_match': exact_path,
        'leakage_partial_match': partial_path,
        'leakage_structured_entity_generation': structured_entity_path,
        'leakage_canary_rate': canary_path,
    }
