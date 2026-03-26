"""
Utilitários para orquestração de experimentos ponta a ponta.

Responsabilidades principais:
- Executar módulos Python do projeto via subprocess de forma reprodutível.
- Localizar automaticamente artefatos recentes em results/.
- Ler relatórios produzidos pelas etapas de sanity check, FL e ataque.
- Consolidar resumos simples por condição e entre condições.

Como este arquivo se encaixa no projeto:
- É usado pelos scripts em experiments/ para encadear as execuções já
  implementadas no restante do projeto, sem duplicar lógica de treino,
  geração ou ataque.
"""

from __future__ import annotations

import subprocess
import sys
import select
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils.io import load_json, save_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class CommandResult:
    """Representa o resultado de um comando executado.

    Args:
        name: Nome lógico da etapa executada.
        module: Módulo Python chamado com `python -m`.
        args: Argumentos passados ao módulo.
        return_code: Código de retorno do processo.
        stdout_path: Caminho do log de stdout.
        stderr_path: Caminho do log de stderr.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    name: str
    module: str
    args: list[str]
    return_code: int
    stdout_path: str
    stderr_path: str


@dataclass
class ExperimentArtifacts:
    """Agrupa os artefatos principais produzidos por uma condição experimental.

    Args:
        condition: Nome da condição experimental.
        sanity_dir: Diretório do sanity check centralizado.
        sanity_checkpoint_dir: Diretório do checkpoint do sanity check.
        sanity_attack_dir: Diretório do ataque ao checkpoint do sanity check.
        federated_dir: Diretório da execução federada.
        federated_checkpoint_dir: Diretório do checkpoint final do FL.
        federated_attack_dir: Diretório do ataque ao checkpoint final do FL.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    condition: str
    sanity_dir: str
    sanity_checkpoint_dir: str
    sanity_attack_dir: str
    federated_dir: str
    federated_checkpoint_dir: str
    federated_attack_dir: str


def ensure_directory(path: Path) -> Path:
    """Cria um diretório, se necessário.

    Args:
        path: Diretório alvo.

    Returns:
        O próprio caminho criado/existente.

    Raises:
        OSError: Se a criação falhar.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_python_module(
    module: str,
    args: Iterable[str],
    logs_dir: Path,
    step_name: str,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Executa um módulo Python do projeto e salva logs em disco + terminal."""
    ensure_directory(logs_dir)

    stdout_path = logs_dir / f'{step_name}.stdout.log'
    stderr_path = logs_dir / f'{step_name}.stderr.log'

    args_list = list(args)
    command = [sys.executable, '-m', module, *args_list]
    LOGGER.info('Executando comando: %s', ' '.join(command))

    with (
        stdout_path.open('w', encoding='utf-8') as stdout_file,
        stderr_path.open('w', encoding='utf-8') as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,  # line-buffered
        )

        pipes = [process.stdout, process.stderr]

        while pipes:
            readable, _, _ = select.select(pipes, [], [])
            for pipe in readable:
                line = pipe.readline()

                if not line:
                    pipes.remove(pipe)
                    continue

                if pipe is process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    stdout_file.write(line)
                    stdout_file.flush()
                else:
                    sys.stderr.write(line)
                    sys.stderr.flush()
                    stderr_file.write(line)
                    stderr_file.flush()

        process.wait()

    result = CommandResult(
        name=step_name,
        module=module,
        args=args_list,
        return_code=process.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )

    if process.returncode != 0:
        raise RuntimeError(
            'Falha ao executar etapa '
            f"'{step_name}' (módulo={module}, return_code={process.returncode}). "
            f'Verifique os logs em {stdout_path} e {stderr_path}.'
        )

    return result


def find_latest_directory_by_prefix(base_dir: Path, prefix: str) -> Path:
    """Localiza o diretório mais recente com determinado prefixo.

    Args:
        base_dir: Diretório base de busca.
        prefix: Prefixo esperado no nome do diretório.

    Returns:
        Caminho do diretório mais recente.

    Raises:
        FileNotFoundError: Se nenhum diretório compatível for encontrado.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f'Diretório base não encontrado: {base_dir}')

    matches = [
        path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)
    ]
    if not matches:
        raise FileNotFoundError(
            f"Nenhum diretório com prefixo '{prefix}' encontrado em {base_dir}."
        )

    matches.sort(key=lambda path: (path.stat().st_mtime, path.name))
    return matches[-1]


def find_latest_directory_by_prefix_with_file(
    base_dir: Path,
    prefix: str,
    required_file: str,
) -> Path:
    """Localiza o diretório mais recente com prefixo e arquivo obrigatório.

    Args:
        base_dir: Diretório base de busca.
        prefix: Prefixo esperado no nome do diretório.
        required_file: Arquivo que deve existir dentro do diretório encontrado.

    Returns:
        Caminho do diretório mais recente compatível.

    Raises:
        FileNotFoundError: Se nenhum diretório compatível for encontrado.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f'Diretório base não encontrado: {base_dir}')

    matches = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and (path / required_file).exists()
    ]
    if not matches:
        raise FileNotFoundError(
            f"Nenhum diretório com prefixo '{prefix}' e arquivo '{required_file}' encontrado em {base_dir}."
        )

    matches.sort(key=lambda path: (path.stat().st_mtime, path.name))
    return matches[-1]


def resolve_exact_or_prefixed_directory(
    base_dir: Path,
    exact_dir_name: str | None,
    fallback_prefixes: list[str],
    required_file: str | None = None,
) -> Path:
    """Resolve um diretório por nome exato ou por prefixos alternativos.

    Esta função existe para lidar com a coexistência de padrões antigos e novos
    de nomenclatura dos diretórios de resultados.

    Args:
        base_dir: Diretório base onde a busca será feita.
        exact_dir_name: Nome exato preferido.
        fallback_prefixes: Lista ordenada de prefixos alternativos.
        required_file: Arquivo obrigatório dentro do diretório, se aplicável.

    Returns:
        Caminho do diretório encontrado.

    Raises:
        FileNotFoundError: Se nenhum diretório compatível for encontrado.
    """
    if exact_dir_name:
        exact_path = base_dir / exact_dir_name
        if exact_path.exists() and exact_path.is_dir():
            if required_file is None or (exact_path / required_file).exists():
                return exact_path

    for prefix in fallback_prefixes:
        try:
            if required_file is None:
                return find_latest_directory_by_prefix(base_dir, prefix)
            return find_latest_directory_by_prefix_with_file(
                base_dir=base_dir,
                prefix=prefix,
                required_file=required_file,
            )
        except FileNotFoundError:
            continue

    expected = [exact_dir_name] if exact_dir_name else []
    expected.extend(fallback_prefixes)
    raise FileNotFoundError(
        f'Nenhum diretório compatível encontrado em {base_dir}. '
        f'Buscas tentadas: {expected}. Arquivo requerido: {required_file}.'
    )


def resolve_federated_run_dir(
    base_dir: Path,
    requested_run_name: str,
    condition_name: str,
) -> Path:
    """Resolve o diretório da execução federada com tolerância a nomes legados.

    Args:
        base_dir: Diretório base de resultados.
        requested_run_name: Nome solicitado pelo script chamador.
        condition_name: Condição experimental do treino federado.

    Returns:
        Caminho da execução federada.

    Raises:
        FileNotFoundError: Se nenhum diretório federado compatível for encontrado.
    """
    fallback_prefixes = [
        requested_run_name,
        f'federated_{condition_name}_',
        f'federated_{condition_name}',
    ]
    return resolve_exact_or_prefixed_directory(
        base_dir=base_dir,
        exact_dir_name=requested_run_name,
        fallback_prefixes=fallback_prefixes,
        required_file='federated_report.json',
    )


def find_latest_round_checkpoint(federated_dir: Path) -> Path:
    """Localiza o checkpoint mais recente de uma execução federada.

    Args:
        federated_dir: Diretório raiz da execução federada.

    Returns:
        Caminho para `round_xxx/checkpoint`.

    Raises:
        FileNotFoundError: Se nenhum checkpoint de rodada for encontrado.
    """
    round_dirs = [
        path for path in federated_dir.iterdir() if path.is_dir() and path.name.startswith('round_')
    ]
    if not round_dirs:
        raise FileNotFoundError(f'Nenhuma rodada encontrada em {federated_dir}')

    round_dirs.sort(key=lambda path: path.name)
    checkpoint_dir = round_dirs[-1] / 'checkpoint'
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f'Checkpoint não encontrado: {checkpoint_dir}')
    return checkpoint_dir


def load_json_if_exists(path: Path) -> dict[str, Any]:
    """Lê um JSON se ele existir; caso contrário, retorna dict vazio.

    Args:
        path: Caminho do arquivo JSON.

    Returns:
        Conteúdo do JSON ou `{}`.

    Raises:
        Não se aplica.
    """
    if not path.exists():
        return {}
    return load_json(path)


def summarize_sanity_run(sanity_dir: Path) -> dict[str, Any]:
    """Extrai métricas relevantes de uma execução de sanity check.

    Args:
        sanity_dir: Diretório da execução.

    Returns:
        Dicionário resumido com métricas e caminhos principais.

    Raises:
        Não se aplica.
    """
    run_summary = load_json_if_exists(sanity_dir / 'run_summary.json')
    training_report = load_json_if_exists(sanity_dir / 'training_report.json')

    splits = run_summary.get('final_metrics', {}) or training_report.get(
        'evaluation_report', {}
    ).get('splits', {})

    return {
        'run_dir': str(sanity_dir),
        'checkpoint_dir': str(sanity_dir / 'checkpoint'),
        'test_loss': splits.get('test', {}).get('loss'),
        'test_perplexity': splits.get('test', {}).get('perplexity'),
        'domain_test_loss': splits.get('domain_test', {}).get('loss'),
        'domain_test_perplexity': splits.get('domain_test', {}).get('perplexity'),
    }


def summarize_federated_run(federated_dir: Path) -> dict[str, Any]:
    """Extrai métricas relevantes de uma execução federada.

    Args:
        federated_dir: Diretório da execução federada.

    Returns:
        Dicionário resumido com métricas e caminhos principais.

    Raises:
        Não se aplica.
    """
    report = load_json_if_exists(federated_dir / 'federated_report.json')
    history = report.get('history', [])

    final_round = history[-1] if history else {}
    final_splits = final_round.get('global_evaluation', {}).get('splits', {})

    checkpoint_dir = str(find_latest_round_checkpoint(federated_dir))

    return {
        'run_dir': str(federated_dir),
        'checkpoint_dir': checkpoint_dir,
        'num_rounds': report.get('num_rounds'),
        'test_loss': final_splits.get('test', {}).get('loss'),
        'test_perplexity': final_splits.get('test', {}).get('perplexity'),
        'domain_test_loss': final_splits.get('domain_test', {}).get('loss'),
        'domain_test_perplexity': final_splits.get('domain_test', {}).get('perplexity'),
    }


def summarize_attack_run(attack_dir: Path) -> dict[str, Any]:
    attack_summary = load_json_if_exists(attack_dir / 'attack_summary.json')
    attack_report = load_json_if_exists(attack_dir / 'attack_report.json')

    source = attack_summary or attack_report
    metrics = source.get('metrics', source.get('summary', source))
    full_report = source.get('full_report', source)
    attack_config = full_report.get('attack_config', {})

    return {
        'run_dir': str(attack_dir),
        'exact_match_rate': metrics.get('exact_match_rate'),
        'partial_match_rate': metrics.get('partial_match_rate'),
        'structured_entity_generation_rate': metrics.get('structured_entity_generation_rate'),
        'canary_recovery_rate': metrics.get('canary_recovery_rate'),
        'num_prompts': (
            metrics.get('num_prompts')
            if metrics.get('num_prompts') is not None
            else attack_config.get('num_prompts')
        ),
        'num_generations': (
            metrics.get('num_generations')
            if metrics.get('num_generations') is not None
            else metrics.get('num_records')
        ),
    }


def build_condition_summary(
    condition: str,
    sanity_dir: Path,
    sanity_attack_dir: Path,
    federated_dir: Path,
    federated_attack_dir: Path,
    command_results: list[CommandResult],
) -> dict[str, Any]:
    """Monta um resumo consolidado de uma condição experimental.

    Args:
        condition: Nome da condição.
        sanity_dir: Diretório do sanity check.
        sanity_attack_dir: Diretório do ataque sobre o sanity check.
        federated_dir: Diretório da execução federada.
        federated_attack_dir: Diretório do ataque sobre o checkpoint federado.
        command_results: Lista de etapas executadas.

    Returns:
        Dicionário consolidado da condição.

    Raises:
        Não se aplica.
    """
    return {
        'condition': condition,
        'artifacts': asdict(
            ExperimentArtifacts(
                condition=condition,
                sanity_dir=str(sanity_dir),
                sanity_checkpoint_dir=str(sanity_dir / 'checkpoint'),
                sanity_attack_dir=str(sanity_attack_dir),
                federated_dir=str(federated_dir),
                federated_checkpoint_dir=str(find_latest_round_checkpoint(federated_dir)),
                federated_attack_dir=str(federated_attack_dir),
            )
        ),
        'sanity': summarize_sanity_run(sanity_dir),
        'sanity_attack': summarize_attack_run(sanity_attack_dir),
        'federated': summarize_federated_run(federated_dir),
        'federated_attack': summarize_attack_run(federated_attack_dir),
        'commands': [asdict(item) for item in command_results],
    }


def compute_relative_change(
    baseline_value: float | None,
    new_value: float | None,
) -> float | None:
    """Calcula mudança percentual relativa de `new_value` sobre `baseline_value`.

    Args:
        baseline_value: Valor de referência.
        new_value: Novo valor a comparar.

    Returns:
        Percentual de mudança, ou `None` se não for possível calcular.

    Raises:
        Não se aplica.
    """
    if baseline_value is None or new_value is None:
        return None
    if baseline_value == 0:
        return None
    return ((new_value - baseline_value) / baseline_value) * 100.0


def compute_relative_reduction(
    baseline_value: float | None,
    defended_value: float | None,
) -> float | None:
    """Calcula redução relativa percentual entre baseline e condição defendida.

    Args:
        baseline_value: Valor do baseline sem defesa.
        defended_value: Valor da condição defendida.

    Returns:
        Percentual de redução, ou `None` se não for possível.

    Raises:
        Não se aplica.
    """
    if baseline_value is None or defended_value is None:
        return None
    if baseline_value == 0:
        return None
    return ((baseline_value - defended_value) / baseline_value) * 100.0


def build_three_condition_summary(
    no_attacker_summary: dict[str, Any],
    attack_summary: dict[str, Any],
    defense_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compara as três condições principais do experimento.

    Args:
        no_attacker_summary: Resumo consolidado da condição sem atacante.
        attack_summary: Resumo consolidado da condição com atacante.
        defense_summary: Resumo consolidado da condição com atacante + defesa.

    Returns:
        Dicionário com comparações úteis para análise final.

    Raises:
        Não se aplica.
    """
    no_attack_fed = no_attacker_summary.get('federated', {})
    attack_fed = attack_summary.get('federated', {})
    defense_fed = defense_summary.get('federated', {})

    no_attack_leak = no_attacker_summary.get('federated_attack', {})
    attack_leak = attack_summary.get('federated_attack', {})
    defense_leak = defense_summary.get('federated_attack', {})

    attack_gain_exact = compute_relative_change(
        no_attack_leak.get('exact_match_rate'),
        attack_leak.get('exact_match_rate'),
    )
    attack_gain_partial = compute_relative_change(
        no_attack_leak.get('partial_match_rate'),
        attack_leak.get('partial_match_rate'),
    )
    attack_gain_entity = compute_relative_change(
        no_attack_leak.get('entity_leakage_rate'),
        attack_leak.get('entity_leakage_rate'),
    )
    attack_gain_canary = compute_relative_change(
        no_attack_leak.get('canary_recovery_rate'),
        attack_leak.get('canary_recovery_rate'),
    )

    defense_reduction_exact = compute_relative_reduction(
        attack_leak.get('exact_match_rate'),
        defense_leak.get('exact_match_rate'),
    )
    defense_reduction_partial = compute_relative_reduction(
        attack_leak.get('partial_match_rate'),
        defense_leak.get('partial_match_rate'),
    )
    defense_reduction_entity = compute_relative_reduction(
        attack_leak.get('entity_leakage_rate'),
        defense_leak.get('entity_leakage_rate'),
    )
    defense_reduction_canary = compute_relative_reduction(
        attack_leak.get('canary_recovery_rate'),
        defense_leak.get('canary_recovery_rate'),
    )

    attack_vs_no_attacker_ppl_delta = None
    defense_vs_attack_ppl_delta = None
    defense_vs_no_attacker_ppl_delta = None

    if (
        no_attack_fed.get('test_perplexity') is not None
        and attack_fed.get('test_perplexity') is not None
    ):
        attack_vs_no_attacker_ppl_delta = (
            attack_fed['test_perplexity'] - no_attack_fed['test_perplexity']
        )

    if (
        attack_fed.get('test_perplexity') is not None
        and defense_fed.get('test_perplexity') is not None
    ):
        defense_vs_attack_ppl_delta = defense_fed['test_perplexity'] - attack_fed['test_perplexity']

    if (
        no_attack_fed.get('test_perplexity') is not None
        and defense_fed.get('test_perplexity') is not None
    ):
        defense_vs_no_attacker_ppl_delta = (
            defense_fed['test_perplexity'] - no_attack_fed['test_perplexity']
        )

    interpretation: list[str] = []

    if attack_gain_exact is not None:
        interpretation.append(
            'A presença do cliente malicioso aumentou '
            f'{attack_gain_exact:.2f}% a taxa de exact_match_rate em relação à condição sem atacante.'
        )
    if attack_gain_entity is not None:
        interpretation.append(
            'A presença do cliente malicioso aumentou '
            f'{attack_gain_entity:.2f}% a taxa de entity_leakage_rate em relação à condição sem atacante.'
        )
    if attack_gain_canary is not None:
        interpretation.append(
            'A presença do cliente malicioso aumentou '
            f'{attack_gain_canary:.2f}% a taxa de canary_recovery_rate em relação à condição sem atacante.'
        )
    if defense_reduction_exact is not None:
        interpretation.append(
            'A defesa semântica reduziu '
            f'{defense_reduction_exact:.2f}% da taxa de exact_match_rate em relação à condição com atacante.'
        )
    if defense_reduction_entity is not None:
        interpretation.append(
            'A defesa semântica reduziu '
            f'{defense_reduction_entity:.2f}% da taxa de entity_leakage_rate em relação à condição com atacante.'
        )
    if defense_reduction_canary is not None:
        interpretation.append(
            'A defesa semântica reduziu '
            f'{defense_reduction_canary:.2f}% da taxa de canary_recovery_rate em relação à condição com atacante.'
        )
    if defense_vs_attack_ppl_delta is not None:
        interpretation.append(
            'A diferença de perplexidade final em teste limpo '
            f'(defesa - ataque) foi {defense_vs_attack_ppl_delta:.4f}.'
        )

    return {
        'no_attacker': no_attacker_summary,
        'attack': attack_summary,
        'attack_semantic_defense': defense_summary,
        'comparisons': {
            'attack_gain_over_no_attacker': {
                'exact_match_rate_pct': attack_gain_exact,
                'partial_match_rate_pct': attack_gain_partial,
                'entity_leakage_rate_pct': attack_gain_entity,
                'canary_recovery_rate_pct': attack_gain_canary,
            },
            'defense_reduction_over_attack': {
                'exact_match_rate_pct': defense_reduction_exact,
                'partial_match_rate_pct': defense_reduction_partial,
                'entity_leakage_rate_pct': defense_reduction_entity,
                'canary_recovery_rate_pct': defense_reduction_canary,
            },
            'utility': {
                'attack_vs_no_attacker_test_perplexity_delta': attack_vs_no_attacker_ppl_delta,
                'defense_vs_attack_test_perplexity_delta': defense_vs_attack_ppl_delta,
                'defense_vs_no_attacker_test_perplexity_delta': defense_vs_no_attacker_ppl_delta,
            },
        },
        'interpretation': interpretation,
    }


def persist_summary(summary: dict[str, Any], output_path: Path) -> Path:
    """Salva resumo consolidado em JSON.

    Args:
        summary: Estrutura serializável.
        output_path: Caminho do arquivo de saída.

    Returns:
        Caminho salvo.

    Raises:
        OSError: Se a gravação falhar.
    """
    ensure_directory(output_path.parent)
    save_json(summary, output_path)
    return output_path


def persist_text_summary(summary: dict[str, Any], output_path: Path) -> Path:
    """Salva resumo textual simples para inspeção humana.

    Args:
        summary: Resumo consolidado.
        output_path: Caminho do arquivo `.txt`.

    Returns:
        Caminho salvo.

    Raises:
        OSError: Se a gravação falhar.
    """
    ensure_directory(output_path.parent)

    lines: list[str] = []
    lines.append('Resumo comparativo de condições experimentais')
    lines.append('=' * 60)
    lines.append('')

    comparisons = summary.get('comparisons', {})
    interpretation = summary.get('interpretation', [])

    for key, value in comparisons.items():
        lines.append(f'{key}: {value}')

    lines.append('')
    lines.append('Interpretação automática:')
    for item in interpretation:
        lines.append(f'- {item}')

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path
