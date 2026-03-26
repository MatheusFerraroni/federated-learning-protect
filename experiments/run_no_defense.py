# """
# Script de execução ponta a ponta para a condição sem defesa.

# Responsabilidades principais:
# - Executar sanity check centralizado usando a condição raw.
# - Executar avaliação de vazamento no checkpoint do sanity check.
# - Executar treino federado com FedAvg usando a condição raw.
# - Executar avaliação de vazamento no checkpoint final federado.
# - Consolidar os artefatos e métricas em um único resumo.

# Como este arquivo se encaixa no projeto:
# - É o ponto de entrada prático para a condição [A] "Sem defesa",
#   permitindo reproduzir o pipeline mínimo do experimento de tese.
# """

# from __future__ import annotations

# import argparse
# from datetime import datetime

# from configs.paths import RESULTS_DIR
# from src.utils.experiment_runner import (
#     build_condition_summary,
#     find_latest_directory_by_prefix,
#     persist_summary,
#     run_python_module,
# )
# from src.utils.logging_utils import configure_logging, get_logger

# LOGGER = get_logger(__name__)


# def parse_args() -> argparse.Namespace:
#     """Lê argumentos de linha de comando.

#     Args:
#         Nenhum.

#     Returns:
#         Namespace com os parâmetros da execução.

#     Raises:
#         Não se aplica.
#     """
#     parser = argparse.ArgumentParser(
#         description='Executa o pipeline completo da condição sem defesa (raw).'
#     )
#     parser.add_argument('--seed', type=int, default=42, help='Seed global.')
#     parser.add_argument('--epochs', type=int, default=1, help='Épocas locais/centrais.')
#     parser.add_argument('--rounds', type=int, default=1, help='Número de rodadas federadas.')
#     parser.add_argument('--batch-size', type=int, default=4, help='Batch size de treino.')
#     parser.add_argument(
#         '--eval-batch-size',
#         type=int,
#         default=4,
#         help='Batch size de avaliação.',
#     )
#     parser.add_argument('--max-length', type=int, default=64, help='Comprimento máximo.')
#     parser.add_argument(
#         '--learning-rate',
#         type=float,
#         default=5e-5,
#         help='Learning rate.',
#     )
#     parser.add_argument(
#         '--num-prompts',
#         type=int,
#         default=100,
#         help='Número de prompts de ataque.',
#     )
#     parser.add_argument(
#         '--run-name',
#         type=str,
#         default=None,
#         help='Nome base opcional da execução.',
#     )
#     return parser.parse_args()


# def build_base_run_name(explicit_name: str | None) -> str:
#     """Resolve nome base da execução.

#     Args:
#         explicit_name: Nome fornecido pelo usuário.

#     Returns:
#         Nome base da execução.

#     Raises:
#         Não se aplica.
#     """
#     if explicit_name:
#         return explicit_name
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     return f'pipeline_raw_{timestamp}'


# def main() -> int:
#     """Executa o pipeline completo da condição raw.

#     Args:
#         Nenhum.

#     Returns:
#         Código de status do processo.

#     Raises:
#         RuntimeError: Se alguma etapa falhar.
#         FileNotFoundError: Se artefatos esperados não forem encontrados.
#     """
#     configure_logging()
#     args = parse_args()

#     condition = 'raw'
#     base_run_name = build_base_run_name(args.run_name)

#     output_dir = RESULTS_DIR / base_run_name
#     logs_dir = output_dir / 'logs'
#     summaries_dir = output_dir / 'summaries'
#     output_dir.mkdir(parents=True, exist_ok=True)

#     command_results = []

#     sanity_run_name = f'{base_run_name}_sanity'
#     command_results.append(
#         run_python_module(
#             module='experiments.run_sanity_check',
#             args=[
#                 '--condition',
#                 condition,
#                 '--epochs',
#                 str(args.epochs),
#                 '--batch-size',
#                 str(args.batch_size),
#                 '--eval-batch-size',
#                 str(args.eval_batch_size),
#                 '--max-length',
#                 str(args.max_length),
#                 '--learning-rate',
#                 str(args.learning_rate),
#                 '--seed',
#                 str(args.seed),
#                 '--run-name',
#                 sanity_run_name,
#             ],
#             logs_dir=logs_dir,
#             step_name='01_sanity_check',
#         )
#     )

#     sanity_dir = find_latest_directory_by_prefix(RESULTS_DIR, sanity_run_name)
#     sanity_checkpoint_dir = sanity_dir / 'checkpoint'

#     sanity_attack_output_dir = output_dir / 'attack_sanity_raw'
#     command_results.append(
#         run_python_module(
#             module='experiments.run_attack_eval',
#             args=[
#                 '--checkpoint-dir',
#                 str(sanity_checkpoint_dir),
#                 '--condition',
#                 condition,
#                 '--reference-condition',
#                 condition,
#                 '--num-prompts',
#                 str(args.num_prompts),
#                 '--seed',
#                 str(args.seed),
#                 '--output-dir',
#                 str(sanity_attack_output_dir),
#             ],
#             logs_dir=logs_dir,
#             step_name='02_attack_sanity',
#         )
#     )

#     federated_run_name = f'{base_run_name}_federated'
#     command_results.append(
#         run_python_module(
#             module='src.fl.run_federated',
#             args=[
#                 '--condition',
#                 condition,
#                 '--rounds',
#                 str(args.rounds),
#                 '--local-epochs',
#                 str(args.epochs),
#                 '--batch-size',
#                 str(args.batch_size),
#                 '--eval-batch-size',
#                 str(args.eval_batch_size),
#                 '--max-length',
#                 str(args.max_length),
#                 '--learning-rate',
#                 str(args.learning_rate),
#                 '--seed',
#                 str(args.seed),
#                 '--run-name',
#                 federated_run_name,
#             ],
#             logs_dir=logs_dir,
#             step_name='03_federated_train',
#         )
#     )

#     federated_dir = find_latest_directory_by_prefix(RESULTS_DIR, federated_run_name)
#     round_dirs = sorted(
#         [
#             path
#             for path in federated_dir.iterdir()
#             if path.is_dir() and path.name.startswith('round_')
#         ]
#     )
#     if not round_dirs:
#         raise FileNotFoundError(f'Nenhuma rodada encontrada em {federated_dir}')
#     federated_checkpoint_dir = round_dirs[-1] / 'checkpoint'

#     federated_attack_output_dir = output_dir / 'attack_federated_raw'
#     command_results.append(
#         run_python_module(
#             module='experiments.run_attack_eval',
#             args=[
#                 '--checkpoint-dir',
#                 str(federated_checkpoint_dir),
#                 '--condition',
#                 condition,
#                 '--reference-condition',
#                 condition,
#                 '--num-prompts',
#                 str(args.num_prompts),
#                 '--seed',
#                 str(args.seed),
#                 '--output-dir',
#                 str(federated_attack_output_dir),
#             ],
#             logs_dir=logs_dir,
#             step_name='04_attack_federated',
#         )
#     )

#     summary = build_condition_summary(
#         condition=condition,
#         sanity_dir=sanity_dir,
#         sanity_attack_dir=sanity_attack_output_dir,
#         federated_dir=federated_dir,
#         federated_attack_dir=federated_attack_output_dir,
#         command_results=command_results,
#     )

#     summary_path = persist_summary(summary, summaries_dir / 'raw_pipeline_summary.json')

#     LOGGER.info('Pipeline raw concluído. Resumo salvo em %s', summary_path)
#     return 0


# if __name__ == '__main__':
#     raise SystemExit(main())
