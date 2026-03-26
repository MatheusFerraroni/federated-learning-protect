"""Centraliza caminhos persistentes de dados, clientes e resultados.

Responsabilidades principais:
- Resolver o PROJECT_ROOT a partir das settings.
- Declarar caminhos usados por geração de dataset, splits globais,
  clientes honestos, clientes maliciosos e resultados.
- Expor diretórios de checkpoints persistentes quando houver DRIVE_ROOT.

Como este arquivo se encaixa no projeto:
- Evita strings de caminho duplicadas no restante do projeto.
- Garante consistência entre scripts, treino federado e avaliação.
"""

from __future__ import annotations

from pathlib import Path

from configs.default import settings


PROJECT_ROOT = Path(settings.PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

CLIENT_DATA_DIR = DATA_DIR / 'clients'
CLIENT_RAW_DATA_DIR = CLIENT_DATA_DIR / 'raw'
CLIENT_SEMANTIC_DATA_DIR = CLIENT_DATA_DIR / 'semantic_substitution'
CLIENT_ATTACK_RAW_DATA_DIR = CLIENT_DATA_DIR / 'attack_raw'
CLIENT_ATTACK_SEMANTIC_DATA_DIR = CLIENT_DATA_DIR / 'attack_semantic_substitution'
CLIENT_METADATA_DIR = CLIENT_DATA_DIR / 'metadata'
CLIENT_SUMMARIES_DIR = CLIENT_DATA_DIR / 'summaries'

GLOBAL_SPLITS_DIR = DATA_DIR / 'global_splits'

RESULTS_DIR = PROJECT_ROOT / 'results'
SRC_DIR = PROJECT_ROOT / 'src'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'

CHECKPOINT_DIR = None
PERSISTENT_RESULTS_DIR = None

if settings.DRIVE_ROOT is not None:
    DRIVE_ROOT = Path(settings.DRIVE_ROOT)
    CHECKPOINT_DIR = DRIVE_ROOT / 'checkpoints'
    PERSISTENT_RESULTS_DIR = DRIVE_ROOT / 'results'
