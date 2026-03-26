"""Utilitários de dataset e dataloaders para causal language modeling.

Responsabilidades principais:
- Resolver caminhos dos splits globais persistidos em JSONL.
- Carregar registros textuais e extrair o campo principal de texto.
- Tokenizar exemplos para causal LM com comprimento máximo configurável.
- Construir Dataset e DataLoader compatíveis com treino padrão e DP-SGD/Opacus.
- Aplicar padding, máscara de atenção e labels com ignore_index para causal LM.

Como este arquivo se encaixa no projeto:
- É a camada de preparação de dados usada por sanity check, treino local
  e treino federado.
- Mantém a interface de dados consistente entre condições experimentais.
- Foi ajustado para retornar tensores diretamente, evitando incompatibilidades
  do collate do Opacus com listas Python em batches vazios/amostragem Poisson.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from configs.experiment_config import ExperimentConfig, ModelConfig
from configs.paths import GLOBAL_SPLITS_DIR
from src.utils.io import load_jsonl
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)

VALID_SPLIT_NAMES = {'train', 'val', 'test', 'domain_test'}


def resolve_global_split_path(condition_name: str, split_name: str) -> Path:
    """Resolve o caminho de um split global salvo em disco.

    Args:
        condition_name: Nome da condição experimental.
        split_name: Nome do split. Deve estar em ``VALID_SPLIT_NAMES``.

    Returns:
        Caminho absoluto/relativo do arquivo JSONL correspondente.

    Raises:
        ValueError: Se ``split_name`` for inválido.
    """
    if split_name not in VALID_SPLIT_NAMES:
        raise ValueError(
            f'split_name inválido: {split_name}. Esperado um de {sorted(VALID_SPLIT_NAMES)}'
        )
    return GLOBAL_SPLITS_DIR / condition_name / f'{split_name}.jsonl'


def load_records_from_jsonl(path: str | Path) -> list[dict]:
    """Carrega uma lista de registros a partir de um arquivo JSONL.

    Args:
        path: Caminho do arquivo.

    Returns:
        Lista de registros.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {resolved_path}')
    return load_jsonl(resolved_path)


def load_global_split_records(condition_name: str, split_name: str) -> list[dict]:
    """Carrega os registros de um split global de uma condição.

    Args:
        condition_name: Nome da condição experimental.
        split_name: Nome do split.

    Returns:
        Lista de registros do split.

    Raises:
        ValueError: Se o split for inválido.
        FileNotFoundError: Se o arquivo não existir.
    """
    path = resolve_global_split_path(condition_name=condition_name, split_name=split_name)
    records = load_records_from_jsonl(path)
    LOGGER.info(
        'Carregado split global: condition=%s split=%s num_records=%d path=%s',
        condition_name,
        split_name,
        len(records),
        path,
    )
    return records


def extract_texts(records: Sequence[dict], text_key: str = 'text') -> list[str]:
    """Extrai textos válidos de uma sequência de registros.

    Args:
        records: Registros carregados do dataset.
        text_key: Chave que contém o texto bruto.

    Returns:
        Lista de textos normalizados e não vazios.

    Raises:
        KeyError: Se algum registro não contiver ``text_key``.
        TypeError: Se algum texto não for string.
    """
    texts: list[str] = []
    for index, record in enumerate(records):
        if text_key not in record:
            raise KeyError(f'Record na posição {index} não possui a chave {text_key!r}.')
        text = record[text_key]
        if not isinstance(text, str):
            raise TypeError(f'Record na posição {index} possui text não-string.')
        normalized = text.strip()
        if normalized:
            texts.append(normalized)
    return texts


@dataclass
class TokenizedSample:
    """Representa uma amostra tokenizada para causal LM.

    Args:
        input_ids: Tokens do exemplo.
        attention_mask: Máscara de atenção do exemplo.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class CausalLMDataset(Dataset):
    """Dataset tokenizado para causal language modeling.

    Args:
        texts: Lista de textos de entrada.
        tokenizer: Tokenizer do modelo causal LM.
        max_length: Comprimento máximo de tokenização.

    Returns:
        Não se aplica.

    Raises:
        ValueError: Se ``texts`` for vazio, ``max_length`` inválido ou nenhuma
            amostra válida for criada após tokenização.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        """Inicializa e tokeniza o dataset.

        Args:
            texts: Textos brutos.
            tokenizer: Tokenizer HF.
            max_length: Comprimento máximo por exemplo.

        Returns:
            Não se aplica.

        Raises:
            ValueError: Se os argumentos forem inválidos ou se nenhuma amostra
                válida for produzida.
        """
        if not texts:
            raise ValueError('texts não pode ser vazio.')
        if max_length <= 0:
            raise ValueError('max_length deve ser > 0.')

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[TokenizedSample] = []

        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
            )

            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']

            if not input_ids:
                continue

            self.samples.append(
                TokenizedSample(
                    input_ids=torch.tensor(input_ids, dtype=torch.long),
                    attention_mask=torch.tensor(attention_mask, dtype=torch.long),
                )
            )

        if not self.samples:
            raise ValueError('Nenhuma amostra válida foi criada após tokenização.')

    def __len__(self) -> int:
        """Retorna o número de amostras tokenizadas.

        Args:
            Nenhum.

        Returns:
            Quantidade de amostras.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Retorna uma amostra individual já em tensores.

        Args:
            index: Índice da amostra.

        Returns:
            Dicionário com ``input_ids`` e ``attention_mask`` em ``torch.long``.

        Raises:
            IndexError: Se o índice for inválido.
        """
        sample = self.samples[index]
        return {
            'input_ids': sample.input_ids,
            'attention_mask': sample.attention_mask,
        }


class CausalLMDataCollator:
    """Collator para causal LM compatível com padding manual em PyTorch.

    Args:
        tokenizer: Tokenizer do modelo.

    Returns:
        Não se aplica.

    Raises:
        ValueError: Se o tokenizer não tiver ``pad_token_id`` definido.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Inicializa o collator.

        Args:
            tokenizer: Tokenizer do modelo.

        Returns:
            Não se aplica.

        Raises:
            ValueError: Se ``pad_token_id`` não estiver definido.
        """
        if tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer precisa ter pad_token_id definido.')
        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Faz padding do batch e constrói labels para causal LM.

        Args:
            features: Lista de amostras individuais.

        Returns:
            Batch com ``input_ids``, ``attention_mask`` e ``labels``, todos em
            ``torch.long``. Tokens de padding recebem ``-100`` em ``labels``.

        Raises:
            ValueError: Se ``features`` for vazio.
            TypeError: Se algum campo não for tensor.
        """
        if not features:
            raise ValueError('features não pode ser vazio.')

        input_ids_list: list[torch.Tensor] = []
        attention_mask_list: list[torch.Tensor] = []

        for feature in features:
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']

            if not isinstance(input_ids, torch.Tensor):
                raise TypeError('feature["input_ids"] deve ser torch.Tensor.')
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError('feature["attention_mask"] deve ser torch.Tensor.')

            input_ids_list.append(input_ids.to(dtype=torch.long))
            attention_mask_list.append(attention_mask.to(dtype=torch.long))

        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0,
        )

        labels = input_ids.clone().to(dtype=torch.long)
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids.to(dtype=torch.long),
            'attention_mask': attention_mask.to(dtype=torch.long),
            'labels': labels.to(dtype=torch.long),
        }


def build_causal_lm_dataset(
    records: Sequence[dict],
    tokenizer: PreTrainedTokenizerBase,
    model_config: ModelConfig,
) -> CausalLMDataset:
    """Constrói o dataset tokenizado de causal LM.

    Args:
        records: Registros brutos.
        tokenizer: Tokenizer do modelo.
        model_config: Configuração do modelo.

    Returns:
        Instância de ``CausalLMDataset``.

    Raises:
        ValueError: Se nenhuma amostra válida for criada.
    """
    texts = extract_texts(records)
    dataset = CausalLMDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=model_config.max_length,
    )
    LOGGER.info(
        'Dataset causal LM criado: num_records=%d num_samples=%d max_length=%d',
        len(records),
        len(dataset),
        model_config.max_length,
    )
    return dataset


def build_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Constrói um DataLoader para treino/avaliação causal LM.

    Args:
        dataset: Dataset tokenizado.
        tokenizer: Tokenizer do modelo.
        batch_size: Tamanho do batch.
        shuffle: Se embaralha o dataset.
        num_workers: Número de workers do DataLoader.

    Returns:
        ``DataLoader`` configurado.

    Raises:
        ValueError: Se ``batch_size`` for inválido.
    """
    if batch_size <= 0:
        raise ValueError('batch_size deve ser > 0.')

    collator = CausalLMDataCollator(tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )


def build_global_split_dataloader(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
    condition_name: str,
    split_name: str,
    shuffle: bool = False,
) -> DataLoader:
    """Constrói um DataLoader para um split global persistido.

    Args:
        config: Configuração agregada do experimento.
        tokenizer: Tokenizer do modelo.
        condition_name: Nome da condição experimental.
        split_name: Nome do split.
        shuffle: Se embaralha o split.

    Returns:
        ``DataLoader`` pronto para treino ou avaliação.

    Raises:
        ValueError: Se parâmetros forem inválidos.
        FileNotFoundError: Se o split não existir.
    """
    records = load_global_split_records(condition_name=condition_name, split_name=split_name)
    dataset = build_causal_lm_dataset(
        records=records,
        tokenizer=tokenizer,
        model_config=config.model,
    )

    batch_size = config.model.batch_size if split_name == 'train' else config.model.eval_batch_size

    return build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.model.num_workers,
    )


def get_split_sizes(paths: Iterable[str | Path]) -> dict[str, int]:
    """Retorna o número de registros de múltiplos arquivos JSONL.

    Args:
        paths: Coleção de caminhos de arquivos JSONL.

    Returns:
        Dicionário ``{stem_do_arquivo: quantidade_de_registros}``.

    Raises:
        FileNotFoundError: Se algum arquivo não existir.
    """
    sizes: dict[str, int] = {}
    for path_like in paths:
        path = Path(path_like)
        records = load_records_from_jsonl(path)
        sizes[path.stem] = len(records)
    return sizes
