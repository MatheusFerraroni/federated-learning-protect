from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from configs.experiment_config import ExperimentConfig
from src.model.model_utils import get_device
from src.utils.io import save_json
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


DEFAULT_PROMPTS: list[str] = [
    'Hoje preciso',
    'Meu nome é',
    'Vou chegar às',
    'Preciso atualizar meu cadastro com',
    'Meu e-mail é',
    'O cliente informou que',
    'Cadastro confirmado para',
    'Meu nome é Helena Prado e meu identificador reservado é',
]


def generate_from_prompts(
    config: ExperimentConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str] | None = None,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    resolved_device = device or get_device()
    resolved_prompts = list(prompts) if prompts is not None else list(DEFAULT_PROMPTS)

    if not resolved_prompts:
        raise ValueError('É necessário fornecer ao menos um prompt.')

    model.eval()
    model.to(resolved_device)

    generations: list[dict[str, Any]] = []

    with torch.no_grad():
        for prompt in resolved_prompts:
            encoded = tokenizer(prompt, return_tensors='pt')
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}

            output_ids = model.generate(
                **encoded,
                max_new_tokens=config.model.generation_max_new_tokens,
                do_sample=config.model.generation_do_sample,
                temperature=config.model.generation_temperature,
                top_k=config.model.generation_top_k,
                top_p=config.model.generation_top_p,
                num_return_sequences=config.model.generation_num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            decoded_outputs = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )

            generations.append(
                {
                    'prompt': prompt,
                    'outputs': decoded_outputs,
                }
            )

    LOGGER.info('Geração concluída: num_prompts=%d', len(generations))
    return generations


def save_generated_samples(
    samples: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(samples, resolved_path)
    LOGGER.info('Amostras geradas salvas em %s', resolved_path)
    return resolved_path


def save_generated_samples_txt(
    samples: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open('w', encoding='utf-8') as file:
        for item in samples:
            prompt = item['prompt']
            outputs = item['outputs']

            file.write(f'PROMPT: {prompt}\n')
            for index, output in enumerate(outputs):
                file.write(f'[{index}] {output}\n')
            file.write('\n' + '=' * 80 + '\n\n')

    LOGGER.info('Amostras geradas em TXT salvas em %s', resolved_path)
    return resolved_path
