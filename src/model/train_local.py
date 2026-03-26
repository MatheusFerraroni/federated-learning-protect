"""Implementa treino centralizado/local para causal LM, com suporte opcional a DP-SGD.

Responsabilidades principais:
- Construir optimizer e scheduler para treino padrão.
- Configurar baseline privada com Opacus quando ``config.dp.enabled`` estiver ativo.
- Executar treino por época tanto no modo não-privado quanto no modo privado.
- Expor utilitários reutilizáveis por sanity check centralizado e por clientes FL.
- Salvar relatórios e checkpoints do treino.

Como este arquivo se encaixa no projeto:
- É o núcleo de treinamento reutilizado pelo sanity check centralizado.
- Fornece a lógica base que também é reaproveitada no treinamento local dos clientes.
- Centraliza a baseline opcional com DP-SGD / Opacus para comparação experimental.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup
from tqdm import tqdm
from configs.experiment_config import DPConfig, ExperimentConfig
from configs.paths import CHECKPOINT_DIR, RESULTS_DIR
from src.model.dataset_utils import build_global_split_dataloader
from src.model.evaluate import evaluate_dataloader, evaluate_global_condition
from src.model.model_utils import get_device, move_batch_to_device, save_model_checkpoint
from src.utils.io import save_json
from src.utils.logging_utils import get_logger

try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
except ImportError:  # pragma: no cover
    PrivacyEngine = None
    BatchMemoryManager = None


LOGGER = get_logger(__name__)


def _to_float(value: torch.Tensor | float | int) -> float:
    """Converte tensores e escalares numéricos em ``float``.

    Args:
        value: Valor de entrada.

    Returns:
        Valor convertido para ``float``.

    Raises:
        TypeError: Se o valor não puder ser convertido.
    """
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def build_optimizer(
    model: PreTrainedModel,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Cria o optimizer AdamW com grupos decay/no-decay.

    Args:
        model: Modelo a ser otimizado.
        learning_rate: Learning rate do optimizer.
        weight_decay: Weight decay aplicado aos grupos elegíveis.

    Returns:
        Optimizer configurado.

    Raises:
        ValueError: Se hiperparâmetros forem inválidos.
    """
    if learning_rate <= 0:
        raise ValueError('learning_rate deve ser > 0.')
    if weight_decay < 0:
        raise ValueError('weight_decay deve ser >= 0.')

    no_decay = {
        'bias',
        'LayerNorm.weight',
        'layer_norm.weight',
        'ln_f.weight',
        'ln_1.weight',
        'ln_2.weight',
    }
    optimizer_grouped_parameters = [
        {
            'params': [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and not any(term in name for term in no_decay)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and any(term in name for term in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=learning_rate)


def build_scheduler(
    optimizer: AdamW,
    num_training_steps: int,
    warmup_ratio: float,
):
    """Cria scheduler linear com warmup.

    Args:
        optimizer: Optimizer associado.
        num_training_steps: Número total de steps.
        warmup_ratio: Fração de warmup.

    Returns:
        Scheduler compatível com Transformers.

    Raises:
        ValueError: Se os parâmetros forem inválidos.
    """
    if num_training_steps <= 0:
        raise ValueError('num_training_steps deve ser > 0.')
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError('warmup_ratio deve estar em [0.0, 1.0).')

    num_warmup_steps = int(num_training_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def configure_dp_trainable_layers(model: PreTrainedModel, dp_config: DPConfig) -> dict[str, Any]:
    """Congela parte do modelo para tornar a baseline DP viável em NLP.

    Args:
        model: Modelo causal LM carregado.
        dp_config: Configuração DP.

    Returns:
        Resumo da quantidade de parâmetros treináveis e congelados.

    Raises:
        ValueError: Se ``num_trainable_transformer_blocks`` for negativo.
    """
    if dp_config.num_trainable_transformer_blocks < 0:
        raise ValueError('num_trainable_transformer_blocks deve ser >= 0.')

    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_parameter_names: list[str] = []

    if dp_config.train_lm_head and hasattr(model, 'lm_head'):
        for name, parameter in model.lm_head.named_parameters(prefix='lm_head'):
            parameter.requires_grad = True
            trainable_parameter_names.append(name)

    transformer = getattr(model, 'transformer', None)

    if (
        transformer is not None
        and dp_config.train_final_layer_norm
        and hasattr(transformer, 'ln_f')
    ):
        for name, parameter in transformer.ln_f.named_parameters(prefix='transformer.ln_f'):
            parameter.requires_grad = True
            trainable_parameter_names.append(name)

    if transformer is not None and dp_config.freeze_embeddings is False:
        for attr_name in ('wte', 'wpe'):
            module = getattr(transformer, attr_name, None)
            if module is None:
                continue
            for name, parameter in module.named_parameters(prefix=f'transformer.{attr_name}'):
                parameter.requires_grad = True
                trainable_parameter_names.append(name)

    block_container = getattr(transformer, 'h', None) if transformer is not None else None
    if block_container is not None:
        total_blocks = len(block_container)
        start_index = max(0, total_blocks - dp_config.num_trainable_transformer_blocks)
        for block_index in range(start_index, total_blocks):
            block = block_container[block_index]
            for name, parameter in block.named_parameters(prefix=f'transformer.h.{block_index}'):
                parameter.requires_grad = True
                trainable_parameter_names.append(name)

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_parameters = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )

    summary = {
        'trainable_parameters': trainable_parameters,
        'frozen_parameters': frozen_parameters,
        'num_trainable_transformer_blocks': dp_config.num_trainable_transformer_blocks,
        'freeze_embeddings': dp_config.freeze_embeddings,
        'train_lm_head': dp_config.train_lm_head,
        'train_final_layer_norm': dp_config.train_final_layer_norm,
        'trainable_parameter_name_preview': sorted(trainable_parameter_names)[:25],
    }
    LOGGER.info('Configuração DP de camadas aplicada: %s', summary)
    return summary


def infer_dp_delta(dp_config: DPConfig, dataset_size: int) -> float:
    """Infere ``delta`` seguro para o dataset quando necessário.

    Args:
        dp_config: Configuração DP.
        dataset_size: Número de exemplos do dataset.

    Returns:
        Valor de delta usado na contabilidade.

    Raises:
        ValueError: Se ``dataset_size`` for inválido.
    """
    if dataset_size <= 0:
        raise ValueError('dataset_size deve ser > 0.')

    if dp_config.target_delta > 0:
        return float(dp_config.target_delta)

    denominator = max(dataset_size, dp_config.delta_denominator_floor)
    return 1.0 / float(denominator)


def _privacy_engine_available() -> bool:
    """Indica se Opacus está disponível no ambiente."""
    return PrivacyEngine is not None


def prepare_private_training(
    config: ExperimentConfig,
    model: PreTrainedModel,
    optimizer: AdamW,
    train_dataloader: DataLoader,
    num_epochs: int,
) -> tuple[PreTrainedModel, AdamW, DataLoader, Any, dict[str, Any]]:
    """Aplica ``PrivacyEngine`` a um trio ``(model, optimizer, dataloader)``.

    Args:
        config: Configuração agregada do experimento.
        model: Modelo a privatizar.
        optimizer: Optimizer base.
        train_dataloader: DataLoader lógico de treino.
        num_epochs: Número de épocas planejadas.

    Returns:
        Tupla ``(private_model, private_optimizer, private_loader,
        privacy_engine, privacy_metadata)``.

    Raises:
        ImportError: Se Opacus não estiver instalado.
        ValueError: Se a configuração DP for inválida.
    """
    if not _privacy_engine_available():
        raise ImportError(
            'Opacus não está instalado. Adicione a dependência ao ambiente antes de ativar config.dp.enabled.'
        )

    dp_config = config.dp
    dataset_size = len(train_dataloader.dataset)
    delta = infer_dp_delta(dp_config=dp_config, dataset_size=dataset_size)

    # Modelos HF carregados com from_pretrained normalmente ficam em eval().
    # O Opacus exige modo treino antes de make_private().
    model.train()

    privacy_engine = PrivacyEngine(secure_mode=dp_config.secure_mode)

    metadata: dict[str, Any] = {
        'enabled': True,
        'dataset_size': dataset_size,
        'delta': delta,
        'poisson_sampling': dp_config.poisson_sampling,
        'grad_sample_mode': dp_config.grad_sample_mode,
        'secure_mode': dp_config.secure_mode,
    }

    LOGGER.info(
        'Preparando treino privado com Opacus: dataset_size=%d delta=%s poisson_sampling=%s grad_sample_mode=%s',
        dataset_size,
        delta,
        dp_config.poisson_sampling,
        dp_config.grad_sample_mode,
    )

    if dp_config.target_epsilon is not None:
        private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_epsilon=float(dp_config.target_epsilon),
            target_delta=delta,
            epochs=num_epochs,
            max_grad_norm=float(dp_config.max_grad_norm),
            poisson_sampling=dp_config.poisson_sampling,
            grad_sample_mode=dp_config.grad_sample_mode,
        )
        metadata['target_epsilon'] = float(dp_config.target_epsilon)
        metadata['noise_multiplier'] = float(private_optimizer.noise_multiplier)
    else:
        private_model, private_optimizer, private_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=float(dp_config.noise_multiplier),
            max_grad_norm=float(dp_config.max_grad_norm),
            poisson_sampling=dp_config.poisson_sampling,
            grad_sample_mode=dp_config.grad_sample_mode,
        )
        metadata['noise_multiplier'] = float(dp_config.noise_multiplier)

    metadata['max_grad_norm'] = float(dp_config.max_grad_norm)
    LOGGER.info('Treino privado preparado: %s', metadata)
    return private_model, private_optimizer, private_loader, privacy_engine, metadata


def _resolve_training_batches_context(
    config: ExperimentConfig,
    dataloader: DataLoader,
) -> Any:
    """Resolve contexto de iteração com BatchMemoryManager quando necessário."""
    dp_config = config.dp
    if not dp_config.enabled or not dp_config.use_batch_memory_manager:
        return nullcontext(dataloader)

    if BatchMemoryManager is None:
        return nullcontext(dataloader)

    max_physical_batch_size = dp_config.max_physical_batch_size
    if max_physical_batch_size is None:
        return nullcontext(dataloader)
    if max_physical_batch_size <= 0:
        raise ValueError('config.dp.max_physical_batch_size deve ser > 0 quando definido.')

    logical_batch_size = getattr(dataloader, 'batch_size', None)
    if logical_batch_size is None or logical_batch_size <= max_physical_batch_size:
        return nullcontext(dataloader)

    optimizer = getattr(dataloader, 'optimizer', None)
    if optimizer is None:
        return nullcontext(dataloader)

    return BatchMemoryManager(
        data_loader=dataloader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer,
    )


def _extract_privacy_metrics(
    privacy_engine: Any | None,
    delta: float | None,
    optimizer: Any | None,
) -> dict[str, Any] | None:
    """Extrai métricas de contabilidade DP ao final do treino."""
    if privacy_engine is None or delta is None:
        return None

    epsilon_value: float | None = None
    try:
        epsilon_value = float(privacy_engine.get_epsilon(delta=delta))
    except Exception as exc:  # pragma: no cover
        LOGGER.warning('Falha ao calcular epsilon no Opacus: %s', exc)

    noise_multiplier = getattr(optimizer, 'noise_multiplier', None)
    max_grad_norm = getattr(optimizer, 'max_grad_norm', None)

    return {
        'epsilon': epsilon_value,
        'delta': float(delta),
        'noise_multiplier': None if noise_multiplier is None else float(noise_multiplier),
        'max_grad_norm': None if max_grad_norm is None else float(max_grad_norm),
    }


def train_one_epoch(
    model: PreTrainedModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    epoch_index: int = 0,
    dp_enabled: bool = False,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    """Treina uma época completa.

    Args:
        model: Modelo em treino.
        dataloader: Dataloader de treino.
        optimizer: Optimizer associado.
        scheduler: Scheduler associado.
        device: Device alvo.
        gradient_accumulation_steps: Acumulação de gradiente no modo não-DP.
        epoch_index: Índice da época atual.
        dp_enabled: Se o treino está usando Opacus.
        amp_enabled: Se deve ativar AMP em CUDA.

    Returns:
        Métricas da época.

    Raises:
        ValueError: Se parâmetros forem inválidos.
        RuntimeError: Se a loss vier nula.
    """
    if gradient_accumulation_steps <= 0:
        raise ValueError('gradient_accumulation_steps deve ser > 0.')
    if dp_enabled and gradient_accumulation_steps != 1:
        raise ValueError('gradient_accumulation_steps deve ser 1 quando DP-SGD estiver ativo.')

    model.train()

    total_loss = 0.0
    total_examples = 0
    total_batches = 0

    use_amp = bool(amp_enabled and device.type == 'cuda' and not dp_enabled)

    # Compatível com versões mais novas/antigas do PyTorch.
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        autocast_context = lambda: torch.amp.autocast('cuda', enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_context = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    optimizer.zero_grad(set_to_none=True)

    batch_bar = tqdm(
        dataloader,
        desc=f'Batches | epoch {epoch_index + 1}',
        position=1,
        leave=False,
        dynamic_ncols=True,
    )

    for step_index, batch in enumerate(batch_bar, start=1):
        batch = move_batch_to_device(batch, device)

        with autocast_context():
            outputs = model(**batch)
            raw_loss = outputs.loss

            if raw_loss is None:
                raise RuntimeError('Model retornou loss=None durante o treino.')

            loss = raw_loss / gradient_accumulation_steps

        batch_bar.set_postfix(loss=f'{raw_loss.detach().float().item():.4f}')

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        batch_size = int(batch['input_ids'].size(0))
        total_loss += _to_float(raw_loss) * batch_size
        total_examples += batch_size
        total_batches += 1

        should_step = dp_enabled or step_index % gradient_accumulation_steps == 0
        is_last_step = step_index == len(dataloader)

        if should_step or is_last_step:
            if not dp_enabled:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    if total_examples == 0:
        raise ValueError('Nenhum exemplo foi processado no treino.')

    metrics = {
        'epoch_index': epoch_index,
        'train_loss': total_loss / total_examples,
        'num_examples': total_examples,
        'num_batches': total_batches,
        'learning_rate_end': float(optimizer.param_groups[0]['lr']),
        'dp_enabled': dp_enabled,
        'amp_enabled': use_amp,
    }
    LOGGER.info('Epoch concluída: %s', metrics)
    return metrics


def train_model(
    config: ExperimentConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Executa treino completo com ou sem DP-SGD."""
    del tokenizer

    resolved_device = device or get_device()
    model.to(resolved_device)
    model.train()

    num_epochs = int(config.model.local_epochs)
    if num_epochs <= 0:
        raise ValueError('config.model.local_epochs deve ser > 0.')

    gradient_accumulation_steps = int(config.model.gradient_accumulation_steps)
    if config.dp.enabled and not config.dp.allow_grad_accumulation:
        gradient_accumulation_steps = 1

    trainable_summary = None
    if config.dp.enabled:
        trainable_summary = configure_dp_trainable_layers(model=model, dp_config=config.dp)

    total_training_steps = len(train_dataloader) * num_epochs
    optimizer = build_optimizer(
        model=model,
        learning_rate=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )

    privacy_engine = None
    privacy_metadata: dict[str, Any] | None = None
    if config.dp.enabled:
        model, optimizer, train_dataloader, privacy_engine, privacy_metadata = (
            prepare_private_training(
                config=config,
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                num_epochs=num_epochs,
            )
        )

    scheduler = build_scheduler(
        optimizer=optimizer,
        num_training_steps=max(1, total_training_steps),
        warmup_ratio=config.model.warmup_ratio,
    )

    history: list[dict[str, Any]] = []
    best_val_loss: float | None = None
    best_epoch_index: int | None = None

    LOGGER.info(
        'Iniciando treino: epochs=%d steps_per_epoch=%d total_steps=%d device=%s dp_enabled=%s',
        num_epochs,
        len(train_dataloader),
        total_training_steps,
        resolved_device,
        config.dp.enabled,
    )

    for epoch_index in tqdm(range(num_epochs), desc='Epochs', position=0):
        batch_context = _resolve_training_batches_context(
            config=config, dataloader=train_dataloader
        )
        with batch_context as epoch_loader:
            train_metrics = train_one_epoch(
                model=model,
                dataloader=epoch_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=resolved_device,
                gradient_accumulation_steps=gradient_accumulation_steps,
                epoch_index=epoch_index,
                dp_enabled=config.dp.enabled,
                amp_enabled=config.model.fp16,
            )

        epoch_report: dict[str, Any] = {
            'epoch_index': epoch_index,
            'train': train_metrics,
        }

        if val_dataloader is not None:
            val_metrics = evaluate_dataloader(
                model=model,
                dataloader=val_dataloader,
                device=resolved_device,
            )
            epoch_report['val'] = val_metrics

            current_val_loss = float(val_metrics['loss'])
            if best_val_loss is None or current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch_index = epoch_index

        history.append(epoch_report)

    privacy_report = None
    if config.dp.enabled and config.dp.log_privacy_metrics and privacy_metadata is not None:
        privacy_report = dict(privacy_metadata)
        privacy_report.update(
            _extract_privacy_metrics(
                privacy_engine=privacy_engine,
                delta=privacy_metadata.get('delta'),
                optimizer=optimizer,
            )
            or {}
        )
        privacy_report['trainable_layers'] = trainable_summary

    report: dict[str, Any] = {
        'training_config': {
            'seed': config.seed,
            'model': asdict(config.model),
            'dp': asdict(config.dp),
        },
        'history': history,
        'best_val_loss': best_val_loss,
        'best_epoch_index': best_epoch_index,
        'privacy_report': privacy_report,
    }
    return report


def build_train_and_val_dataloaders(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
    condition_name: str,
) -> tuple[DataLoader, DataLoader | None]:
    """Monta dataloaders globais de treino e validação.

    Permite que o split global de validação esteja vazio, retornando ``None``
    nesse caso. O split de treino continua obrigatório.
    """
    train_loader = build_global_split_dataloader(
        config=config,
        tokenizer=tokenizer,
        condition_name=condition_name,
        split_name='train',
        shuffle=True,
    )

    try:
        val_loader = build_global_split_dataloader(
            config=config,
            tokenizer=tokenizer,
            condition_name=condition_name,
            split_name='val',
            shuffle=False,
        )
    except ValueError as exc:
        if 'texts não pode ser vazio' not in str(exc):
            raise
        LOGGER.warning(
            'Split global de validação vazio; sanity seguirá sem val: condition=%s',
            condition_name,
        )
        val_loader = None

    return train_loader, val_loader


def train_on_global_condition(
    config: ExperimentConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    condition_name: str,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Executa treino centralizado completo para uma condição global."""
    resolved_device = device or get_device()

    train_loader, val_loader = build_train_and_val_dataloaders(
        config=config,
        tokenizer=tokenizer,
        condition_name=condition_name,
    )

    train_report = train_model(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=resolved_device,
    )

    evaluation_report = evaluate_global_condition(
        config=config,
        model=model,
        tokenizer=tokenizer,
        condition_name=condition_name,
        split_names=['train', 'val', 'test', 'domain_test'],
        device=resolved_device,
    )

    return {
        'condition': condition_name,
        'device': str(resolved_device),
        'train_report': train_report,
        'evaluation_report': evaluation_report,
        'validation_available': val_loader is not None,
    }


def resolve_run_output_dir(
    run_name: str,
    use_persistent_dir: bool = False,
) -> Path:
    """Resolve o diretório de saída do treino."""
    base_dir = CHECKPOINT_DIR if use_persistent_dir and CHECKPOINT_DIR is not None else RESULTS_DIR
    output_dir = Path(base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_training_run_artifacts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    run_report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Salva relatório e checkpoint de um treino."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / 'training_report.json'
    save_json(run_report, report_path)

    checkpoint_dir = output_path / 'checkpoint'
    save_model_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=checkpoint_dir,
        extra_state=run_report,
    )

    paths = {
        'output_dir': output_path,
        'report_path': report_path,
        'checkpoint_dir': checkpoint_dir,
    }
    LOGGER.info('Artefatos de treino salvos: %s', paths)
    return paths
