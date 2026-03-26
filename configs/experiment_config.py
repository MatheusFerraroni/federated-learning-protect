"""Define as configurações centrais do experimento federado de vazamento.

Responsabilidades principais:
- Declarar dataclasses de configuração para dataset, defesa, modelo, FL,
  ataque de extração, cliente malicioso e baseline DP-SGD com Opacus.
- Centralizar hiperparâmetros usados por geração de dados, particionamento,
  treino local, treino federado e avaliação de leakage.
- Expor uma configuração default consistente para todo o projeto.

Como este arquivo se encaixa no projeto:
- É a principal fonte de verdade para parâmetros reproduzíveis.
- Todos os módulos devem depender destas dataclasses, evitando parâmetros
  espalhados em scripts e funções.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    """Configuração de geração e particionamento do dataset sintético.

    Args:
        num_clients: Número de clientes honestos.
        samples_per_client: Quantidade de exemplos por cliente honesto.
        t1_ratio: Proporção de textos cotidianos T1.
        t2_ratio: Proporção de textos pessoais T2.
        t3_ratio: Proporção de textos estruturados T3.
        canary_ratio: Fração aproximada de canários no dataset.
        repeated_ratio: Fração aproximada de registros repetidos.
        train_ratio: Proporção do split de treino.
        val_ratio: Proporção do split de validação.
        test_ratio: Proporção do split de teste.
        domain_test_ratio: Fração do split de teste clonada para domain_test.
        min_sensitive_ratio_per_client: Mínimo de exemplos sensíveis por cliente.
        client_style_skew_strength: Intensidade do não-IID de estilo textual.
        client_entity_skew_strength: Intensidade do não-IID por tipos de entidade.
        canary_bias_strength: Intensidade de viés de canários por cliente.
        output_dataset_name: Prefixo padrão do dataset global.
        condition_names: Condições experimentais persistidas em disco.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    num_clients: int = 3
    samples_per_client: int = 80

    t1_ratio: float = 0.10
    t2_ratio: float = 0.40
    t3_ratio: float = 0.50

    canary_ratio: float = 0.08
    repeated_ratio: float = 0.40

    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    domain_test_ratio: float = 0.10

    min_sensitive_ratio_per_client: float = 0.90
    client_style_skew_strength: float = 0.10
    client_entity_skew_strength: float = 0.10
    canary_bias_strength: float = 0.60

    output_dataset_name: str = 'synthetic_global_dataset'
    condition_names: List[str] = field(
        default_factory=lambda: [
            'raw',
            'semantic_substitution',
            'attack_raw',
            'attack_semantic_substitution',
        ]
    )


@dataclass
class DefenseConfig:
    """Configuração da defesa local baseada em substituição semântica.

    Args:
        enabled: Indica se a defesa está habilitada.
        strategy: Estratégia de defesa aplicada.
        protect_canaries: Se verdadeiro, canários também são substituídos.
        keep_original_entities: Se verdadeiro, mantém entidades originais em metadados.
        fail_on_missing_entity_in_text: Falha se nenhuma substituição textual ocorrer.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    enabled: bool = True
    strategy: str = 'semantic_substitution'
    protect_canaries: bool = True
    keep_original_entities: bool = False
    fail_on_missing_entity_in_text: bool = False


@dataclass
class DPConfig:
    """Configuração da baseline DP-SGD com Opacus.

    Args:
        enabled: Se ativa treino privado com Opacus.
        noise_multiplier: Multiplicador do ruído Gaussiano. Ignorado quando
            ``target_epsilon`` for usado com ``make_private_with_epsilon``.
        max_grad_norm: Limite de clipping por exemplo.
        target_epsilon: Orçamento-alvo opcional. Quando informado, o Opacus
            estima automaticamente ``noise_multiplier``.
        target_delta: Delta alvo da garantia DP. Valor típico: 1 / N.
        poisson_sampling: Se usa amostragem de Poisson no dataloader privado.
        secure_mode: Se ativa modo seguro do gerador de ruído.
        grad_sample_mode: Modo de captura de gradientes por exemplo.
        max_physical_batch_size: Batch físico opcional para BatchMemoryManager.
        allow_grad_accumulation: Se permite acumulação de gradientes no modo DP.
        freeze_embeddings: Se congela embeddings durante a baseline DP.
        train_lm_head: Se mantém a camada de saída treinável.
        train_final_layer_norm: Se mantém a layer norm final treinável.
        num_trainable_transformer_blocks: Quantos blocos finais do transformer
            permanecem treináveis. 0 significa congelar todos os blocos.
        delta_denominator_floor: Piso do denominador ao inferir delta do dataset.
        log_privacy_metrics: Se registra epsilon e ruído nos relatórios.
        use_batch_memory_manager: Se usa BatchMemoryManager quando possível.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    enabled: bool = False
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    target_epsilon: Optional[float] = None
    target_delta: float = 1e-5
    poisson_sampling: bool = False
    secure_mode: bool = False
    grad_sample_mode: str = 'hooks'
    max_physical_batch_size: Optional[int] = None
    allow_grad_accumulation: bool = False
    freeze_embeddings: bool = True
    train_lm_head: bool = True
    train_final_layer_norm: bool = True
    num_trainable_transformer_blocks: int = 2
    delta_denominator_floor: int = 1000
    log_privacy_metrics: bool = True
    use_batch_memory_manager: bool = True


@dataclass
class ModelConfig:
    """Configuração do modelo causal LM e do treinamento local.

    Args:
        model_name: Nome do modelo no Hugging Face Hub.
        max_length: Comprimento máximo tokenizado.
        batch_size: Batch size de treino.
        eval_batch_size: Batch size de avaliação.
        learning_rate: Learning rate do otimizador.
        weight_decay: Weight decay do AdamW.
        warmup_ratio: Fração de warmup do scheduler.
        local_epochs: Número de épocas locais/centralizadas.
        gradient_accumulation_steps: Passos de acumulação de gradiente.
        num_workers: Workers do DataLoader.
        fp16: Reservado para extensões futuras.
        use_cache_during_training: Se ativa cache do modelo durante treino.
        generation_max_new_tokens: Tokens máximos em geração.
        generation_do_sample: Se usa sampling.
        generation_temperature: Temperatura de geração.
        generation_top_k: Top-k de geração.
        generation_top_p: Top-p de geração.
        generation_num_return_sequences: Quantidade de sequências por prompt.
        save_model_checkpoint: Se checkpoints devem ser salvos.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    model_name: str = 'pierreguillou/gpt2-small-portuguese'

    max_length: int = 128
    batch_size: int = 2
    eval_batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    local_epochs: int = 4
    gradient_accumulation_steps: int = 1
    num_workers: int = 0

    fp16: bool = False
    use_cache_during_training: bool = False

    generation_max_new_tokens: int = 96
    generation_do_sample: bool = True
    generation_temperature: float = 0.1
    generation_top_k: int = 1
    generation_top_p: float = 1.0
    generation_num_return_sequences: int = 20

    save_model_checkpoint: bool = True


@dataclass
class FederatedConfig:
    """Configuração do loop federado manual com FedAvg.

    Args:
        num_rounds: Número de rodadas federadas.
        clients_per_round: Número total de clientes selecionados por rodada.
        local_epochs: Épocas locais por rodada.
        sample_malicious_separately: Se verdadeiro, amostra honestos e maliciosos separadamente.
        max_malicious_clients_per_round: Limite superior de maliciosos por rodada.
        min_malicious_clients_per_round: Limite inferior de maliciosos por rodada.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    num_rounds: int = 8
    clients_per_round: int = 3
    local_epochs: int = 4
    sample_malicious_separately: bool = True
    max_malicious_clients_per_round: int = 1
    min_malicious_clients_per_round: int = 1


@dataclass
class AttackConfig:
    """Configuração do ataque de extração por prompting.

    Args:
        num_prompts: Número total de prompts de ataque.
        max_generation_tokens: Tokens máximos gerados por prompt.
        generation_do_sample: Se usa sampling.
        generation_temperature: Temperatura da geração.
        generation_top_k: Top-k da geração.
        generation_top_p: Top-p da geração.
        generation_num_return_sequences: Número de sequências por prompt.
        prompt_seed: Seed da amostragem de prompts.
        direct_attack_ratio: Fração de prompts de prefixo direto.
        partial_attack_ratio: Fração de prompts de prefixo parcial.
        canary_attack_ratio: Fração de prompts focados em canários.
        partial_match_min_ratio: Similaridade mínima para partial match.
        max_prompt_characters: Comprimento máximo de prompt em caracteres.
        min_prompt_characters: Comprimento mínimo de prompt em caracteres.
        enable_reference_eval: Se avalia contra um conjunto de referência.
        enable_train_eval: Se avalia contra o conjunto de treino usado para prompts.
        restrict_same_type_matching: Restringe matching ao mesmo tipo de entidade.
        ignore_short_numeric_for_partial: Ignora numéricos curtos em partial match.
        save_generations_jsonl: Se salva gerações em JSONL.
        save_report_json: Se salva relatório em JSON.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    num_prompts: int = 300
    max_generation_tokens: int = 128

    generation_do_sample: bool = True
    generation_temperature: float = 0.1
    generation_top_k: int = 1
    generation_top_p: float = 1.0
    generation_num_return_sequences: int = 20
    generation_batch_size: int = 8

    prompt_seed: int = 1234
    direct_attack_ratio: float = 0.80
    partial_attack_ratio: float = 0.10
    canary_attack_ratio: float = 0.10

    partial_match_min_ratio: float = 0.60
    max_prompt_characters: int = 240
    min_prompt_characters: int = 8

    enable_reference_eval: bool = True
    enable_train_eval: bool = True
    restrict_same_type_matching: bool = True
    ignore_short_numeric_for_partial: bool = True
    save_generations_jsonl: bool = True
    save_report_json: bool = True


@dataclass
class MaliciousConfig:
    """Configuração explícita do cliente malicioso e do poisoning.

    Args:
        enabled: Se ativa geração de clientes maliciosos.
        num_malicious_clients: Quantos clientes maliciosos criar.
        malicious_client_prefix: Prefixo dos IDs dos clientes maliciosos.
        samples_per_malicious_client: Quantidade de amostras por cliente malicioso.
        poisoning_ratio: Fração do dataset local do atacante dedicada ao poisoning.
        min_poisoning_records: Piso absoluto de exemplos envenenados por atacante.
        trigger_repetition_factor: Fator de repetição dos gatilhos.
        trigger_types: Tipos de gatilho usados no poisoning.
        target_entity_types: Tipos de entidade alvo do ataque.
        prefer_canaries: Prioriza canários no catálogo-alvo de avaliação.
        num_targets_per_client: Quantos alvos de referência registrar por atacante.
        max_targets_pool: Tamanho máximo do catálogo de alvos extraído dos honestos.
        allow_target_reuse: Se um mesmo alvo pode ser reutilizado.
        fraction_malicious_selected_per_round: Fração desejada de maliciosos por rodada.
        save_attack_metadata: Se salva metadados de ataque nos registros.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    enabled: bool = True
    num_malicious_clients: int = 1
    malicious_client_prefix: str = 'attacker_'
    samples_per_malicious_client: int = 500

    poisoning_ratio: float = 0.90
    min_poisoning_records: int = 300
    trigger_repetition_factor: int = 10

    trigger_types: List[str] = field(
        default_factory=lambda: [
            'reserved_identifier',
            'structured_profile',
            'confirmed_registration',
            'client_report',
        ]
    )
    target_entity_types: List[str] = field(
        default_factory=lambda: [
            'secret_token',
            'cpf',
            'email',
            'rg',
            'passport',
            'name',
        ]
    )

    prefer_canaries: bool = True
    num_targets_per_client: int = 15
    max_targets_pool: int = 50
    allow_target_reuse: bool = True

    fraction_malicious_selected_per_round: float = 1.0
    save_attack_metadata: bool = True


@dataclass
class PartitionConfig:
    """Configuração do particionamento dos dados entre clientes.

    Args:
        shuffle_before_split: Embaralha antes do particionamento.
        save_global_splits: Se salva visões globais agregadas por condição.
        enforce_all_clients_have_sensitive_data: Garante piso de dados sensíveis por cliente honesto.
        allow_capacity_overflow: Reservado para extensões futuras.
        save_attack_conditions: Se persiste também as condições com atacante.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    shuffle_before_split: bool = True
    save_global_splits: bool = True
    enforce_all_clients_have_sensitive_data: bool = True
    allow_capacity_overflow: bool = False
    save_attack_conditions: bool = True


@dataclass
class ExperimentConfig:
    """Configuração agregada do experimento.

    Args:
        seed: Seed global reprodutível.
        dataset: Configuração do dataset.
        defense: Configuração da defesa.
        dp: Configuração da baseline DP-SGD com Opacus.
        model: Configuração do modelo.
        federated: Configuração federada.
        attack: Configuração do ataque de extração.
        malicious: Configuração do atacante.
        partition: Configuração de particionamento.

    Returns:
        Não se aplica.

    Raises:
        Não se aplica.
    """

    seed: int = 42

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    malicious: MaliciousConfig = field(default_factory=MaliciousConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)


def get_default_config() -> ExperimentConfig:
    """Retorna a configuração padrão do projeto.

    Args:
        Nenhum.

    Returns:
        Instância default de ExperimentConfig.

    Raises:
        Não se aplica.
    """
    return ExperimentConfig()
