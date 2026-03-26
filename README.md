# Federated Learning Privacy Leakage Benchmark

Este repositório implementa um pipeline **completo, reprodutível e modular** para avaliar **vazamento de informação (privacy leakage)** em modelos de linguagem treinados em ambiente federado.

O sistema cobre:

- Geração de dataset sintético com entidades sensíveis
- Treinamento centralizado (sanity check)
- Treinamento federado (FedAvg)
- Ataque de extração via prompting
- Defesa por substituição semântica
- Baseline com Differential Privacy (Opacus)
- Consolidação e visualização de resultados

---

# Estrutura do Projeto

:contentReference[oaicite:0]{index=0}

Principais diretórios:

- `configs/` → Configuração centralizada (dataclasses)
- `data/` → Dataset e splits por condição
- `experiments/` → Scripts executáveis (entrypoints)
- `src/` → Implementação principal
- `results/` → Outputs dos experimentos

---

# 🧠 Pipeline Experimental

O pipeline completo é composto por **4 condições experimentais**:

| Condição | Descrição |
|--------|----------|
| A | Sem atacante (`raw`) |
| B | Com atacante (`attack_raw`) |
| C | Atacante + defesa semântica (`attack_semantic_substitution`) |
| D | Atacante + DP-SGD (`attack_raw_dp`) |

Cada condição executa:

1. Sanity check (treino centralizado)
2. Attack evaluation (checkpoint inicial)
3. Treino federado
4. Attack evaluation (checkpoint final)
5. Geração de summary estruturado

---

# Métricas de Leakage (Atualizado)

Após as correções, o pipeline usa métricas **consistentes e normalizadas**:

### Métricas principais

- `exact_match_rate`
  - Taxa de recuperação exata de segredos da referência

- `partial_match_rate`
  - Recuperação parcial (similaridade ≥ threshold)

- `structured_entity_generation_rate`
  - Geração de entidades estruturadas (CPF, email, etc.)

- `canary_recovery_rate`
  - Recuperação de canários inseridos

---

### Mudanças importantes (corrigido)

- Métricas agora são **agregadas corretamente por registro**
- `aggregate_attack_records` gera taxas reais (mean)
- `_normalize_attack_metrics` unifica nomes inconsistentes
- Alias conflitantes foram removidos
- `None` ≠ `0.0` (tratamento corrigido no plot)

---

# Estrutura dos Resultados

Cada execução gera:

```

results/
└── <run_name>/
├── logs/
├── summaries/
├── attack_sanity_*/
├── attack_federated_*/
└── comparison/

```

### Arquivo principal de cada condição

```

summaries/<condition>_pipeline_summary.json

````

Contém:

```json
{
  "condition": "...",
  "sanity": {...},
  "federated": {...},
  "attack_metrics": {
    "exact_match_rate": float,
    "partial_match_rate": float,
    "canary_recovery_rate": float,
    "structured_entity_generation_rate": float
  }
}
````

---

# Como Executar

## 1. Gerar dataset

```bash
python -m experiments.run_generate_dataset
```

---

## 2. Executar pipeline completo

```bash
python -m experiments.run_all \
  --epochs 1 \
  --rounds 1 \
  --num-prompts 100
```

---

## 3. Gerar relatório final

```bash
python -m experiments.run_results_report \
  --run-prefix <nome_do_run>
```

---

# Attack Evaluation

O ataque é baseado em prompting:

* Prefixos diretos
* Prefixos parciais
* Ataques focados em canários

Executado via:

```bash
python -m experiments.run_attack_eval \
  --checkpoint-dir <path> \
  --condition attack_raw
```

---

# Componentes Principais

## Dataset

* `src/data/generate_dataset.py`
* Geração de dados sintéticos com:

  * Entidades sensíveis
  * Canários
  * Não-IID entre clientes

---

## Federated Learning

* `src/fl/run_federated.py`
* Implementação simples de FedAvg

---

## Ataque

* `src/model/attack.py`
* Pipeline completo:

  * Geração
  * Matching (exato / parcial)
  * Métricas

---

## Métricas

* `src/utils/metrics.py`

Responsável por:

* Avaliação por geração
* Agregação por dataset
* Cálculo das taxas finais

---

## Relatórios e plots

* `src/utils/plots.py`
* `experiments/run_results_report.py`

---

#  Configuração

Centralizada em:

```
configs/experiment_config.py
```

Inclui:

* DatasetConfig
* ModelConfig
* FederatedConfig
* AttackConfig
* MaliciousConfig
* DPConfig

---

# Reprodutibilidade

* Seed global configurável
* Pipeline determinístico
* Config única centralizada
* Outputs versionados por timestamp

---

# Limitações conhecidas

* Pequena escala (toy setup)
* Modelo leve (GPT2 small)
* Ataque baseado apenas em prompting
* Não inclui ataques gradiente-based

---

# Próximos passos sugeridos

* Escalar número de clientes
* Aumentar diversidade de ataques
* Avaliar modelos maiores
* Integrar métricas de utilidade mais robustas

---

# Estado atual do projeto

✔ Pipeline executável ponta a ponta
✔ Métricas consistentes e corrigidas
✔ Comparação entre 4 condições
✔ Relatório automático com tabelas e gráficos

---

# Licença

Ver `LICENSE.md`