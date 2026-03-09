# CLIP Fine-Tuning com LoRA / SoRA

Script de treino para classificação de imagens usando a torre visual do CLIP (`openai/clip-vit-base-patch32`) com head linear e suporte a LoRA e SoRA (Sparse optimized LoRA).

O projeto usa como padrão o dataset `enterprise-explorers/oxford-pets` no Hugging Face e carrega todos os parâmetros a partir de um arquivo YAML.

## Requisitos

- Python 3
- Ambiente virtual recomendado
- Dependências listadas em `requirements.txt`

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuração

Os parâmetros de treino ficam em `train_config.yml`.

Exemplo completo:

```yaml
dataset:
  name: enterprise-explorers/oxford-pets

model:
  name: openai/clip-vit-base-patch32
  lora:
    mode: with_lora          # with_lora | without_lora | both | with_sora
    r: 16
    alpha: 32
    dropout: 0.05
    bias: none
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - out_proj
  sora:
    sparse_lambda: 10        # peso do L1 na loss total (CE + lambda * sparse)
    sparse_lambda_2: 3.0e-4  # lambda inicial do SparseAdamW (soft-thresholding)
    sparse_lr: null           # LR separado para gates (null = usa optimizer.lr)
    lambda_schedule: null     # null | linear | log_linear | exp_linear
    max_lambda: null          # lambda final do schedule
    lambda_num: null          # número de fases do schedule
    schedule_epochs: 15       # épocas por fase do schedule

training:
  batch_size: 32
  epochs: 3
  test_size: 0.2
  seed: 42

optimizer:
  lr: 0.0005
  weight_decay: 0.01

scheduler:
  step_size: 1
  gamma: 0.7

output:
  weights_path: clip_pets_finetuned.pth
```

## Execução

Rodar com a configuração padrão:

```bash
python3 clip_train.py
```

Rodar com outro YAML:

```bash
python3 clip_train.py --config outro_arquivo.yml
```

## Modos de treino

O campo `model.lora.mode` aceita:

| Modo | Descrição |
|------|-----------|
| `with_lora` | Treina com LoRA (via PEFT) |
| `without_lora` | Treina só o head de classificação, sem adaptadores |
| `both` | Executa dois treinos em sequência: com LoRA e sem |
| `with_sora` | Treina com SoRA — LoRA esparso com gates e proximal step |

Quando o modo for `both`, o script salva dois checkpoints separados:
`clip_pets_finetuned_with_lora.pth` e `clip_pets_finetuned_without_lora.pth`.

## SoRA (Sparse optimized LoRA)

O SoRA adiciona gates multiplicativos nos adaptadores LoRA. Um `SparseAdamW` aplica soft-thresholding nos gates a cada step, zerando os que ficam abaixo do limiar (`sparse_lambda_2`). Isso produz adaptadores esparsos — menos parâmetros ativos com menor perda de acurácia.

A seção `model.sora` só é usada quando `mode: with_sora`.

### SoRA sem schedule (lambda fixo)

O modo mais simples. O lambda do soft-thresholding é fixo durante todo o treino.

```yaml
model:
  lora:
    mode: with_sora
    r: 16
    alpha: 32
    dropout: 0.05
    bias: none
    target_modules: [q_proj, k_proj, v_proj, out_proj]
  sora:
    sparse_lambda: 10
    sparse_lambda_2: 3.0e-4
    sparse_lr: null
    lambda_schedule: null     # <-- sem schedule
    max_lambda: null
    lambda_num: null
```

Nesse modo o treino roda `training.epochs` épocas com um único valor de lambda.

### SoRA com schedule (Algorithm 1 do paper)

O lambda do soft-thresholding cresce ao longo de múltiplas fases, forçando esparsidade progressiva. Após as `training.epochs` épocas iniciais, o script executa `lambda_num - 1` fases adicionais, cada uma com `schedule_epochs` épocas e um lambda maior.

```yaml
model:
  lora:
    mode: with_sora
    r: 16
    alpha: 32
    dropout: 0.05
    bias: none
    target_modules: [q_proj, k_proj, v_proj, out_proj]
  sora:
    sparse_lambda: 10
    sparse_lambda_2: 3.0e-4   # lambda inicial
    sparse_lr: null
    lambda_schedule: linear    # linear | log_linear | exp_linear
    max_lambda: 1.0            # lambda final
    lambda_num: 5              # número total de lambdas (gera 4 fases extras)
    schedule_epochs: 15        # épocas por fase
```

Os tipos de schedule disponíveis:

| Schedule | Comportamento |
|----------|---------------|
| `linear` | Lambda cresce linearmente de `sparse_lambda_2` até `max_lambda` |
| `log_linear` | Interpolação linear no espaço log (cresce rápido no início, desacelera) |
| `exp_linear` | Interpolação linear no espaço exponencial (cresce devagar no início, acelera) |

## Métricas de logging

O script imprime métricas por época. As métricas exibidas dependem do modo:

**Todos os modos** (`with_lora`, `without_lora`, `with_sora`):

| Métrica | Descrição |
|---------|-----------|
| `CE` | Cross-entropy loss (sem componente sparse) |
| `Total` | Loss total (CE + lambda * sparse, ou igual a CE se não for SoRA) |
| `Acc` | Acurácia no conjunto de validação |
| `LR` | Learning rate atual |

**Apenas `with_sora`** (adicionadas ao final da linha):

| Métrica | Descrição |
|---------|-----------|
| `Sparse` | L1 loss média dos gates (normalizada pelo total de parâmetros) |
| `Sparsity` | Percentual de gates que são exatamente zero |
| `λ` | Lambda atual do SparseAdamW (soft-thresholding) |

Exemplo de saída para `with_sora`:

```
[with_sora] Epoch 1/3 - CE: 3.6743 - Total: 3.7802 - Acc: 12.50% - LR: 0.000350 - Sparse: 0.0106 - Sparsity: 4.2% (27/640) - λ: 0.0003
```

Exemplo de saída para `with_lora`:

```
[with_lora] Epoch 1/3 - CE: 3.6743 - Total: 3.6743 - Acc: 15.30% - LR: 0.000350
```

## Saída

O checkpoint gerado por padrão é:

```bash
clip_pets_finetuned.pth
```

Esse tipo de artefato está ignorado no git por `*.pth` em `.gitignore`.

## Observações

- O primeiro treino faz download do dataset e do modelo base.
- Sem `HF_TOKEN`, o Hugging Face pode aplicar limites mais baixos de download.
- A seleção de dispositivo é automática: `cuda` quando disponível, senão `cpu`.
