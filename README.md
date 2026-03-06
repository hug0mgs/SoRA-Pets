# CLIP Fine-Tuning com LoRA

Script de treino para classificação de imagens usando a torre visual do CLIP (`openai/clip-vit-base-patch32`) com head linear e suporte opcional a LoRA.

O projeto usa como padrão o dataset `enterprise-explorers/oxford-pets` no Hugging Face e carrega todos os parâmetros a partir de um arquivo YAML.

## Requisitos

- Python 3
- Ambiente virtual recomendado
- Dependências listadas em [requirements.txt](/home/wallace/Projetcs/rafael/requirements.txt)

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuração

Os parâmetros de treino ficam em [train_config.yml](/home/wallace/Projetcs/rafael/train_config.yml).

Exemplo de estrutura:

```yaml
dataset:
  name: enterprise-explorers/oxford-pets

model:
  name: openai/clip-vit-base-patch32
  lora:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
    bias: none
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - out_proj

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

## O que o script faz

- baixa o dataset configurado no Hugging Face
- divide `train` em treino e validação
- carrega o `CLIPProcessor` e o `CLIPModel`
- congela o backbone do CLIP
- aplica LoRA na vision tower quando `model.lora.enabled: true`
- treina o head de classificação
- avalia a acurácia ao fim de cada época
- salva os pesos finais em `output.weights_path`

## Saída

O checkpoint gerado por padrão é:

```bash
clip_pets_finetuned.pth
```

Esse tipo de artefato está ignorado no git por `*.pth` em [.gitignore](/home/wallace/Projetcs/rafael/.gitignore).

## Observações

- O primeiro treino faz download do dataset e do modelo base.
- Sem `HF_TOKEN`, o Hugging Face pode aplicar limites mais baixos de download.
- A seleção de dispositivo é automática: `cuda` quando disponível, senão `cpu`.
