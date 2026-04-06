# CLIP Fine-Tuning com LoRA / SoRA (Modular & Optimized)

Este projeto realiza o ajuste fino (fine-tuning) da torre visual do CLIP (`openai/clip-vit-base-patch32`) para classificação de imagens, utilizando técnicas avançadas de adaptação de baixo rank (LoRA) e adaptação esparsa otimizada (SoRA).

Esta versão foi reestruturada para ser **modular, eficiente e pronta para produção**, incluindo otimizações de hardware e compressão de modelos.

## Novidades nesta Versão

- **Arquitetura Modular**: Código organizado em módulos (`src/`) com separação clara de responsabilidades (Model, Data, Training, Optimization).
- **Performance (SDPA)**: Utiliza *Scaled Dot Product Attention* nativo do PyTorch para redução drástica de VRAM e aumento de velocidade.
- **Estratégia PaCA**: Suporte para aplicação de adaptadores apenas nas camadas superiores (*upper layers*) do encoder.
- **Compressão de Modelo**:
    - **Poda Estrutural (Pruning)**: Converte automaticamente SoRA esparso em LoRA denso e menor após o treino.
    - **Quantização INT8**: Reduz o tamanho dos pesos salvos em até 4x com perda mínima de precisão.
    - **Extração de Adapters**: Salva apenas os pesos treinados (LoRA + Head), ideal para ambientes de banda limitada ou Federated Learning.

## Estrutura do Projeto

```
src/
├── main.py          # Ponto de entrada e orquestração do pipeline
├── clip_setup.py    # Configurações de modelo, dados e utilitários de compressão
├── trainer.py       # Classe ModelTrainer (gerenciamento do ciclo de vida do treino)
└── sora.py          # Implementação core do SoRA e algoritmos de poda
```

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

Para iniciar o treinamento com as novas otimizações:

```bash
python3 src/main.py --config train_config.yml
```

## Configurações Avançadas (YAML)

O arquivo `train_config.yml` suporta novas chaves:

```yaml
model:
  name: openai/clip-vit-base-patch32
  paca:
    enabled: true
    upper_layers: 4          # Aplica SoRA apenas nas últimas 4 camadas
  lora:
    mode: with_sora_schedule # with_lora | without_lora | both | with_sora_no_schedule | with_sora_schedule
    r: 16
    alpha: 32
  sora:
    pre_prune_ratio: 0.1     # Poda 10% do backbone original antes de começar
    sparse_lambda: 10
    # ... outras configs de schedule
```

## Ciclo de Vida do Modelo

1. **Setup**: O modelo é carregado com SDPA e, opcionalmente, sofre uma pré-poda do backbone.
2. **Benchmark**: Um teste rápido de latência e VRAM é executado automaticamente.
3. **Treino**: O `ModelTrainer` executa as épocas, monitorando a esparsidade dos gates SoRA.
4. **Finalização**:
    - **Pruning**: O modelo SoRA é "compactado" em um LoRA de rank menor.
    - **Extraction**: Apenas os pesos modificados são extraídos.
    - **Quantization**: Os pesos são convertidos para INT8.
5. **Exportação**: O arquivo `.pth` final contém um modelo otimizado e extremamente leve.

## Modos de Treino (LoRA/SoRA)

| Modo | Descrição |
|------|-----------|
| `with_lora` | Treina com adaptadores LoRA padrão. |
| `without_lora` | Treina apenas o cabeçalho de classificação. |
| `with_sora_no_schedule` | SoRA com fator de esparsidade fixo. |
| `with_sora_schedule` | SoRA com esparsidade progressiva (Algorithm 1 do paper). |

## Créditos e Referências

O SoRA implementa os conceitos do paper *SoRA: Sparse Low-Rank Adaptation*, focando em reduzir a redundância de parâmetros em adaptadores de modelos de larga escala.
