# Project SoRA-Pets

This project provides a highly optimized pipeline for fine-tuning the vision encoder of CLIP (`openai/clip-vit-base-patch32`) for image classification. It employs advanced adaptation techniques such as **LoRA** (Low-Rank Adaptation), **SoRA** (Sparse Low-Rank Adaptation), and **PLD** (Progressive Layer Dropping) to achieve high performance with minimal parameter updates and memory footprint.

## Key Technologies
- **Frameworks**: PyTorch, Hugging Face Transformers, PEFT, Datasets.
- **Optimization**: Scaled Dot Product Attention (SDPA), INT8 Quantization, Structural Pruning.
- **Hardware Support**: CUDA, MPS (Metal Performance Shaders), and CPU.

## Architecture
The project is organized into a modular `src/` directory:
- `src/main.py`: The central orchestration script for the entire pipeline.
- `src/clip_setup.py`: Contains utility functions for model construction, data loading, optimizer setup, and model patching.
- `src/trainer.py`: Implements the `ModelTrainer` class, managing the training lifecycle, evaluation, and benchmarking.
- `src/sora.py`: Core implementation of Sparse Low-Rank Adaptation (SoRA) and structural pruning algorithms.
- `src/pld.py`: Implements the `PLDScheduler` for Progressive Layer Dropping.

## Building and Running

### Environment Setup
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
To start the fine-tuning process using the default configuration:
```bash
python3 src/main.py
```

### Configuration
Training is controlled via a YAML configuration file (e.g., `src/config/train_config.yml`). Key sections include:
- `dataset`: Name of the dataset (e.g., `enterprise-explorers/oxford-pets`).
- `model`: Model name, LoRA/SoRA settings, PaCA (Partial Calibration) layers, and PLD limits.
- `training`: Batch size, epochs, test size, and seed.
- `optimizer`/`scheduler`: Learning rates, weight decay, and decay steps.

## Development Conventions

### Coding Style & Standards
- **Modular Design**: Keep logic separated into specialized modules.
- **Configuration-Driven**: All hyperparameters and run modes should be defined in the YAML config.
- **Documentation**: Use descriptive docstrings (PT-BR is currently prevalent in the codebase) for classes and major functions.
- **Type Hinting**: Utilize Python type hints for clarity.

### Testing & Validation
- **Benchmarking**: The project includes automatic benchmarks for attention (SDPA) and inference performance (latency/throughput).
- **Metrics**: Training history is automatically saved to `plot/training_metrics.yml`.
- **Validation**: Accuracy is validated after each epoch using `scikit-learn`.

### Post-Processing
- **Pruning**: SoRA models are automatically pruned to a dense LoRA format post-training.
- **Quantization**: Final weights are quantized to INT8 to reduce storage size (~4x reduction).
- **Extraction**: Only trainable parameters (adapters and head) are saved to the final `.pth` file.
