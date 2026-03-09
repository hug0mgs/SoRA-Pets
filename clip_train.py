import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from datasets import ClassLabel, load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from sora import SoRAWrappedLinear, SparseAdamW


DEFAULT_CONFIG_PATH = Path("train_config.yml")
VALID_LORA_MODES = {"with_lora", "without_lora", "both", "with_sora_no_schedule", "with_sora_schedule"}
SORA_MODES = {"with_sora_no_schedule", "with_sora_schedule"}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP vision encoder for classification.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("The YAML config must define a mapping at the top level.")

    required_keys = {"dataset", "model", "training", "optimizer", "scheduler", "output"}
    missing_keys = required_keys - config.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Missing required config sections: {missing}")

    lora_config = config["model"].get("lora", {})
    lora_mode = lora_config.get("mode")
    if lora_mode is None:
        if "enabled" in lora_config:
            lora_mode = "with_lora" if lora_config["enabled"] else "without_lora"
        else:
            lora_mode = "with_lora"
        lora_config["mode"] = lora_mode

    if lora_mode not in VALID_LORA_MODES:
        expected = ", ".join(sorted(VALID_LORA_MODES))
        raise ValueError(f"Invalid model.lora.mode: {lora_mode!r}. Expected one of: {expected}")

    return config


def resolve_label_metadata(dataset_split):
    label_feature = dataset_split.features.get("label")
    raw_labels = dataset_split["label"]
    unique_labels = sorted(set(raw_labels))

    if isinstance(label_feature, ClassLabel):
        class_names = list(label_feature.names)
        label_to_idx = {idx: idx for idx in range(len(class_names))}
        label_to_idx.update({name: idx for idx, name in enumerate(class_names)})
        return class_names, label_to_idx

    if unique_labels and isinstance(unique_labels[0], str):
        class_names = unique_labels
        label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        return class_names, label_to_idx

    class_names = [str(label) for label in unique_labels]
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_to_idx.update({str(label): idx for idx, label in enumerate(unique_labels)})
    return class_names, label_to_idx


def encode_label(label, label_to_idx):
    if isinstance(label, torch.Tensor):
        label = label.item()

    if label in label_to_idx:
        return label_to_idx[label]

    string_label = str(label)
    if string_label in label_to_idx:
        return label_to_idx[string_label]

    raise KeyError(f"Unknown label value: {label!r}")


def make_collate_fn(processor, label_to_idx, device):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        labels = [encode_label(item["label"], label_to_idx) for item in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        return pixel_values, label_tensor

    return collate_fn


def build_dataloaders(config, processor, device):
    dataset_config = config["dataset"]
    training_config = config["training"]

    dataset = load_dataset(dataset_config["name"])
    split_dataset = dataset["train"].train_test_split(
        test_size=training_config["test_size"],
        shuffle=True,
        seed=training_config["seed"],
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    class_names, label_to_idx = resolve_label_metadata(dataset["train"])

    collate_fn = make_collate_fn(processor, label_to_idx, device)
    batch_size = training_config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, eval_loader, class_names


class CLIPForClassification(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip = clip_model
        self.num_classes = num_classes
        hidden_size = self.clip.vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, pixel_values, labels=None):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return {"logits": logits, "loss": loss}


def apply_sora(model, lora_config):
    """Replace target linear modules with SoRAWrappedLinear in the vision model."""
    target_modules = lora_config["target_modules"]
    r = lora_config["r"]
    alpha = lora_config["alpha"]
    dropout = lora_config["dropout"]

    # Collect targets before modifying the module tree
    modules_map = dict(model.clip.vision_model.named_modules())
    replacements = []
    for name in list(modules_map.keys()):
        for target in target_modules:
            if not name.endswith(target):
                continue
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = modules_map[parts[0]]
                child_name = parts[1]
            else:
                parent = model.clip.vision_model
                child_name = parts[0]

            original_linear = getattr(parent, child_name)
            if isinstance(original_linear, nn.Linear):
                replacements.append((parent, child_name, original_linear))

    for parent, child_name, original_linear in replacements:
        wrapped = SoRAWrappedLinear(original_linear, r=r, lora_alpha=alpha, lora_dropout=dropout)
        setattr(parent, child_name, wrapped)

    print(f"SoRA applied to {len(replacements)} modules")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")


def build_model(config, num_classes, device):
    model_config = config["model"]
    lora_config = model_config["lora"]

    clip_model = CLIPModel.from_pretrained(model_config["name"])
    model = CLIPForClassification(clip_model, num_classes)

    mode = lora_config["mode"]
    if mode == "with_lora":
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["dropout"],
            bias=lora_config["bias"],
        )
        model.clip.vision_model = get_peft_model(model.clip.vision_model, peft_config)
        model.clip.vision_model.print_trainable_parameters()
    elif mode in SORA_MODES:
        apply_sora(model, lora_config)

    model.to(device)
    return model


def resolve_run_modes(config):
    lora_mode = config["model"]["lora"]["mode"]
    if lora_mode == "both":
        return ["with_lora", "without_lora"]
    return [lora_mode]


def build_run_config(config, run_mode):
    run_config = yaml.safe_load(yaml.safe_dump(config))
    run_config["model"]["lora"]["mode"] = run_mode
    return run_config


def build_output_path(base_output_path, run_mode, multiple_runs):
    output_path = Path(base_output_path)
    if not multiple_runs:
        return output_path

    return output_path.with_name(f"{output_path.stem}_{run_mode}{output_path.suffix}")


def build_optimizer(model, config):
    optimizer_config = config["optimizer"]
    sora_config = config["model"].get("sora", {})
    mode = config["model"]["lora"]["mode"]
    is_sora = mode in SORA_MODES

    if is_sora:
        gate_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "sora" in name and "gate" in name:
                gate_params.append(param)
            else:
                other_params.append(param)

        if not other_params and not gate_params:
            raise ValueError("No trainable parameters found.")

        main_optimizer = AdamW(
            other_params,
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
        )

        sparse_lr = sora_config.get("sparse_lr") or optimizer_config["lr"]

        if mode == "with_sora_schedule":
            lambda_schedule = sora_config.get("lambda_schedule")
            max_lambda = sora_config.get("max_lambda")
            lambda_num = sora_config.get("lambda_num")
        else:
            lambda_schedule = None
            max_lambda = None
            lambda_num = None

        sparse_optimizer = SparseAdamW(
            sparse_lambda=sora_config.get("sparse_lambda_2", 3e-4),
            lambda_schedule=lambda_schedule,
            max_lambda=max_lambda,
            lambda_num=lambda_num,
            params=gate_params,
            lr=sparse_lr,
            weight_decay=0.0,
        )

        print(f"Gate params: {sum(p.numel() for p in gate_params):,}")
        print(f"Other trainable params: {sum(p.numel() for p in other_params):,}")
        return main_optimizer, sparse_optimizer

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check the LoRA and classifier setup.")

    return AdamW(
        trainable_params,
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
    ), None


def build_scheduler(optimizer, config):
    scheduler_config = config["scheduler"]
    return StepLR(
        optimizer,
        step_size=scheduler_config["step_size"],
        gamma=scheduler_config["gamma"],
    )


def train_epoch(model, loader, optimizer, sparse_optimizer=None, sparse_lambda=0.0):
    model.train()
    total_ce_loss = 0.0
    total_sparse_loss = 0.0
    total_loss = 0.0

    for pixel_values, labels in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        if sparse_optimizer is not None:
            sparse_optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values, labels=labels)
        ce_loss = outputs["loss"]
        loss = ce_loss

        sparse_loss_val = 0.0
        if sparse_optimizer is not None and sparse_lambda > 0:
            gate_params = [p for n, p in model.named_parameters() if "sora" in n and "gate" in n]
            sparse_loss = sum(torch.sum(torch.abs(p)) for p in gate_params)
            p_total = sum(p.numel() for p in gate_params)
            if p_total > 0:
                sparse_loss_val = sparse_loss.item() / p_total
                loss = ce_loss + sparse_lambda * sparse_loss / p_total

        loss.backward()
        optimizer.step()
        if sparse_optimizer is not None:
            sparse_optimizer.step()

        total_ce_loss += ce_loss.item()
        total_sparse_loss += sparse_loss_val
        total_loss += loss.item()

    n = max(len(loader), 1)
    return {
        "ce_loss": total_ce_loss / n,
        "sparse_loss": total_sparse_loss / n,
        "total_loss": total_loss / n,
    }


def compute_gate_sparsity(model):
    """Return (num_zero, num_total) across all gate params."""
    total = 0
    zeros = 0
    for n, p in model.named_parameters():
        if "sora" in n and "gate" in n:
            total += p.numel()
            zeros += (p.data == 0).sum().item()
    return zeros, total


def evaluate(model, loader):
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for pixel_values, labels in tqdm(loader, desc="eval", leave=False):
            outputs = model(pixel_values=pixel_values)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            true_labels.extend(labels.cpu().numpy())

    return accuracy_score(true_labels, preds)


def run_training(config, train_loader, eval_loader, class_names, device):
    run_mode = config["model"]["lora"]["mode"]
    model = build_model(config, num_classes=len(class_names), device=device)
    optimizer, sparse_optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    sparse_scheduler = build_scheduler(sparse_optimizer, config) if sparse_optimizer is not None else None

    sora_config = config["model"].get("sora", {})
    is_sora = run_mode in SORA_MODES
    sparse_lambda = sora_config.get("sparse_lambda", 0.0) if is_sora else 0.0

    total_epochs = config["training"]["epochs"]

    def _run_epochs(num_epochs, phase_label=""):
        for epoch in range(num_epochs):
            metrics = train_epoch(model, train_loader, optimizer, sparse_optimizer, sparse_lambda)
            eval_acc = evaluate(model, eval_loader)
            scheduler.step()
            if sparse_scheduler is not None:
                sparse_scheduler.step()
            label = f"[{run_mode}]{phase_label}"
            current_lr = scheduler.get_last_lr()[0]

            line = (
                f"{label} Epoch {epoch + 1}/{num_epochs} - "
                f"CE: {metrics['ce_loss']:.4f} - "
                f"Total: {metrics['total_loss']:.4f} - "
                f"Acc: {eval_acc * 100:.2f}% - "
                f"LR: {current_lr:.6f}"
            )

            if is_sora:
                zeros, total_gates = compute_gate_sparsity(model)
                sparsity = zeros / total_gates * 100 if total_gates > 0 else 0
                current_lambda = sparse_optimizer.sparse_lambda
                line += (
                    f" - Sparse: {metrics['sparse_loss']:.4f}"
                    f" - Sparsity: {sparsity:.1f}% ({zeros}/{total_gates})"
                    f" - \u03bb: {current_lambda}"
                )

            print(line)

    print(f"Starting run: {run_mode}")
    _run_epochs(total_epochs)

    # Lambda schedule phases (Algorithm 1 from the SoRA paper)
    if run_mode == "with_sora_schedule" and sparse_optimizer is not None:
        lambda_num = sora_config.get("lambda_num")
        schedule_epochs = sora_config.get("schedule_epochs", total_epochs)

        if lambda_num is not None:
            for phase in range(1, lambda_num):
                sparse_optimizer.step_lambda()
                print(f"\n--- Schedule phase {phase}/{lambda_num - 1}, lambda={sparse_optimizer.sparse_lambda} ---")
                _run_epochs(schedule_epochs, phase_label=f" [phase {phase}]")

    return model


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    processor = CLIPProcessor.from_pretrained(config["model"]["name"])
    train_loader, eval_loader, class_names = build_dataloaders(config, processor, device)
    run_modes = resolve_run_modes(config)
    multiple_runs = len(run_modes) > 1

    for run_mode in run_modes:
        run_config = build_run_config(config, run_mode=run_mode)
        model = run_training(run_config, train_loader, eval_loader, class_names, device)
        output_path = build_output_path(
            config["output"]["weights_path"],
            run_mode=run_mode,
            multiple_runs=multiple_runs,
        )
        torch.save(model.state_dict(), output_path)
        print(f"Saved weights to {output_path}")


if __name__ == "__main__":
    main()
