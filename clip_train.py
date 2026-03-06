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


DEFAULT_CONFIG_PATH = Path("train_config.yml")
VALID_LORA_MODES = {"with_lora", "without_lora", "both"}


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


def build_model(config, num_classes, device):
    model_config = config["model"]
    lora_config = model_config["lora"]

    clip_model = CLIPModel.from_pretrained(model_config["name"])
    model = CLIPForClassification(clip_model, num_classes)

    if lora_config["mode"] == "with_lora":
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["dropout"],
            bias=lora_config["bias"],
        )
        model.clip.vision_model = get_peft_model(model.clip.vision_model, peft_config)
        model.clip.vision_model.print_trainable_parameters()

    model.to(device)
    return model


def resolve_run_modes(config):
    lora_mode = config["model"]["lora"]["mode"]
    if lora_mode == "both":
        return [("with_lora", True), ("without_lora", False)]
    return [(lora_mode, lora_mode == "with_lora")]


def build_run_config(config, use_lora):
    run_config = yaml.safe_load(yaml.safe_dump(config))
    run_config["model"]["lora"]["mode"] = "with_lora" if use_lora else "without_lora"
    return run_config


def build_output_path(base_output_path, run_mode, multiple_runs):
    output_path = Path(base_output_path)
    if not multiple_runs:
        return output_path

    suffix = "_with_lora" if run_mode == "with_lora" else "_without_lora"
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def build_optimizer(model, config):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check the LoRA and classifier setup.")

    optimizer_config = config["optimizer"]
    return AdamW(
        trainable_params,
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
    )


def build_scheduler(optimizer, config):
    scheduler_config = config["scheduler"]
    return StepLR(
        optimizer,
        step_size=scheduler_config["step_size"],
        gamma=scheduler_config["gamma"],
    )


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for pixel_values, labels in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


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
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    total_epochs = config["training"]["epochs"]

    print(f"Starting run: {run_mode}")
    for epoch in range(total_epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        eval_acc = evaluate(model, eval_loader)
        scheduler.step()
        print(
            f"[{run_mode}] Epoch {epoch + 1}/{total_epochs} - "
            f"Train Loss: {train_loss:.4f} - Eval Accuracy: {eval_acc * 100:.2f}%"
        )

    return model


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    processor = CLIPProcessor.from_pretrained(config["model"]["name"])
    train_loader, eval_loader, class_names = build_dataloaders(config, processor, device)
    run_modes = resolve_run_modes(config)
    multiple_runs = len(run_modes) > 1

    for run_mode, use_lora in run_modes:
        run_config = build_run_config(config, use_lora=use_lora)
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
