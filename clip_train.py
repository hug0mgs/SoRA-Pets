import argparse

import torch
import torch.nn as nn
from datasets import ClassLabel, load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


DEFAULT_DATASET = "enterprise-explorers/oxford-pets"
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_OUTPUT = "clip_pets_finetuned.pth"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP vision encoder for classification.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Pretrained CLIP model name.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save the trained weights.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for train and eval.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer weight decay.")
    parser.add_argument("--scheduler-gamma", type=float, default=0.7, help="StepLR gamma.")
    parser.add_argument("--scheduler-step-size", type=int, default=1, help="StepLR step size.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/validation split.")
    parser.add_argument(
        "--disable-lora",
        action="store_true",
        help="Train only the classifier head and keep the CLIP backbone frozen.",
    )
    return parser.parse_args()


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


def build_dataloaders(dataset_name, batch_size, test_size, seed, processor, device):
    dataset = load_dataset(dataset_name)
    split_dataset = dataset["train"].train_test_split(test_size=test_size, shuffle=True, seed=seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    class_names, label_to_idx = resolve_label_metadata(dataset["train"])

    collate_fn = make_collate_fn(processor, label_to_idx, device)
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


def build_model(model_name, num_classes, device, use_lora):
    clip_model = CLIPModel.from_pretrained(model_name)
    model = CLIPForClassification(clip_model, num_classes)

    if use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model.clip.vision_model = get_peft_model(model.clip.vision_model, lora_config)
        model.clip.vision_model.print_trainable_parameters()

    model.to(device)
    return model


def build_optimizer(model, lr, weight_decay):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check the LoRA and classifier setup.")
    return AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


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


def main():
    args = parse_args()
    device = get_device()

    processor = CLIPProcessor.from_pretrained(args.model_name)
    train_loader, eval_loader, class_names = build_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        test_size=args.test_size,
        seed=args.seed,
        processor=processor,
        device=device,
    )

    model = build_model(
        model_name=args.model_name,
        num_classes=len(class_names),
        device=device,
        use_lora=not args.disable_lora,
    )
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(
        optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
    )

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        eval_acc = evaluate(model, eval_loader)
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} - Eval Accuracy: {eval_acc * 100:.2f}%"
        )

    torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    main()
