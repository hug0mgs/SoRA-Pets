# Imports organizados
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn

# Configurações globais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregamento do dataset e splits
dataset = load_dataset("enterprise-explorers/oxford-pets")
split_dataset = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Extração de classes e prompts (prompts opcionais para zero-shot, mas mantidos para completude)
class_names = sorted(set(dataset["train"]["label"]))
num_classes = len(class_names)

def get_text_prompts(class_names):
    return [f"a photo of a {name}" for name in class_names]

text_prompts = get_text_prompts(class_names)

# Carregamento do modelo e processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

# Tokenização de prompts (para zero-shot, se necessário)
tokenized_texts = processor(text=text_prompts, padding=True, return_tensors="pt").to(device)

# Função de collate para DataLoader
def collate_fn(batch):
    images = [item["image"] for item in batch]
    labels = [class_names.index(item["label"]) for item in batch]  # Mapeia strings para índices numéricos
    inputs = processor(images=images, return_tensors="pt", padding=True)
    return inputs.pixel_values.to(device), torch.tensor(labels).to(device)

# Criação dos DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Classe custom para classificação: Usa vision_model do CLIP + head linear
class CLIPForClassification(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Linear(self.clip.vision_model.config.hidden_size, num_classes)
        
        # Congela o backbone por default para eficiência
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, pixel_values, labels=None):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output  # (batch, hidden_size)
        logits = self.classifier(pooled_output)  # (batch, num_classes)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_classes), labels.view(-1))
        
        return {"logits": logits, "loss": loss}

# Instancia o modelo custom, reutilizando o CLIP pré-carregado
model = CLIPForClassification(model, num_classes)
model.to(device)

# Configuração opcional de LoRA para adaptação eficiente
use_lora = True  # Mude para False se quiser treinar só o head
if use_lora:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # Módulos específicos do CLIPAttention
        lora_dropout=0.05,
        bias="none",
    )
    model.clip.vision_model = get_peft_model(model.clip.vision_model, lora_config)
    
    # Verificação opcional de parâmetros treináveis (para depuração)
    model.print_trainable_parameters()

# Otimizador e scheduler (apenas nos parâmetros treináveis)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=5e-4, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Função para treinamento de uma época
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for pixel_values, labels in tqdm(loader):
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Função para avaliação
def evaluate(model, loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for pixel_values, labels in tqdm(loader):
            outputs = model(pixel_values=pixel_values)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            true_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(true_labels, preds)
    return acc

# Loop principal de treinamento
num_epochs = 3  # Ajuste conforme necessário
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    eval_acc = evaluate(model, test_loader, device)
    scheduler.step()
    print(f"Época {epoch+1}/{num_epochs} - Loss Treino: {train_loss:.4f} - Acurácia Eval: {eval_acc * 100:.2f}%")

# Salva o modelo treinado
torch.save(model.state_dict(), "clip_pets_finetuned.pth")
# Opcional: Push para o Hugging Face Hub (requer huggingface-cli login)
# model.clip.push_to_hub("seu-repo")