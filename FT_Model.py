import torch
from torch.utils.data import DataLoader
# from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM,AdamW, get_linear_schedule_with_warmup


# Define hyperparameters
learning_rate = 2e-05
train_batch_size = 32
eval_batch_size = 16
seed = 42
num_epochs = 5

# Set random seed for reproducibility
torch.manual_seed(seed)

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)


# Load dataset
dataset = load_dataset("openwebtext100k")
train_dataloader = DataLoader(dataset["train"], batch_size=train_batch_size, shuffle=True)
eval_dataloader = DataLoader(dataset["test"], batch_size=eval_batch_size)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)
        labels = inputs["input_ids"].clone()
        labels.to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}")

# Evaluation loop
model.eval()
total_eval_loss = 0.0
with torch.no_grad():
    for batch in tqdm(eval_dataloader, desc="Evaluating", unit="batch"):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)
        labels = inputs["input_ids"].clone()
        labels.to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_eval_loss += loss.item()
avg_eval_loss = total_eval_loss / len(eval_dataloader)
print(f"Average evaluation loss: {avg_eval_loss:.4f}")
