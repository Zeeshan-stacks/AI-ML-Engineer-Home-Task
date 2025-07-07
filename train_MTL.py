#train_mtl.py

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from multi_task_model import MultiTaskSentenceTransformer
import random

# testing data
sentences = [
    "Hello, How Are you?",
    "I am on my way to office.",
    "Cutomer service should be quick and satisficing.",
    "How can I get the refund.",
    "what an amazing product it is."
]

#labeling the statements 
LABELS_TASK_A = [0, 1, 2]  ## 0-statement, 1-question, 2-command
LABELS_TASK_B = [0, 1, 2]  ## 0-negative, 1-neutral, 2-positive

def generate_mock_labels(num):
    task_a = torch.tensor(random.choices(LABELS_TASK_A, k=num))
    task_b = torch.tensor(random.choices(LABELS_TASK_B, k=num))
    return task_a, task_b

# envoirment setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
transformer = AutoModel.from_pretrained("distilbert-base-uncased")

model = MultiTaskSentenceTransformer(transformer).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# loop for traning model 
model.train()
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
labels_a, labels_b = generate_mock_labels(len(sentences))

for epoch in range(1):
    optimizer.zero_grad()
    outputs = model(inputs["input_ids"], inputs["attention_mask"])

    loss_a = criterion(outputs["task_a_logits"], labels_a.to(device))
    loss_b = criterion(outputs["task_b_logits"], labels_b.to(device))
    loss = loss_a + loss_b  # sum of losses 

    print(f"Epoch {epoch+1} | Loss A: {loss_a.item():.4f} | Loss B: {loss_b.item():.4f} | Total Loss: {loss.item():.4f}")

    # accuracy calculation 
    preds_a = torch.argmax(outputs["task_a_logits"], dim=1)
    preds_b = torch.argmax(outputs["task_b_logits"], dim=1)

    acc_a = (preds_a == labels_a.to(device)).float().mean().item()
    acc_b = (preds_b == labels_b.to(device)).float().mean().item()

    print(f"Accuracy Task A: {acc_a:.2%} | Accuracy Task B: {acc_b:.2%}")
    print("-" * 50)

    loss.backward()
    optimizer.step()
