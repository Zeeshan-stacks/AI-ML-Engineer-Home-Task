# multi_task_demorun.py

from transformers import AutoTokenizer, AutoModel
import torch
from multi_task_model import MultiTaskSentenceTransformer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
transformer = AutoModel.from_pretrained("distilbert-base-uncased")
model = MultiTaskSentenceTransformer(transformer)
model.eval()

sentences = [
    "Hey, How are you doing?",
    "Please hold these files for me.",
    "Designs of new I phones are attractive."
]

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(inputs["input_ids"], inputs["attention_mask"])

# printing outputs
for i, sent in enumerate(sentences):
    print(f"\nSentence: {sent}")
    print("Task A logits (e.g., Statement/Question/Command):", outputs["task_a_logits"][i].numpy())
    print("Task B logits (e.g., Sentiment):", outputs["task_b_logits"][i].numpy())
