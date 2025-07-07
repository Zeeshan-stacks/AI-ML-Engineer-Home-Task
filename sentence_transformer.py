# transformer for sentence.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

# loadig tokenizer and pretrained transformer
MODEL= "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
transformer = AutoModel.from_pretrained(MODEL)

# sentence encoding module
class SentenceEncoder(nn.Module):
    def __init__(self, transformer, embedding_dim=768):
        super(SentenceEncoder, self).__init__()
        self.transformer = transformer
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.post_linear = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # shape:(B, T, H)
        
        # transpose for pooling (B, H, T)
        pooled = self.pooling(last_hidden.permute(0, 2, 1)).squeeze(-1)
        sentence_embedding = self.activation(self.post_linear(pooled))
        return sentence_embedding

#preprocess input sentences
def tokenize_sentences(sent_list, max_len=32):
    return tokenizer(
        sent_list,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

# run model
if __name__ == "__main__":
    sentences = [
        "Machine learning results depends on data.",
        "For NLP tasks transformers perform well.",
        "Here is my new car."
    ]

    model = SentenceEncoder(transformer)
    model.eval()

    with torch.no_grad():
        tokens = tokenize_sentences(sentences)
        embeddings = model(tokens['input_ids'], tokens['attention_mask'])

    print("Sentence Embeddings:\n")
    for idx, emb in enumerate(embeddings):
        print(f"Sentence: {sentences[idx]}")
        print(f"Embedding: {emb.numpy()[:8]}...")  # printing few embabded values  
        print("-" * 60)
