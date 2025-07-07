# multi_task_model.py

import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, transformer_model: AutoModel, embedding_dim=768,
                 num_classes_task_a=3, num_classes_task_b=3):
        super(MultiTaskSentenceTransformer, self).__init__()

        self.transformer = transformer_model
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.shared_linear = nn.Linear(embedding_dim, embedding_dim)
        self.shared_activation = nn.Tanh()

        # Task-specific heads
        self.task_a_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes_task_a)
        )

        self.task_b_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes_task_b)
        )

    def forward(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.pooling(x.permute(0, 2, 1)).squeeze(-1)
        sentence_embedding = self.shared_activation(self.shared_linear(pooled))

        out_task_a = self.task_a_classifier(sentence_embedding)
        out_task_b = self.task_b_classifier(sentence_embedding)

        return {
            "embedding": sentence_embedding,
            "task_a_logits": out_task_a,
            "task_b_logits": out_task_b
        }
