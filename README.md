# AI-ML-Engineer-Home-Task

## Multi-Task Sentence Transformer

A customized sentence transformer model that has been expanded to facilitate multi-task learning (MTL) is implemented in this study. The Hugging Face Transformers library and PyTorch are used to encode phrases into fixed-length embeddings and carry out several NLP actions at once. According to the responsibilities outlined in the take-home assignment, the solution is organized into four stages.

## Project Structure

```
multi-task-transformer/
├── sentence_transformer.py         # Sentence Encoder
├── multi_task_model.py             # MTL Architecture
├── multi_task_demo_run.py
├── train_MTL.py                    # Simulated Training Loop
├── requirements.txt                # Dependencies
└── README.md
```

## Part 1: Sentence Transformer Implementation

### Objective

Create a sentence encoder that generates fixed-length embeddings by utilizing a transformer backbone and a specialized pooling layer.

### Design

- Pretrained model: distilbert-base-uncased  
- Pooling: Adaptive average pooling  
- Output projection: Linear layer + Tanh activation

### Output

```
Sentence: Machine learning results depends on data.
Embedding: [ 0.28474316 -0.0534733   0.03675864 -0.10443391 -0.2735257 ...]
```

## Part 2: Multi-Task Learning Expansion

### Tasks

- Task A: Sentence Classification (e.g., statement, question, command)  
- Task B: Sentiment Analysis (negative, neutral, positive)

### Design

- Shared encoder for sentence embedding  
- Two task-specific heads (each a 2-layer MLP with ReLU)

### Output

```
Sentence: Hello, How Are you?
Task A logits: [-0.07, 0.10, 0.00]
Task B logits: [0.00, -0.01, -0.01]
```

## Part 3: Training Strategy & Transfer Learning

### Freezing Strategies

1. Freeze All Layers: Use as a baseline.  
2. Freeze Transformer: Only train classification heads.  
3. Freeze One Task Head: Preserve performance on a specific task.

### Transfer Learning Plan

- Model: distilbert-base-uncased  
- Freeze: Lower 3 transformer layers  
- Fine-tune: Upper layers and both task heads

### Example

```python
for name, param in model.transformer.named_parameters():
    if any(layer in name for layer in ["layer.0", "layer.1", "layer.2"]):
        param.requires_grad = False
```

## Part 4: Training Loop Simulation

### Features

- Simulates multi-task training using dummy labels  
- Logs individual task loss and accuracy  
- Uses CrossEntropyLoss and AdamW optimizer

### Log

```
Loss A: 0.8871, Loss B: 1.0210, Total: 1.9081
Acc A: 60.00%, Acc B: 40.00%
```

## Setup & Installation

### Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Run Sample Training Loop

```bash
python train_MTL.py
```
