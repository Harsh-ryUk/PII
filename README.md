## Project Overview

This repository contains code and experiments for the Kaggle competition **The Learning Agency Lab - PII Data Detection**.  
The goal is to build models that automatically detect and classify **Personally Identifiable Information (PII)** entities—such as names, emails, phone numbers, addresses, user names, IDs, and URLs—in educational documents. The task is formulated as a token classification (NER) problem using modern transformer-based methods.

---

## Repository Structure

- **data/**  
  Folder for raw and processed datasets (not included). Place competition files here:  
  `train.json`, `test.json`, `train_labels.csv`, `metadata.csv`

- **notebooks/**  
  EDA and baseline model exploration.

- **src/**  
  Full training and preprocessing codebase.

- **models/**  
  Saved model weights (optional).

- **submissions/**  
  Submission CSVs for Kaggle evaluation.

- **requirements.txt**  
  Python dependencies.

- **config.py**  
  Project configuration files (sample).

- **README.md**  
  This file.

---

## Getting the Data

1. Visit the Kaggle competition page for PII Data Detection.
2. Accept competition rules.
3. Download training and test data and place in the `data/` folder:
    - `train.json`
    - `test.json`
    - `train_labels.csv`
    - `metadata.csv`

---

## Setup

Create a Python virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Minimal `requirements.txt` example:
```
numpy
pandas
scikit-learn
torch
transformers
evaluate
seqeval
tqdm
matplotlib
```

---

## How to Run

### 1. Preprocess

Tokenize and format data for NER training and evaluation:

```bash
python src/preprocess.py --inputdir data/train --outputdir data/processed --config config.py
```

### 2. Train

Train transformer model for token classification (e.g., DeBERTa V3):

```bash
python src/train.py --datadir data/processed --modeldir models/exp1 --epochs 8 --batch-size 2
```

**Key hyperparameters:**
- Model: `microsoft/deberta-v3-base`
- Max sequence length: 1024
- Optimizer: AdamW, lr=2e-5
- Scheduler: cosine
- Metric: Macro F1/Binary F1
- Batch size: 2

### 3. Inference

Generate predictions for test documents:

```bash
python src/inference.py --modelpath models/exp1/best.pth --testdir data/test --output submissions/submission_exp1.csv
```

---

## Notebooks

- Sensor data EDA — visualize PII label distributions, entity frequencies, and dataset characteristics
- Baseline models — test token classification using pre-trained transformer architectures (BERT, DeBERTa, etc.)

---

## Tips and Best Practices

- Downsample negative examples to handle heavy class imbalance
- Use external public datasets for augmentation
- Pay careful attention to tokenization (handling whitespace and overlapping entities)
- Experiment with windowing approaches for long sequences
- Focal loss or class weighting can improve performance for rare entities
- Submit your final results using Kaggle's recommended submission format

---

## References

- Kaggle: The Learning Agency Lab - PII Data Detection competition
- HuggingFace Transformers documentation
- Seqeval metrics for NER
- Official community starter notebooks
'''

with open('README-PII-Kaggle.md', 'w', encoding='utf-8') as f:
    f.write(pii_readme_md)

'File README-PII-Kaggle.md created.'
