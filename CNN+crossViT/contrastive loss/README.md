# CNN-CrossViT Model for Handwriting Author Verification
This notebook implements a hybrid neural network model for handwriting author verification — determining whether two handwriting samples belong to the same person.

## Model Architecture
The CNNCrossViT model combines three main components:

### 1. CNN Branch (ResNet-18)
Extracts local features from images

Uses three ResNet layers with FPN (Feature Pyramid Network) for multi-level feature extraction

Output: 384-dimensional feature vector

### 2. CrossViT Branch
CrossViT Base 240 adapted for grayscale images

Captures global contextual features

Output: 768-dimensional feature vector

### 3. Fusion Module
Concatenates features (1152 dimensions total)

Passes through MLP: Linear(1152→512) → BatchNorm → ReLU → Dropout → Linear(512→256)

Final L2 normalization produces 256-dimensional embeddings

## Training Details
Loss function: Contrastive Loss with margin = 1.0

Optimizer: AdamW (lr = 0.0001, weight decay = 0.001)

Scheduler: 5-epoch warmup + Cosine Annealing (to 1e-6)

Training pairs: 5000 pairs per epoch (balanced 50/50 positive/negative)

## Results

The model achieves strong performance on validation data and demonstrates good generalization across different handwriting datasets.

### Validation Performance

| Metric | Best Value |
|--------|------------|
| ROC-AUC | 97.40% |
| EER | 7.25% |
| Best Accuracy | 93.50% |

### Test Set Performance (threshold = 0.5)

| Test Set | ROC-AUC | Accuracy |
|----------|---------|----------|
| Primary Test | 98.21% | 93.50% |
| ICDAR Chinese | 72.20% | 58.62% |
| ICDAR Dutch | 87.02% | 68.58% |
| CVL | 83.23% | 55.94% |

Dataset: Mixed dataset (IAM handwriting, CEDAR, Bengali, Hindi signatures)
