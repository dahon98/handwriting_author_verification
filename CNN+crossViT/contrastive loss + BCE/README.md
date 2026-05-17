# CNN-CrossViT Model for Handwriting Author Verification
This notebook implements a hybrid neural network model for handwriting author verification — determining whether two handwriting samples belong to the same person.

## Model Architecture
The CNNCrossViTWithHead model combines three main components:

### 1. CNN Branch (ResNet-18)
Extracts local features from images

Uses three ResNet layers with FPN for multi-level feature extraction

Output: 384-dimensional feature vector

### 2. CrossViT Branch
CrossViT Base 240 adapted for grayscale images

Captures global contextual features

Output: 768-dimensional feature vector

### 3. Fusion & Classification
Concatenates features (1152 dimensions) → passes through MLP → produces 256-dim embeddings

Classification head computes absolute difference between embeddings and outputs similarity score

## Training Details
Loss function: Combined Contrastive Loss + Binary Cross-Entropy (BCE weight: 0.2)

Optimizer: AdamW with cosine annealing learning rate scheduler

Dataset: Mixed dataset (IAM handwriting, CEDAR, Bengali, Hindi signatures)

## Results
The model achieves strong performance on validation data with ROC-AUC up to 98.35% and EER as low as 6.65%. Evaluation on multiple test sets (Chinese, Dutch, CVL) demonstrates good generalization capabilities.
