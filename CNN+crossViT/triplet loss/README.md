# CNN-CrossViT Model for Handwriting Author Verification
This notebook implements a hybrid neural network model for handwriting author verification — determining whether two handwriting samples belong to the same person.

## Model Architecture
The CNN-CrossViT Triplet model combines two parallel feature extractors:

### 1. CNN Branch (ResNet-18)
Extracts local features from grayscale images (pen strokes, texture patterns)

Uses FPN (Feature Pyramid Network) for multi-level feature extraction from three ResNet layers (64, 128, 256 channels)

Output: 192-dimensional feature vector

### 2. CrossViT Branch
CrossViT Base 240 adapted for grayscale images (originally pretrained on RGB)

Captures global contextual information across the entire image

Output: 768-dimensional feature vector

### 3. Fusion & Embedding
Concatenates features (960 dimensions) → passes through MLP with BatchNorm and Dropout (0.6)

Produces L2-normalized embeddings (256 dimensions) optimized for triplet loss

## Training Details
Loss function: Triplet Margin Loss (margin = 1.0)

Optimizer: AdamW with weight decay (0.01)

Learning rate schedule: 5 epochs warmup → cosine annealing to 1e-6

Dataset: Mixed dataset (IAM handwriting lines + CEDAR, Bengali, Hindi signatures)

Triplet sampling: IAM (same/different authors) + signature datasets (genuine/forgery pairs)

## Results
The model achieves strong performance on validation data with ROC-AUC up to 86.6% and EER as low as 22%. Evaluation on multiple test sets (general, Chinese, Dutch, CVL) demonstrates reasonable generalization across different handwriting styles and languages.
