# Siamese Network for Handwriting Author Verification
This notebook implements a Siamese neural network for handwriting author verification using a ResNet50 backbone. The model determines whether two handwriting samples belong to the same author or different authors.

## Overview
The network uses a Siamese architecture with a pretrained ResNet50 as the backbone, modified to accept single-channel (grayscale) inputs. The model is trained with a combination of Contrastive Loss and Binary Cross-Entropy (BCE) loss.

### Dataset
The model is trained on handwriting examples from four datasets:

IAM (handwriting lines)

CEDAR (signatures with genuine and forgery samples)

Bengali (signatures with genuine and forgery samples)

Hindi (signatures with genuine and forgery samples)

### Model Architecture
Backbone: ResNet50 (pretrained on ImageNet)

Input: Grayscale images resized to 224×224

Embedding size: 256-dimensional feature vectors

Loss: Combined Contrastive Loss + BCE Loss

### Freeze Modes
Three different freeze configurations are experimented with:

Aggressive: Only layer4 and conv1 are trainable (67% trainable parameters)

Moderate: Layers 3, 4, and conv1 are trainable (94.4% trainable parameters)

None: All layers are trainable (100% trainable parameters)

### Training Configuration
Batch size: 8

Epochs: 30

Optimizer: AdamW

Learning rate: 0.001 (with ReduceLROnPlateau scheduler)

Weight decay: 0.0001

### Evaluation Metrics
ROC-AUC (Area Under Curve)

Accuracy (threshold = 0.5)

Equal Error Rate (EER)

Distance statistics (positive and negative pairs)

