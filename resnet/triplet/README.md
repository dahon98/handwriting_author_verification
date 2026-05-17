# Handwriting Author Verification with Siamese Networks
This notebook implements a Siamese network for handwriting author verification using a ResNet50 backbone and triplet loss.

## Overview
The model determines whether two handwritten samples belong to the same author by learning discriminative feature embeddings. It is trained on a combination of four classic handwriting datasets:

IAM (handwriting lines)

CEDAR (signatures with forgeries)

Bengali (signatures)

Hindi (signatures)

## Key Features
Triplet Loss with margin=1.0 to pull same-author embeddings together and push different-author embeddings apart

ResNet50 backbone adapted for single-channel (grayscale) input

Configurable freezing modes (aggressive/moderate/none)

Dynamic triplet generation – positive pairs: same author; negative pairs: different authors or forgeries

Fixed validation/test pairs saved as pickle files for reproducibility

Metrics tracked: ROC-AUC, Equal Error Rate (EER), and accuracy

## Training Details
Epochs: 30

Learning rate: 0.001 with ReduceLROnPlateau scheduler

Optimizer: AdamW (weight decay: 0.0001)

Image size: 224×224

Embedding dimension: 256

## Dataset Split
Authors are split 70/15/15 across train/validation/test with no overlap between sets.

## Results
The best model achieved 89.77% validation ROC-AUC and 17.85% EER when training all layers (no freezing).

### External Datasets
The notebook also evaluates the trained model on:

ICDAR (Chinese and Dutch)

CVL dataset
