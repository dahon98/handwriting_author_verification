# Handwriting Author Verification with Siamese Network
This notebook implements a Siamese Neural Network for handwriting author verification using a ResNet50 backbone. The model determines whether two handwriting samples come from the same author or different authors.

## Dataset
The model is trained on four classic handwriting datasets:

IAM - Handwriting lines

CEDAR - Signature dataset

Bengali - Signature dataset with forgeries

Hindi - Signature dataset with forgeries

## Model Architecture
Backbone: ResNet50 (pretrained on ImageNet) adapted for single-channel input

Embedding: 256-dimensional feature vector

Loss Function: Contrastive Loss (margin=1.0)

## Freezing Strategies
The notebook explores three different freezing modes:

Aggressive - Only layer4 and conv1 trainable (17.3M trainable params)

Moderate - Layers 3, 4, and conv1 trainable (24.4M trainable params)

None - All layers trainable (25.9M trainable params)

## Training
Pairs per epoch: 5,000

Epochs: 30

Optimizer: AdamW

Learning Rate: 0.001 with ReduceLROnPlateau scheduler

Weight Decay: 0.0001

## Evaluation
The model is evaluated on:

Custom test set (authors not seen during training)

ICDAR Chinese dataset

ICDAR Dutch dataset

## Metrics
ROC-AUC

Equal Error Rate (EER)

Accuracy

Embedding distances (positive/negative pairs)
