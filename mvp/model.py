import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import timm


def adapt_conv2d_to_grayscale(module):
    """Adapt 3-channel conv layers to 1-channel by averaging weights."""
    if isinstance(module, nn.Conv2d) and module.in_channels == 3:
        new_conv = nn.Conv2d(
            1, module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None
        )
        with torch.no_grad():
            new_conv.weight.data = module.weight.data.mean(dim=1, keepdim=True)
            if module.bias is not None:
                new_conv.bias.data = module.bias.data
        module.in_channels = 1
        module.weight.data = new_conv.weight.data
        if module.bias is not None:
            module.bias.data = new_conv.bias.data
        return True
    return False


def adapt_model_to_grayscale(model):
    """Recursively adapt all conv layers to grayscale input."""
    for module in model.modules():
        adapt_conv2d_to_grayscale(module)
    return model


class CNNCrossViTWithHead(nn.Module):
    """Hybrid CNN-CrossViT model with classification head for signature verification."""
    
    def __init__(self, embedding_size=256, pretrained=True, freeze_backbone='none'):
        super(CNNCrossViTWithHead, self).__init__()

        # 1. CNN BRANCH (ResNet-18)
        if pretrained:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet18(weights=None)

        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        self.cnn_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.cnn_layer1 = resnet.layer1
        self.cnn_layer2 = resnet.layer2
        self.cnn_layer3 = resnet.layer3

        # FPN (Feature Pyramid Network)
        self.fpn_lateral = nn.ModuleList([
            nn.Conv2d(64, 128, 1),
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(256, 128, 1)
        ])
        self.fpn_output = nn.ModuleList([
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1)
        ])

        # 2. CROSSVIT BRANCH
        self.crossvit = timm.create_model('crossvit_base_240', pretrained=pretrained, num_classes=0)
        adapt_model_to_grayscale(self.crossvit)

        cnn_out_dim = 128 * 3  # 384
        crossvit_out_dim = self.crossvit.num_features

        # 3. FUSION HEAD
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + crossvit_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        # 4. CLASSIFICATION HEAD
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.embedding_size = embedding_size

        # 5. FREEZE LAYERS (if needed)
        if freeze_backbone == 'partial':
            self._freeze_layers_partial()
        elif freeze_backbone == 'full':
            self._freeze_layers_full()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CNN-CrossViT with Classification Head initialized")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    def _freeze_layers_partial(self):
        """Partially freeze backbone layers."""
        for param in self.cnn_conv1.parameters():
            param.requires_grad = False
        for param in self.cnn_layer1.parameters():
            param.requires_grad = False
        for param in self.cnn_layer2.parameters():
            param.requires_grad = False

    def _freeze_layers_full(self):
        """Fully freeze backbone layers."""
        for param in self.cnn_conv1.parameters():
            param.requires_grad = False
        for param in self.cnn_layer1.parameters():
            param.requires_grad = False
        for param in self.cnn_layer2.parameters():
            param.requires_grad = False
        for param in self.cnn_layer3.parameters():
            param.requires_grad = False
        for param in self.crossvit.parameters():
            param.requires_grad = False

    def _fpn_forward(self, c1, c2, c3):
        """Forward pass through Feature Pyramid Network."""
        p3 = self.fpn_lateral[2](c3)
        p2 = self.fpn_lateral[1](c2)
        p1 = self.fpn_lateral[0](c1)

        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode='nearest')
        p1 = p1 + F.interpolate(p2, size=p1.shape[-2:], mode='nearest')

        p3 = self.fpn_output[2](p3)
        p2 = self.fpn_output[1](p2)
        p1 = self.fpn_output[0](p1)

        p1 = F.adaptive_avg_pool2d(p1, 1).flatten(1)
        p2 = F.adaptive_avg_pool2d(p2, 1).flatten(1)
        p3 = F.adaptive_avg_pool2d(p3, 1).flatten(1)

        return torch.cat([p1, p2, p3], dim=1)

    def forward_one(self, x):
        """Extract embedding for a single image."""
        c1 = self.cnn_conv1(x)
        c2 = self.cnn_layer1(c1)
        c3 = self.cnn_layer2(c2)
        c4 = self.cnn_layer3(c3)
        cnn_features = self._fpn_forward(c2, c3, c4)

        crossvit_features = self.crossvit(x)

        combined = torch.cat([cnn_features, crossvit_features], dim=1)
        embedding = self.fusion(combined)

        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x1, x2):
        """
        Forward pass for a pair of images.
        
        Returns:
            logits: raw logits for binary classification
            emb1: normalized embedding for first image
            emb2: normalized embedding for second image
        """
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)

        # Compute absolute difference between embeddings
        distance = torch.abs(emb1 - emb2)
        logits = self.classifier(distance).squeeze(1)

        return logits, emb1, emb2


class ModelInference:
    """Inference class for handwriting verification service."""
    
    def __init__(self, model_path, threshold=0.81, device=None, freeze_backbone='none'):
        """
        Args:
            model_path: path to the trained .pth file
            threshold: decision threshold (default 0.81 as specified)
            device: 'cuda' or 'cpu'
            freeze_backbone: 'none', 'partial', or 'full'
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading CNN-CrossViT model on {self.device}...")
        
        # Create model
        self.model = CNNCrossViTWithHead(
            embedding_size=256,
            pretrained=True,
            freeze_backbone=freeze_backbone
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.threshold = threshold
        
        # Image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        print(f"Model loaded successfully. Decision threshold: {threshold}")
    
    def preprocess(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image, file path, or file-like object
            
        Returns:
            torch.Tensor: preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif isinstance(image, Image.Image):
            image = image.convert('L')
        else:
            image = Image.open(image).convert('L')
        
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, reference_img, questioned_img):
        """
        Predict whether two images are written by the same author.
        
        Args:
            reference_img: reference image (PIL Image or file-like)
            questioned_img: questioned image (PIL Image or file-like)
            
        Returns:
            dict: {
                'is_same_author': bool,
                'probability': float,
                'threshold': float,
                'logits': float,
                'distance': float (optional)
            }
        """
        ref_tensor = self.preprocess(reference_img)
        ques_tensor = self.preprocess(questioned_img)
        
        with torch.no_grad():
            logits, emb1, emb2 = self.model(ref_tensor, ques_tensor)
        
        # Convert logits to probability using sigmoid
        probability = torch.sigmoid(torch.tensor(logits)).item()
        
        # Decision based on threshold
        is_same_author = probability >= self.threshold
        
        # Compute Euclidean distance between embeddings (for reference)
        distance = torch.norm(emb1 - emb2, p=2, dim=1).item()
        
        return {
            'is_same_author': is_same_author,
            'probability': probability,
            'threshold': self.threshold,
            'logits': logits.item() if hasattr(logits, 'item') else float(logits),
            'distance': distance
        }


class DummyModel:
    """Dummy model for testing without a trained model file."""
    
    def __init__(self):
        self.threshold = 0.81
        print("Using dummy model (no real model loaded)")
    
    def predict(self, reference_img, questioned_img):
        import random
        probability = random.uniform(0.0, 1.0)
        is_same_author = probability >= self.threshold
        return {
            'is_same_author': is_same_author,
            'probability': probability,
            'threshold': self.threshold,
            'logits': 0.0,
            'distance': random.uniform(0.1, 1.2)
        }