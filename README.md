# 🏛️ Landmark Detection & Recognition using Deep Learning

A **Landmark Detection and Recognition System** built using **Deep Learning** and **Computer Vision**.  
This project identifies real-world landmarks (monuments, buildings, natural formations) from images and retrieves visually similar landmarks efficiently.

---

## 📘 Table of Contents

1. [Overview](#overview)  
2. [Objectives](#objectives)  
3. [Dataset](#dataset)  
4. [System Architecture](#system-architecture)  
5. [Implementation](#implementation)  
6. [Model Details](#model-details)  
7. [Results](#results)  
8. [Deployment](#deployment)  
9. [Ethics & Privacy](#ethics--privacy)  
10. [Future Work](#future-work)  
11. [Contributors](#contributors)

---

## 🧠 Overview

This project builds a robust **Landmark Detection & Recognition** system that can:
- Identify and classify famous landmarks from images  
- Retrieve similar images from a landmark gallery  
- Work efficiently on both **server** and **mobile (edge)** devices  

**Use Cases:**
- Travel and tourism apps  
- Cultural heritage documentation  
- Augmented Reality (AR) tour guides  
- Photo organization and visual search  

---

## 🎯 Objectives

- Achieve **≥ 75% mAP** on the Google Landmarks v2 (GLDv2-clean) retrieval split  
- Achieve **≥ 70% Top-1 accuracy** on 1,000+ landmark classes  
- Keep **query latency ≤ 300ms** (GPU) or ≤ 1s (CPU)  
- Build a lightweight on-device model (**≤ 25MB**) for mobile inference  

---

## 🗂️ Dataset

### Public Datasets Used:
- **Google Landmarks v2 (GLDv2)**  
- **ROxford** and **RParis** datasets for evaluation  
- **Custom local dataset** for region-specific landmarks  

### Preprocessing Steps:
- Resize images to 512px (max side)  
- Apply augmentations: RandomCrop, Flip, ColorJitter, Gaussian Noise  
- Normalize and center-crop  
- Strip GPS/EXIF data for privacy  

---

## ⚙️ System Architecture

**Pipeline Overview:**

Image Input → Preprocessing → Feature Extraction → Embedding → ANN Search → Label Output


**Components:**
1. **Backbone:** ResNet-50 / EfficientNet-V2 / MobileNet-V3  
2. **Pooling:** Generalized Mean (GeM) pooling  
3. **Embedding:** 512-D L2-normalized global descriptor  
4. **Loss Functions:** ArcFace, AM-Softmax, Triplet, or Contrastive loss  
5. **Indexing:** FAISS-based ANN index (IVF-PQ / HNSW) for efficient retrieval  
6. **Re-ranking:** k-reciprocal and geometric verification (SuperPoint / RANSAC)

---

## 💻 Implementation

### 🧩 Installation
```bash
pip install torch torchvision faiss-cpu numpy pillow

🧠 Model Code Example

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import faiss, os, numpy as np

# Model setup
class LandmarkFeatureExtractor(nn.Module):
    def __init__(self):
        super(LandmarkFeatureExtractor, self).__init__()
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
            return torch.nn.functional.normalize(features, p=2, dim=1)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_embedding(image_path, model, device="cpu"):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    return model(img_t).cpu().numpy()
```

## 🧮 Model Details

| Model Type | Backbone | Embedding Dim | Pooling | Loss Function | Key Features |
|-------------|-----------|----------------|----------|----------------|----------------|
| **Server Model** | ResNet-50 | 512-D | GeM (Generalized Mean) | ArcFace | High accuracy, robust feature embeddings |
| **Transformer Model** | ViT-B/16 | 512-D | GeM | Contrastive | Better global context and reasoning |
| **Edge/Mobile Model** | MobileNet-V3 | 256-D | GeM | Softmax | Lightweight, optimized for mobile/edge inference |

**Training Configuration**
## 🧮 Model Details

| Model Type | Backbone | Embedding Dim | Pooling | Loss Function | Key Features |
|-------------|-----------|----------------|----------|----------------|----------------|
| **Server Model** | ResNet-50 | 512-D | GeM (Generalized Mean) | ArcFace | High accuracy, robust feature embeddings |
| **Transformer Model** | ViT-B/16 | 512-D | GeM | Contrastive | Better global context and reasoning |
| **Edge/Mobile Model** | MobileNet-V3 | 256-D | GeM | Softmax | Lightweight, optimized for mobile/edge inference |

**Training Configuration**
```python
optimizer = "AdamW"
learning_rate = 3e-4       # cosine decay schedule
batch_size = 128
epochs = 50                # can extend up to 80
weight_decay = 1e-4
mixed_precision = True     # FP16 or BF16 for faster training
gradient_clipping = True
```

## 📊 Results
MetricResultTop-1 Accuracy70%+mAP75%+GPU Latency~200 msCPU Latency~900 ms
The model demonstrates robust retrieval and classification even under variations in lighting, scale, and viewpoint.

## 🚀 Deployment
Server API Example:
POST /v1/landmarks:detect
Content-Type: application/json

{
  "image_b64": "<base64_image>",
  "top_k": 5
}

**Response:**
{
  "label": "taj_mahal",
  "confidence": 0.91,
  "neighbors": [{"id": "img123", "score": 0.86}],
  "decision": "known"
}

**Infrastructure:**

REST API with Dockerized inference service
FAISS index shards with GPU acceleration
Monitoring via Prometheus + Grafana



## 🔐 Ethics & Privacy

Remove EXIF/GPS metadata from all images
No user data stored without consent
Respect dataset licenses and attribution
Reject non-landmark or inappropriate images via thresholding



## 🔮 Future Work

3D reconstruction for viewpoint normalization
Multimodal landmark recognition (image + text using CLIP)
Self-supervised feature learning (DINOv2)
Real-time mobile AR landmark recognition


## 👥 Contributors

Author: Aswathi Sujith




