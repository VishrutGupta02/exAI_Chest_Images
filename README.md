# Chest X-Ray Classification & Explainability Analysis
### COVID-19 · Normal · Viral Pneumonia

---

## Overview

This repository covers the development and evaluation of image classification models for chest X-ray diagnosis across three classes: **COVID-19**, **Normal**, and **Viral Pneumonia**. The study goes beyond standard classification and focuses on understanding *why* a model arrives at a specific prediction, not just *what* the prediction is.

Key aspects of the study:
- Train and compare four deep learning architectures — Basic CNN, CheXNet (DenseNet-121), Basic Vision Transformer (ViT), and a proposed Hybrid CNN-ViT model
- Apply explainability methods including **GradCAM**, **Integrated Gradients**, and **Attention Rollout** to interpret model decisions
- Quantitatively evaluate saliency maps using **Insertion**, **Deletion**, **Entropy**, and **AOPC** metrics
- Identify limitations of current evaluation methods and propose a novel, clinically motivated improvement

---

## Repository Structure

```
📦 root
 ├── 📁 Basic_CNN_baseline
 │    ├── Basic_CNN.ipynb
 │    └── 📁 Outputs
 │
 ├── 📁 CheXNet_baseline
 │    ├── train_chexnet.ipynb
 │    └── 📁 Outputs
 │
 ├── 📁 Hybrid_baseline
 │    ├── hybrid_cnn_transformer.ipynb
 │    └── 📁 Outputs
 │
 ├── 📁 ViT_baseline
 │    ├── ViT_training.ipynb
 │    └── 📁 Outputs
 │
 ├── 📁 Explainability
 │    ├── 📁 Random_100_cropped          ← 100 randomly selected cropped test images
 │    └── 📁 Results
 │         ├── 📁 Basic_CNN_GradCam
 │         │    ├── ...                  ← RGB saliency output images
 │         │    └── 📁 Saliency_maps     
 │         ├── 📁 CheXNet_GradCam
 │         │    ├── ...                  
 │         │    └── 📁 Saliency_maps
 │         ├── 📁 Int_Grad_CheXNet
 │         │    ├── ...                  
 │         │    └── 📁 Saliency_maps
 │         └── 📁 Att_Roll_Basic_ViT
 │         │    ├── ...                  
 │              └── 📁 Saliency_maps
 │
 └── model_eval.ipynb                    ← Evaluation script: Insertion, Deletion, AOPC, Entropy
```

---

## Dataset

**Download:** [Google Drive](https://drive.google.com/file/d/1nzZghdzlxduhvil0xMVqCumB6bzbtUcf/view)

| Split      | Images |
|------------|--------|
| Train      | 1,000  |
| Validation | 200    |
| Test       | 145    |

After downloading and extracting, the dataset should be organised as follows:

```
📦 dataset
 ├── 📁 train
 │    ├── 📁 COVID
 │    ├── 📁 Normal
 │    └── 📁 Viral_Pneumonia
 │
 ├── 📁 val
 │    ├── 📁 COVID
 │    ├── 📁 Normal
 │    └── 📁 Viral_Pneumonia
 │
 └── 📁 test
      ├── 📁 COVID
      ├── 📁 Normal
      └── 📁 Viral_Pneumonia
```

---

## Data Preprocessing

A critical issue discovered during early experiments was that the models were learning from **text annotations on the borders** of the images rather than the actual lung regions:

- **Viral Pneumonia** images had an `"R"` written in the top-left corner
- **Normal** images had an `"L"` written in the top-left corner
- **COVID-19** images had `"DX"` and other text written along the top margin

This shortcut learning was confirmed by running GradCAM on early model outputs — the heatmaps highlighted the text regions rather than lung tissue. To fix this, **all images were cropped by 30 pixels from both the top and left edges** before retraining.

Additional preprocessing steps applied:
- Standard normalisation across all splits
- Training images randomly shuffled
- Data augmentation on the training set: **random rotation** and **horizontal flip**
- A fixed set of **100 randomly selected cropped images** was extracted from the test set for all explainability experiments (stored in `Explainability/Random_100_cropped`)

---

## Methodology

### Models Trained

**1. Basic CNN** — Standard convolutional network used as a baseline.

**2. CheXNet (DenseNet-121)** — Industry gold standard for medical imaging. A 121-layer densely connected CNN where each layer receives feature maps from all preceding layers. Pipeline:
```
Input X-ray → DenseNet-121 → Global Average Pooling → Fully Connected Layer → Class Probabilities
```
Training was further improved using a **learning rate scheduler** and **early stopping**.

**3. Basic Vision Transformer (ViT)** — Divides the input image into fixed-size patches, embeds them as tokens, and processes them through transformer encoder layers with multi-head self-attention.

**4. Proposed Hybrid CNN-ViT** — An integrated hybrid where the CNN branch feeds directly into the transformer branch (rather than simple end-concatenation):
```
CNN Stem → Token Projection → Transformer Encoder → Classifier
```

### Explainability Methods

| Method | Applied To | How It Works |
|---|---|---|
| **GradCAM** | Basic CNN, CheXNet | Generates heatmaps using gradients of the predicted class w.r.t. the final convolutional layer |
| **Integrated Gradients** | CheXNet | Attributes importance by integrating gradients along a path from a blank baseline to the actual input |
| **Attention Rollout** | Basic ViT | Recursively multiplies attention matrices across all transformer layers to trace information flow |

> Since GradCAM and Attention Rollout are not model-agnostic, they are applied only to the architectures they are designed for.

---

## Results

### Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Basic CNN | 0.9563 | 0.9566 | 0.9563 | 0.9561 |
| CheXNet (DenseNet-121) | **0.9839** | **0.9841** | **0.9839** | **0.9839** |
| Basic ViT | 0.9816 | 0.9819 | 0.9816 | 0.9817 |
| Proposed Hybrid | 0.9770 | 0.9772 | 0.9770 | 0.9771 |

**CheXNet** achieves the best performance at **98.39% accuracy**, confirming its status as the gold standard for medical imaging. The Basic ViT performs competitively and surpasses the Basic CNN. The Hybrid model performs well overall but falls short of both CheXNet and Basic ViT — likely because generic CNN-ViT fusion designs are sensitive to the exact fusion mechanism used.

---

## Explainability Analysis

### Saliency Map Observations

**Basic CNN — GradCAM**
The heatmaps are diffuse, with broad blue regions and only a few bright red hotspots. The model loosely identifies the chest region but without strong confidence or anatomical precision. Normal X-rays show very few activations, as expected given the absence of visible disease.

**CheXNet — GradCAM & Integrated Gradients**
GradCAM confidently highlights the lung fields and, importantly, the **mediastinum** — the region between the two lung cavities containing the trachea, heart, and lymph nodes. This is clinically meaningful, as physicians routinely examine the mediastinum for signs of disease. Integrated Gradients confirmed these findings with cleaner, easier-to-read maps.

**Basic ViT — Attention Rollout**
The ViT focuses on the **upper abdomen and diaphragm** rather than the lung zones directly. While the diaphragm is a relevant diagnostic region, the relatively low weight given to the lung fields themselves is a potential limitation for COVID-19 and viral pneumonia detection.

### Quantitative Evaluation

| Model | Method | Insertion ↑ | Deletion ↓ | Entropy ↓ | AOPC ↑ |
|---|---|---|---|---|---|
| Basic CNN | GradCAM | 46.21 | 0.358 | 10.19 | 0.505 |
| CheXNet | GradCAM | 40.68 | 0.559 | 10.51 | 0.526 |
| CheXNet | Integrated Gradients | 35.19 | 0.622 | 10.21 | **0.579** |
| Basic ViT | Attention Rollout | 37.88 | 0.828 | 10.60 | 0.056 |

> ↑ = higher is better &nbsp;·&nbsp; ↓ = lower is better

**Key Takeaways:**

- **Insertion** — Basic CNN scores highest (46.21), meaning its identified pixels drive a quicker confidence recovery when inserted back into a blank image.
- **Deletion** — Basic CNN scores best (0.358), with confidence collapsing fastest when its pixels are removed. CheXNet GradCAM performs moderately (0.559).
- **Entropy** — All models show similar values (~10.2–10.6). Basic CNN and CheXNet with IG produce the most focused maps.
- **AOPC** — CheXNet with Integrated Gradients achieves the highest score (0.579), indicating its identified pixels are the most causally important for predictions. The Basic ViT with Attention Rollout scores notably low (0.056) — a known limitation of attention-based explanations: attention shows what the model *looks at*, not necessarily what it *uses* to decide.

---

## Evaluation Metrics Explained

Run the evaluation code in `model_eval.ipynb`.

| Metric | Description |
|---|---|
| **Insertion** | Starts from a blurred image and progressively adds the most important pixels. Confidence should rise quickly for a good explanation. Higher AUC is preferred. |
| **Deletion** | Starts from the original image and removes important pixels first. Confidence should drop quickly. Lower AUC is preferred. |
| **Entropy** | Measures how concentrated the saliency map is. Low entropy = more focused attention on important regions. Lower is preferred. |
| **AOPC** | Measures average confidence drop as important pixels are removed across multiple perturbation steps. Higher is preferred. |
