# Evaluating Artifact Removal Techniques for Hexagonal Fiber Optic Laryngoscopy in Cases of Organic Dysphonia

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

This repository hosts the official engineering code framework and algorithms developed for a Master's Thesis in Medical Engineering at **Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)**. The project provides an advanced, comparative exploration bridging data-driven deep learning (**HAR-CNN**) and classical mathematical optimization pipelines (**TwIST**, **Edge-Directed Interpolation**, and **Star-Pattern Calibration Filters**) to eliminate structural honeycomb grid artifacts from flexible laryngoscopic endoscopy.

---

## 🚀 Clinical Context & Project Overview

During clinical laryngoscopy, flexible nasal fiber-optic endoscopes are essential for checking functional vocal cord patterns during organic dysphonia examinations (detecting conditions like polyps, cysts, and nodules). However, the spatial arrangement of individual internal fiber cores produces a **hexagonal honeycomb grid artifact** that heavily distorts the high-frequency structural boundaries of diagnostic features.

This toolkit offers four unique methodologies designed to eliminate these pixel grid artifacts while meticulously preserving organic structural information critical for clinical diagnostics:

### The Repository Processing Modules
1. **Proposed Deep Learning Solution (`notebooks/HAR_CNN.ipynb`):** A Hierarchical Attentive Residual Convolutional Neural Network (HAR-CNN) with Squeeze-and-Excitation (SE) channel scaling. It maps distorted inputs to targets through Global Residual Learning (GRL).
2. **Classical Iterative Optimization (`notebooks/TWIST with NIQE.ipynb`):** An accelerated Two-Step Iterative Shrinkage/Thresholding solver leveraging Total Variation regularizers (TV-Chambolle proximal mapping) to treat artifact removal as an inverse problem.
3. **Adaptive Geometric Filtering (`notebooks/Interpolation with NIQE.ipynb`):** Edge-directed adaptive spatial sub-pixel interpolation used to smooth fiber-core high frequencies.
4. **Diagnostic & Quality Evaluation (`notebooks/Star With NIQE.ipynb`):** Star-pattern tracking combined with a No-Reference Image Quality Assessment via **NIQE (Natural Image Quality Evaluator)** to compute objective human visual acceptability scores.

---

## 📊 Proposed Deep Learning Architecture (HAR-CNN)

The network optimizes single-channel grayscale inputs by separating periodic grids from clinical diagnostic elements across five explicit steps:

1. **Shallow Feature Extraction:** An expansive $9 \times 9$ Convolutional boundary window (64 filters) with Batch Normalization and ReLU activation to catch broad structural context.
2. **Deep Attention Core:** A cascaded sequence of **8 identical Residual Blocks**. Each block incorporates twin $3 \times 3$ convolutions and an integrated **Squeeze-and-Excitation Channel Attention** block utilizing a dimensional reduction ratio of $r=8$.
3. **Post-Residual Smoothing:** A $3 \times 3$ Convolutional mapping block (64 filters).
4. **Feature Bottleneck Layer:** A $1 \times 1$ Convolution scaling representations down to 32 dimensions to filter out low-frequency data.
5. **Reconstruction & Global Skip Connection:** A final $5 \times 5$ Convolution layer that isolates the underlying artifact field. The network relies on **Global Residual Learning**, meaning it learns the degradation noise map, which is then mathematically added back to the original corrupted input image:
$$\text{Output} = \text{Input} + \text{Predicted Residual}$$

---

## 🧮 Mathematical Frameworks

### 1. Composite Optimization Loss (HAR-CNN)
To avoid the blurry tissue anomalies associated with traditional Mean Squared Error ($L_2$) loss, network convergence is driven by a perceptual hybrid objective function balancing structural semantics and absolute pixel uniformity:
$$\mathcal{L}_{\text{composite}} = \alpha \cdot \mathcal{L}_{\text{SSIM}} + (1 - \alpha) \cdot \mathcal{L}_{\text{MAE}}$$
$$\mathcal{L}_{\text{composite}} = 0.84 \cdot \left(1 - \text{SSIM}(y, \hat{y})\right) + 0.16 \cdot \left(\frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|\right)$$

### 2. TwIST Minimization Objective
The TwIST framework speeds up traditional single-step IST optimization runs over ill-conditioned matrices by tracking momentum vectors over the *two* preceding operational states:
$$x_{t+1} = (1 - \alpha)x_{t-1} + (\alpha - \beta)x_t + \beta \cdot \Psi_{\text{TV}} \left(x_t + A^T(y - Ax_t)\right)$$
Where $\Psi_{\text{TV}}$ represents the total variation proximal denoising operator, and $A$ represents the physics-based forward degradation blur operator.

---

## 📈 Empirical Benchmarks & Performance Summary

Based on rigorous cross-validation evaluation against clinical laryngoscopic imagery, the technical tradeoffs across the evaluated methods are summarized below:

| Methodology | Peak PSNR (dB) | Structural Similarity (SSIM) | NIQE Score (Lower = Better) | Operational Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: |
| **Star Function** | $14.21$ | $0.42$ | $8.54$ | — |
| **Spatial Interpolation** | $17.02$ | $0.59$ | $7.11$ | **30.0+ (Real-Time)** |
| **TwIST Optimization** | $18.45$ | $0.66$ | $6.48$ | $2.41$ (Iterative Latency) |
| **HAR-CNN (Proposed)** | **19.61** | **0.74** | **6.14** | **13.16 (GPU Accelerated)** |

### Key Clinical Insight:
Traditional spatial filters introduce severe blur over lesion textures. The **HAR-CNN** framework achieves superior performance because its embedded **Squeeze-and-Excitation attention mechanism** separates periodic fiber grid frequencies from organic structural data, completely preserving diagnostic tissue boundaries while maintaining high operational throughput.

---

## 🛠️ Repository Directory & Setup

### 1. Target Folder Structure
Ensure your repository files match this layout:
```text
HAR-CNN-Image-Restoration/
├── notebooks/
│   ├── HAR_CNN.ipynb                 # Proposed Deep Attentive Residual CNN
│   ├── TWIST with NIQE.ipynb         # Two-Step Iterative Shrinkage Solver
│   ├── Interpolation with NIQE.ipynb   # Edge-Directed Spatial Interpolation Baseline
│   └── Star With NIQE.ipynb          # Star-Pattern Tracking & Quality Evaluation
├── Requirements.txt                  # System dependency configuration
└── README.md
