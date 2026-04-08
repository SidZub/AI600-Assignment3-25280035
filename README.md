# AI600 – Deep Learning | Assignment 3
**Lahore University of Management Sciences**
Spring 2026 | Student ID: 25280035

---

## Overview
This repository contains the code for Assignment 3 of AI600 – Deep Learning. The assignment covers custom CNN design, shortcut learning, transfer learning, and model interpretability using GradCAM.

---

## Repository Structure
```
├── Task1_MNIST_CMNIST.ipynb       # Task 1: Custom CNN on MNIST and Colored-MNIST
├── Task2_ResNet_GradCAM.ipynb     # Task 2: ResNet-18 fine-tuning and GradCAM
├── model_a_mnist.pth              # Saved weights: Part A MNIST model
├── model_b_cmnist.pth             # Saved weights: Part B C-MNIST model
├── resnet18_stl10.pth             # Saved weights: ResNet-18 STL-10 model
├── mnist_curves.png               # Part A: loss and accuracy curves
├── mnist_filters.png              # Part A: Conv1 filter visualisation
├── cmnist_curves.png              # Part B: C-MNIST loss and accuracy curves
├── gradcam_results.png            # Task 2: GradCAM heatmaps
└── README.md
```

---

## Tasks

### Task 1: Custom CNNs and Shortcut Learning

**Part A – Standard MNIST**
- Custom CNN with 3 convolutional layers and 2 fully connected layers
- Total trainable parameters: 13,498 (under 50,000 limit)
- Trained with Adam optimizer and CrossEntropy loss for 15 epochs
- Final test accuracy: **99.07%**

**Part B – Colored-MNIST (C-MNIST)**
- Same architecture adapted for 3-channel RGB input
- Trained on biased C-MNIST training set
- Biased test accuracy: **99.18%**
- Unbiased test accuracy: **86.68%**
- Demonstrates shortcut learning via color-class spurious correlations

---

### Task 2: Transfer Learning and Interpretability

**Part A – Fine-tuning ResNet-18 on STL-10**
- Pre-trained ResNet-18 loaded from torchvision (ImageNet weights)
- Backbone frozen; only the final classification head trained
- Trainable parameters: 5,130 out of 11,181,642
- Final test accuracy: **94.71%**

**Part B – GradCAM Visualisation**
- GradCAM applied to the final convolutional layer (layer4)
- 2 correctly classified and 2 incorrectly classified samples visualised
- Library used: `pytorch-grad-cam`

---

## Requirements
```
torch
torchvision
numpy
matplotlib
grad-cam
```

Install with:
```bash
pip install torch torchvision numpy matplotlib grad-cam
```

---

## Dataset Notes
- **MNIST**: downloaded automatically via `torchvision.datasets.MNIST`
- **STL-10**: downloaded automatically via `torchvision.datasets.STL10`
- **C-MNIST**: provided as `.pt` files (`train_biased.pt`, `test_biased.pt`, `test_unbiased.pt`) — not included in this repository due to file size

---

## Environment
All code was developed and run on **Google Colab** with a T4 GPU.
