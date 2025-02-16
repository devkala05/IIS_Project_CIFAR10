# ResNet-Based Image Classification on CIFAR-10

## Overview

This project implements a ResNet-based image classification model using the **CIFAR-10 dataset**. The model is trained to classify **5 selected classes** from CIFAR-10 and includes performance enhancements through architectural modifications and learning rate scheduling.

## Features

- **Data Preprocessing**: Normalization, augmentation (horizontal flipping), and dataset splitting into **train, validation, and test sets**.
- **Custom ResNet Architecture**: Built from scratch using **PyTorch** with residual blocks.
- **Training & Optimization**:
  - SGD optimizer with momentum
  - CrossEntropy loss function
  - Learning rate scheduling (ReduceLROnPlateau)
  - Model checkpointing (saving best model based on validation accuracy)
- **Visualization**:
  - Loss and accuracy plots for train & validation sets
  - Feature map visualization for different ResNet layers

## Dataset

- The **CIFAR-10** dataset contains **60,000** images (32x32 pixels, 10 classes).
- The model is trained on a subset with the following 5 classes:
  - **Airplane, Automobile, Bird, Cat, Truck**

## Installation & Setup

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install torch torchvision numpy matplotlib tqdm
```

### Running the Model

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/resnet-cifar10.git
   cd resnet-cifar10
   ```
2. Run the training script:
   ```bash
   python train.py
   ```
3. View loss & accuracy plots after training.

## Results

- The model was trained for **20 epochs**.
- Achieved **high validation accuracy** through architectural improvements.
- Loss and accuracy graphs provide insight into model convergence.

---

