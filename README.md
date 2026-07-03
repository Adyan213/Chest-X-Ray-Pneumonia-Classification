# Chest X-Ray Pneumonia Classification

A deep learning pipeline for binary classification of chest X-rays into Normal and Pneumonia categories, built with PyTorch.

## Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney — 5,800+ chest X-ray images from Guangzhou Women and Children's Medical Center.

## Pipeline
- Custom data loader with train/val/test split (fixed imbalanced val set from 16 → 400+ images)
- Data augmentation: random horizontal flip, random rotation
- Class imbalance handling via weighted loss function (1:3 Normal:Pneumonia ratio)

## Model
Custom CNN with 3 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool) followed by 2 fully connected layers.

## Results
| Metric | Score |
|---|---|
| Test Accuracy | 83% |
| Normal Recall | 62% |
| Pneumonia Recall | 96% |
| Macro F1 | 0.81 |
| AUC-ROC | 0.927 |

## Coming Soon
- Transfer learning with DenseNet121
- Grad-CAM visualizations
- Comparison of baseline CNN vs pretrained model

## Tech Stack
Python, PyTorch, torchvision, scikit-learn, matplotlib

## Notebook
[View Notebook on nbviewer](https://nbviewer.org/github/Adyan213/Chest-X-Ray-Pneumonia-Classification/blob/main/chest-x-ray-pneumonia.ipynb) 
