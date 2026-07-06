# Chest X-Ray Pneumonia Classification

A deep learning pipeline for binary classification of chest X-rays into Normal and Pneumonia categories, built with PyTorch.

## Notebook
[View Baseline Notebook on nbviewer](https://nbviewer.org/github/Adyan213/Chest-X-Ray-Pneumonia-Classification/blob/main/chest-x-ray-pneumonia.ipynb)

[View DenseNet121 Notebook on nbviewer](https://nbviewer.org/github/Adyan213/Chest-X-Ray-Pneumonia-Classification/blob/main/chest-x-ray-pneumonia-densenet121.ipynb)

## Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney — 5,800+ chest X-ray images from Guangzhou Women and Children's Medical Center.

## Pipeline
- Custom data loader with train/val/test split (fixed imbalanced val set from 16 to 400+ images)
- Data augmentation: random horizontal flip, random rotation
- Class imbalance handling via weighted loss function (1:3 Normal:Pneumonia ratio)
- Transfer learning with DenseNet121 pretrained on ImageNet

## Results

| Metric | Baseline CNN | DenseNet121 |
|---|---|---|
| Test Accuracy | 83% | 87% |
| Normal Recall | 60% | 82% |
| Pneumonia Recall | 98% | 90% |
| Macro F1 | 0.81 | 0.86 |
| AUC-ROC | 0.9431 | 0.9384 |

## Coming Soon
- Grad-CAM visualizations
- Fine-tuning pretrained layers

## Tech Stack
Python, PyTorch, torchvision, scikit-learn, matplotlib
