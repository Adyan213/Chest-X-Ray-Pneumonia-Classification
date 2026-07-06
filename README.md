# Chest X-Ray Pneumonia Classification

A deep learning pipeline for binary classification of chest X-rays into Normal and Pneumonia categories, built with PyTorch. Includes from-scratch Grad-CAM implementation for model interpretability.

## Notebook
[View Baseline Notebook](https://github.com/Adyan213/Chest-X-Ray-Pneumonia-Classification/blob/main/chest-x-ray-pneumonia-baseline.ipynb)

[View DenseNet121 Notebook](https://github.com/Adyan213/Chest-X-Ray-Pneumonia-Classification/blob/main/chest-x-ray-pneumonia-densenet121.ipynb)

[View Grad-CAM Notebook](https://github.com/Adyan213/Chest-X-Ray-Pneumonia-Classification/blob/main/chest-x-ray-pneumonia-grad-cam.ipynb)

## Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney — 5,800+ chest X-ray images from Guangzhou Women and Children's Medical Center.

## Pipeline
- Custom data loader with train/val/test split (fixed imbalanced val set from 16 to 400+ images)
- Data augmentation: random horizontal flip, random rotation
- Class imbalance handling via weighted loss function (1:3 Normal:Pneumonia ratio)
- Transfer learning with DenseNet121 pretrained on ImageNet
- From-scratch Grad-CAM implementation for visual interpretability

## Results

| Metric | Baseline CNN | DenseNet121 |
|---|---|---|
| Test Accuracy | 83% | 87% |
| Normal Recall | 60% | 82% |
| Pneumonia Recall | 98% | 90% |
| Macro F1 | 0.81 | 0.86 |
| AUC-ROC | 0.9431 | 0.9384 |

## Grad-CAM Visualizations

Grad-CAM heatmaps show which regions the model focuses on when making predictions. Red regions indicate high attention, blue indicates low attention.
<img width="2101" height="1771" alt="gradcam_visualization" src="https://github.com/user-attachments/assets/7e0f9148-d23f-4c57-a71f-920346c6b7b1" />
The model correctly focuses on lower and middle lung fields for Pneumonia cases, consistent with where consolidations appear clinically. Misclassifications occur when the model incorrectly interprets normal anatomical structures as consolidations.

## Tech Stack
Python, PyTorch, torchvision, scikit-learn, matplotlib
