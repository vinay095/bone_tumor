# Multimodal Bone Tumor Diagnosis Using Radiomics and Deep Features

## Overview

This repository presents an end-to-end multimodal pipeline for bone tumor analysis in X-ray radiographs.

The system combines:
- A YOLOv8-seg-based automated lesion localization module
- A radiomics feature extraction engine (GLCM + LBP)
- A deep feature extractor (EfficientNet-B0)
- A fused feature representation for downstream classification tasks

The primary contribution of this work is the construction of a structured multimodal feature dataset derived from automatically localized regions of interest.

### Project Architecture
The pipeline is divided into three core phases:

**1. Localization & Segmentation (YOLOv8s-seg)**
* **Dataset:** BTXRD (Bone Tumor X-Ray Dataset). Custom scripts were developed to normalize LabelMe JSON polygon/rectangle annotations into standard YOLO segmentation formats.
* **Training:** Trained the model for 200 epochs using dual T4 GPUs. 
* **Performance:** Achieved real-time inference speeds (3.5ms per image), generating a lightweight, deployment-ready `best.pt` weights file (23.9MB).
  ( Despite moderate detection performance, the model provides sufficient ROI proposals to support feature-based analysis in the downstream pipeline. )

**2. Multimodal Feature Extraction**
For every detected lesion, the pipeline automatically crops the Region of Interest (ROI) and extracts the following features:
* **GLCM (Macro-Texture):** Calculates `Contrast` and `Energy` to quantify the physical graininess and uniformity of bone destruction.
* **LBP (Micro-Patterns):** Calculates Local Binary Pattern `Entropy` to measure the microscopic chaos of the bone tissue, a key indicator of aggressive lesions.
* **Deep Features (EfficientNet-B0):** Utilizes transfer learning to pass the ROI through a pre-trained EfficientNet model, extracting a high-dimensional mathematical fingerprint.
* **Fusion Score:** Engineered a novel combinatorial metric mapping physical texture chaos against deep feature magnitudes.

**3. Data Aggregation**
* Generates a master dataset (`tumor_features_final_200e.csv`) containing over 1,300 clinical and deep learning features, formatting the data for seamless ingestion by downstream classifiers (e.g., Random Forest, SVM).

---

### Model Metrics (Epoch 200)
Validation was performed on a strictly separated holdout set of 373 X-ray images containing 926 instances.

| Metric | Score |
|---------|---------|
| Precision | 0.645 |
| Recall | 0.326 |
| mAP50 | 0.349 |
| Inference Speed | 3.5 ms |

---

## Ground Truth vs Prediction

<table>
<tr>
<td align="center"><b>Ground Truth</b></td>
<td align="center"><b>YOLOv8 Prediction</b></td>
</tr>
<tr>
<td><img src="val_batch1_labels.jpg" width="450"></td>
<td><img src="val_batch1_pred.jpg" width="450"></td>
</tr>
</table>

---

## Confusion Matrix

<p align="center">
  <img src="confusion_matrix.png" width="700">
</p>

---

## Feature Correlation Analysis

<p align="center">
  <img src="Feature_Correlation_Matrix.png" width="700">
</p>

---

### Repository Structure
* `data_preprocessing/` - Scripts for converting LabelMe JSONs to YOLO format and generating Train/Val splits.
* `training/` - Configuration files and Kaggle notebook scripts for the YOLOv8 200-epoch run.
* `feature_extraction/` - The core engine utilizing `scikit-image` and `efficientnet_pytorch` to generate the multimodal CSV.
* `visualizations/` - Scripts for generating feature correlation matrices and scatter plots comparing GLCM texture against AI Fusion scores.

### Future Work (Phase 4)
The next iteration of this project will utilize the generated `tumor_features_final_200e.csv` to train a downstream Random Forest classifier. 
