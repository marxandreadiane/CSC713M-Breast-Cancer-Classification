# Breast Cancer Classification (BUSI + BUS-UCLM)

This repository contains a breast ultrasound image classification project (benign, malignant, normal) using:
- classical ML (Random Forest)
- transfer learning with MobileNetV2
- transfer learning with ResNet50

## Repository Structure

- `[CSC713M]_Twin_Fairy_Tacticians_MCO(1).ipynb`
  - Main experiment notebook.
  - Runs data loading, preprocessing, training, evaluation, and final visualizations.

- `data_utils.py`
  - Dataset path helpers.
  - BUSI loader.
  - BUS-UCLM loader.
  - Mask-based label extraction for BUS-UCLM.
  - Dataset merge utility.

- `preprocessing_utils.py`
  - Image preprocessing (grayscale conversion, resize, CLAHE, normalization).
  - Label encoding.
  - Train/validation/test splitting.
  - Class-targeted augmentation utilities.
  - Flattening for Random Forest.
  - Grayscale-to-RGB conversion for CNN backbones.

- `model_utils.py`
  - Random Forest model builder.
  - MobileNetV2 model builders:
    - basic
    - improved (regularized)
    - final (stronger regularization)
  - ResNet50 basic model builder.

- `evaluation_utils.py`
  - Unified test evaluators for sklearn and Keras models.
  - Prints test metrics (including per-class precision/recall/F1).
  - Confusion matrix plotting.
  - Comparative plots and final dashboard utilities.

- `Datasets/`
  - Local dataset folder expected by the notebook.
  - Includes BUSI and BUS-UCLM subfolders.

