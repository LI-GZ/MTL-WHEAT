# MTL-WHEAT: Multi-Task Learning Framework for Winter Wheat Phenotyping

This repository contains the official implementation of the environment-adaptive Multi-Task Learning (MTL) framework. It is designed for the simultaneous prediction of Leaf Area Index (LAI) and Grain Yield (GY) in winter wheat across multi-environment trials using hyperspectral data.

## 📌 Features
- **Spectral Preprocessing:** Includes Multiplicative Scatter Correction (MSC) and Competitive Adaptive Reweighted Sampling (CARS) to extract physiologically meaningful spectral features.
- **Environment-Adaptive MTL:** A PyTorch-based hard-parameter sharing network with environment-specific routing branches to address G×E (Genotype × Environment) interactions.
- **High Efficiency:** Requires fewer parameters than traditional deep learning models while maintaining strong generalization capabilities.

## 📂 Repository Structure
- `MSC+CARS.py`: Pipeline for hyperspectral data preprocessing and feature selection.
- `Multi-Task Learning Framework.py`: The PyTorch implementation of the environment-adaptive MTL model.
- `requirements.txt`: List of required Python packages.
- `data/`: Directory for input datasets (empty by default, requires user to provide data).
- `output/`: Auto-generated directory for saving results and model metrics.

## ⚙️ Environment Setup
It is recommended to use a virtual environment. Install the required dependencies via pip:

```bash
pip install -r requirements.txt
