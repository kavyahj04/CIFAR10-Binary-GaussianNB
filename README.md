# CIFAR-10 Binary Classification — Gaussian Naive Bayes

Binary image classifier built from scratch to distinguish between two CIFAR-10 classes (Bird vs Ship) using Gaussian Naive Bayes — no sklearn for the classifier.

## What This Project Does

- Loads CIFAR-10 dataset using PyTorch
- Selects 2000 images per class (Bird vs Ship)
- Center-crops each 32×32 image to 10×10 → flattens to 300-D feature vector
- Trains a Gaussian Naive Bayes classifier manually using NumPy
- Evaluates accuracy on test set

## Tech Stack
- Python
- PyTorch + TorchVision (data loading only)
- NumPy
- Matplotlib
- Google Colab

## Key Concepts Covered
- Image preprocessing and center cropping
- Feature extraction from raw pixels
- Gaussian Naive Bayes (manual implementation)
- Binary classification (Bird=0, Ship=1)
- Reproducible experiments with fixed seeds

## How to Run

Open `CIFAR10_Binary_GaussianNB.ipynb` in Google Colab and run all cells.
Dataset downloads automatically on first run.
