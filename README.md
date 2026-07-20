# nde-signal-ultrasonic-defect-classification
An end-to-end NDE signal processing and deep learning pipeline combining Butterworth filtering, Discrete/Continuous Wavelet Denoising, and PyTorch 1D CNNs to classify ultrasonic flaw signatures.
# 🔬 NDE Ultrasonic Signal Denoising & Defect Detection using PyTorch & 1D CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![DSP](https://img.shields.io/badge/DSP-Butterworth%20%7C%20Wavelets-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview
In Industrial **Non-Destructive Evaluation (NDE)**, ultrasonic A-scan signals are frequently contaminated by background acoustic noise, attenuation, and structural scattering. This project implements an end-to-end Signal Processing and Deep Learning pipeline capable of:

1. **Filtering & Noise Reduction:** Combining a 4th-order Butterworth Bandpass filter with Soft-Threshold Wavelet Denoising (`db4` wavelet) to enhance Signal-to-Noise Ratio (SNR).
2. **Automated Defect Feature Extraction & Classification:** Training a custom **1D Convolutional Neural Network (1D CNN)** in PyTorch to detect structural flaw echoes embedded within high-noise time-series signals.

---

## 🛠️ Pipeline Architecture
