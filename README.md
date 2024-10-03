---
license: mit
title: adCNN-simple
sdk: gradio
emoji: 📚
app_file: src/app.py
---
# Alzheimer's Disease Classification with CNN

This project implements a bare-bones Convolutional Neural Network (CNN) using PyTorch to classify 2D brain slices of MRI images as indicating Alzheimer's disease or not. The model is trained on the OASIS MRI dataset.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up the Conda Environment](#2-set-up-the-conda-environment)

## Dataset

We use the **OASIS MRI dataset**, which consists of brain MRI images categorized into four classes based on Alzheimer's progression. For simplicity, we combine the demented classes into a single category.

- **Download the Dataset:**
  - The dataset is available on Kaggle: [Alzheimer's Dataset (4 Class of Images)](https://www.kaggle.com/datasets/ninadaithal/imagesoasis).
  - You will need a Kaggle account to download the dataset.

## Prerequisites

- **Conda:** For managing the Python environment.
- **Python 3.8 or higher**
- **Git:** For cloning the repository.

## Installation

Open your terminal and run:

```bash
git clone https://github.com/adMemorAI/adCNN
cd adCNN
conda env create -f environment.yml
conda activate adCNN
```
