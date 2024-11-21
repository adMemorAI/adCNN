# Alzheimer Recognition with Deep Learning

## Table of Contents

1. Project Overview
2. Features
3. Installation
4. Dataset Structure
5. Models
6. Training & Hyperparameter Tuning
7. Project Structure
8. Usage



## Project Overview

This project aims to build a robust neural network system to classify whether a person has Alzheimer's disease based on MRI scan slices. By utilizing different datasets and models, the project seeks to find the most accurate approach for Alzheimer's recognition.

## Features

- **Flexible Dataset Integration**: Easily switch between the OASISKaggle and ADNI datasets.
- **Multiple Model Support**: Choose between ResNet50 and Convolutional Visual Transformer (CvT) models.
- **Hyperparameter Tuning**: Optimize model performance using Weights & Biases (W&B) for hyperparameter sweeps.
- **Binary Classification**: Focused on identifying the presence or absence of Alzheimer's disease.

## Installation

1. **Clone the Repository**

   `git clone https://github.com/yourusername/alzheimer-recognition.git`

   `cd alzheimer-recognition`

2. **Create a Virtual Environment**

   `python3 -m venv venv`

   `source venv/bin/activate`  (On Windows: `venv\Scripts\activate`)

3. **Install Dependencies**

   `pip install -r requirements.txt`

4. **Setup Weights & Biases**

   Sign up for a free account at Weights & Biases and log in:

   `wandb login`

## Dataset Structure

The project supports two datasets: OASISKaggle and ADNI. Both datasets are organized in a structured manner to facilitate easy loading and processing.

```
datasets/
├── adni
│   ├── AD
│   │   ├── Subject_01
│   │   │   ├── Scan_01
│   │   │   │   ├── slice1.png
│   │   │   │   ├── slice2.png
│   │   │   │   └── ...
│   │   │   ├── Scan_02
│   │   │   └── Scan_03
│   │   ├── Subject_02
│   │   └── ...
├── oasis_kaggle/
│   ├── no-dementia/
│   ├── verymild-dementia/
│   ├── mild-dementia/
│   └── moderate-dementia/
├── config.yaml
└── sweep_config.yaml
```

- **AD**: Alzheimer's Disease
- **CN**: Cognitively Normal
- **MCI**: Mild Cognitive Impairment

Each class folder contains subject folders, which in turn contain scan folders with MRI slice images.

## Models

The project currently supports two models:

- **ResNet50**: A widely-used convolutional neural network architecture.
- **Convolutional Visual Transformer (CvT)**: Combines convolutional layers with transformer architectures for enhanced feature extraction.

## Training & Hyperparameter Tuning

Using Weights & Biases (W&B), you can perform hyperparameter sweeps to find the optimal model configuration.

### Configuration Files

- **config.yaml**: Contains default configurations for datasets, models, training parameters, and data transformations.
- **sweep\_config.yaml**: Defines the hyperparameter sweep strategy using Bayesian optimization.

### Running a Training Sweep

1. **Initialize the Sweep**

   `wandb sweep sweep_config.yaml`

2. **Start the Sweep Agents**

   `wandb agent your-sweep-id`

   Replace `your-sweep-id` with the ID returned after initializing the sweep.

## Project Structure

Here's an overview of the project's directory structure:

```
src/
├── dsets/
│   ├── adni.py
│   ├── dataset_factory.py
│   ├── oasis_kaggle.py
│   └── __init__.py
├── losses/
│   ├── focal_loss.py
│   └── __init__.py
├── models/
│   ├── CvT.py
│   ├── ResAD.py
│   ├── model_factory.py
│   └── __init__.py
├── utils/
│   ├── config.py
│   ├── data_loader.py
│   ├── factory_utils.py
│   ├── metrics.py
│   ├── model_utils.py
│   └── wandb_utils.py
├── evaluate_model.py
├── initialize_model.py
├── main.py
├── train_model.py
└── module.py
datasets/
├── adni/
├── oasis_kaggle/
├── config.yaml
└── sweep_config.yaml
```

- **dsets/**: Contains dataset classes and the factory to load them.
- **models/**: Houses model architectures and a factory for model instantiation.
- **losses/**: Custom loss functions.
- **utils/**: Utility scripts for configuration, data loading, metrics, and W&B integration.
- **src/**: Core scripts for training, evaluation, and initialization.

## Usage

### Training the Model

To start training with the default configuration:

`python src/main.py --step train`

##
