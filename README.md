# Luna 16

Luna 16 is a machine learning project focused on developing and training models for medical image analysis. The project aims to leverage advanced techniques in deep learning to improve the accuracy and efficiency of image-based diagnostics.

TBD: Explain this project is still in progress.
It was prepared for experimenting with different models but right now it only implements most basic "real" model.

## Overview

The Luna 16 project is dedicated to the development and training of machine learning models specifically designed for medical image analysis. By utilizing state-of-the-art deep learning techniques, the project aims to enhance the precision and speed of diagnostic processes based on medical images such as CT scans.

### Key Features

- **Advanced Deep Learning Models**: Luna 16 employs cutting-edge neural network architectures to analyze and interpret medical images.
- **Automated Data Processing**: The project includes robust data processing pipelines to handle large volumes of medical imaging data efficiently.
- **Comprehensive Training Framework**: A well-structured training module is provided to facilitate the training and fine-tuning of models.
- **Monitoring and Logging**: Integrated tools for monitoring model performance and system metrics, including MLFlow and TensorBoard, ensure that the training process is transparent and well-documented.
- **Cloud Experimentation**: The project supports running experiments on cloud platforms like RunPod, enabling scalable and flexible model training.

### Technologies Used

- **PyTorch**: For building and training deep learning models.
- **MLFlow**: For experiment tracking, model management, and system metrics logging.
- **TensorBoard**: For visualizing training metrics and model performance.
- **Docker**: To create reproducible environments for development and deployment.
- **Ray**: For distributed hyperparameter tuning.

By integrating these technologies, Luna 16 aims to push the boundaries of medical image analysis, providing tools and models that can significantly aid in the diagnostic process.

### Architecture

#### Model Module

The Model Module in Luna 16 is designed to provide a structured and flexible interface for building and training machine learning models, specifically tailored for medical image analysis.

##### Interface (`models/base.py`)

The base model interface defines the essential methods that any model implementation should have. These include methods for fitting the model to training data, validating the model, and obtaining the model's signature. The interface ensures that all models adhere to a consistent structure, making it easier to integrate and manage different models within the project.

Key methods include:

- `fit_epoch`: Trains the model for one epoch.
- `get_module`: Returns the underlying PyTorch module.
- `get_signature`: Generates a model signature for MLFlow logging.

##### Example Implementation (`classification_model.py`)

The `NoduleClassificationModel` class implements the base model interface and provides the specific logic for training and validating a model designed to classify nodules in medical images. This class includes methods for:

- Initializing the model with necessary components such as the neural network module, optimizer, batch iterator, and logger.
- Training the model (`do_training`) and validating it (`do_validation`).
- Computing the loss for each batch (`compute_batch_loss`).
- Logging metrics to monitor the model's performance (`log_metrics`).

The `NoduleClassificationModel` is designed to be flexible and extensible, allowing for fine-tuning and adaptation to different datasets and training requirements. It leverages PyTorch for model building and training, and integrates with MLFlow for experiment tracking and logging.

##### Usage

To use the Model Module, instantiate the `NoduleClassificationModel` with the appropriate neural network module, optimizer, and other dependencies. Then, use the `fit_epoch` method to train the model for the desired number of epochs, and monitor the training process using the integrated logging and monitoring tools.

By following this structured approach, the Model Module ensures that the development and training of machine learning models in Luna 16 are efficient, reproducible, and easy to manage.

#### Data Modules

TBD

#### Training Module

TBD

#### Launcher Modules

TBD

### Data Processing

TBD

### Monitoring

TBD

#### Logs

TBD

#### Metrics

TBD

#### Dashboards

TBD

## Usage

TBD

### Training & Continuing Training

TBD

### Monitoring & Dashboards

TBD

#### MLFlow

TBD

#### TensorBoard

TBD

## Cloud Experiment

TBD: Write notes on experiment running on RunPod.

### Monitoring Usage

TBD

#### Monitoring Model

Both MLFlow and Tensortboard can visualize metrics in real time. MLFlow even visualizes system metrics for CPU, GPU, and memory.

#### Monitoring CPU and Processes

TBD

```bash
htop
```

#### Monitoring GPU

TBD

```bash
watch -n 1 nvidia-smi
```

# Sources

TBD: Mention Books and Papers

Dive into Deep Learning
Designing Machine Learning Systems
Deep Learning with PyTorch
