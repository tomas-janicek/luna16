# Luna 16

Luna 16 is a machine learning project focused on developing and training models for medical image analysis. The project aims to leverage advanced techniques in deep learning to improve the accuracy and efficiency of image-based diagnostics.

## 🌐 Overview

The Luna 16 project is dedicated to the development and training of machine learning models specifically designed for medical image analysis. By utilizing state-of-the-art deep learning techniques, the project aims to enhance the precision and speed of diagnostic processes based on medical images such as CT scans.

## 🛠️ Technologies Used

- **PyTorch**: For building and training deep learning models.
- **MLFlow**: For experiment tracking, model management, and system metrics logging.
- **TensorBoard**: For visualizing training metrics and model performance.
- **Docker**: To create reproducible environments for development and deployment.
- **Ray**: For distributed hyperparameter tuning.

By integrating these technologies, Luna 16 aims to push the boundaries of medical image analysis, providing tools and models that can significantly aid in the diagnostic process.

## 🎯 Motivation

This project primarily serves educational purposes, guiding users through the entire process of model development. It emphasizes the importance of infrastructure components like MLFlow and TensorBoard for monitoring and logging, which are beneficial for any machine learning project. Additionally, it incorporates Docker for reproducibility and Ray for hyperparameter tuning. The project is designed with modularity in mind, allowing easy replacement of components such as models, datasets, trainers, and loss functions. As a result, it can serve as a starting point for any other machine learning project, particularly those focused on experimentation. Essentially, it acts as a template for any machine learning project, providing a robust foundation for future work.

## ✨ Features

- **Advanced Deep Learning Models**: Luna 16 employs cutting-edge neural network architectures to analyze and interpret medical images.
- **Automated Data Processing**: The project includes robust data processing pipelines to handle large volumes of medical imaging data efficiently.
- **Comprehensive Training Framework**: A well-structured training module is provided to facilitate the training and fine-tuning of models.
- **Monitoring and Logging**: Integrated tools for monitoring model performance and system metrics, including MLFlow and TensorBoard, ensure that the training process is transparent and well-documented.
- **Cloud Experimentation**: The project supports running experiments on cloud platforms like RunPod, enabling scalable and flexible model training.

## 🚀 Next Steps

This project is still in progress and was initially prepared for experimenting with different models. Currently, it only implements the most basic "real" model. Several key components are missing, including:

- Segmentation of CT scans. The project currently focuses on classification tasks, using cutouts of CT scans. However, adding segmentation capabilities to extract nodules from CT scans directly is planned.
- Baseline and other models. I would particularly like to add a Zero Rule Baseline and non-neural network models. Additionally, incorporating other neural network models with different architectures and convolution techniques is planned.
- Hyperparameter tuning. The project aims to integrate Ray for distributed hyperparameter tuning to optimize model performance.
- Cloud deployment. The project will include tools and scripts for deploying models to cloud platforms for inference and production use.
- Additional datasets. The project will include more datasets for training and validation, covering a wider range of medical imaging tasks and modalities.

Current model improvements:
- Add gradient clipping.


## 🏛️ Architecture

The Luna 16 project is designed with a modular architecture that separates concerns and promotes reusability and extensibility.

### 🗃️ Model Module

The Model Module in Luna 16 is designed to provide a structured and flexible interface for building and training machine learning models, specifically tailored for medical image analysis.

#### 📝 Interface (`models/base.py`)

The base model interface defines the essential methods that any model implementation should have. These include methods for fitting the model to training data, validating the model, and obtaining the model's signature. The interface ensures that all models adhere to a consistent structure, making it easier to integrate and manage different models within the project.

Key methods include:

- `fit_epoch`: Trains the model for one epoch.
- `get_module`: Returns the underlying PyTorch module.
- `get_signature`: Generates a model signature for MLFlow logging.

#### 🧩 Example Implementation (`classification_model.py`)

The `NoduleClassificationModel` class implements the base model interface and provides the specific logic for training and validating a model designed to classify nodules in medical images. This class includes methods for:

- Initializing the model with necessary components such as the neural network module, optimizer, batch iterator, and logger.
- Training the model (`do_training`) and validating it (`do_validation`).
- Computing the loss for each batch (`compute_batch_loss`).
- Logging metrics to monitor the model's performance (`log_metrics`).

The `NoduleClassificationModel` is designed to be flexible and extensible, allowing for fine-tuning and adaptation to different datasets and training requirements. It leverages PyTorch for model building and training, and integrates with MLFlow for experiment tracking and logging.

#### 📚 Usage

To use the Model Module, instantiate the `NoduleClassificationModel` with the appropriate neural network module, optimizer, and other dependencies. Then, use the `fit_epoch` method to train the model for the desired number of epochs, and monitor the training process using the integrated logging and monitoring tools.

By following this structured approach, the Model Module ensures that the development and training of machine learning models in Luna 16 are efficient, reproducible, and easy to manage.

### 📊 Data Modules

`DataModule` provides a structured and efficient way to handle datasets, making it easier to train and validate models. It manages the creation of data loaders, which are essential for batching and shuffling data during training and validation processes.

#### 📚 Usage

The `DataModule` takes in training and validation datasets and creates corresponding data loaders. These data loaders are responsible for batching the data and ensuring that it is fed into the model in an optimal manner. The class also handles device configuration, ensuring that data is loaded onto the appropriate device (CPU or GPU) for training.

Key methods include:

- **`__init__`**: Initializes the `DataModule` with batch size, training dataset, and validation dataset. It also sets up the device configuration and creates data loaders.
- **`get_dataloader`**: Returns the appropriate data loader (training or validation) based on the input flag.

By using the `DataModule`, Luna 16 ensures that data handling is efficient and consistent, allowing for smooth and effective training and validation of machine learning models.

### 🏋️ Training Module

The Training Module in Luna 16 is responsible for orchestrating the training process of machine learning models. By using the `Trainer` and `BaseTrainer`, the Luna 16 project ensures that the training process is modular, extensible, and easy to manage.

#### 🏗️ BaseTrainer

The `BaseTrainer` is a protocol that defines the essential methods required for training a model. It ensures that any trainer implementation adheres to a consistent interface, making it easier to manage and extend the training process. The key methods defined in `BaseTrainer` include:

- `fit`: Trains the model for a specified number of epochs.
- `fit_epoch`: Trains the model for a single epoch.

#### 🏋️‍♂️ Trainer

The `Trainer` class implements the `BaseTrainer` protocol and provides the concrete logic for training models. It handles the entire training lifecycle, including:

- Initializing the trainer with a name, version, and logger.
- Managing the training loop, including epoch-wise training and logging.
- Profiling the training process to optimize performance.

The `Trainer` class ensures that the training process is well-organized, efficient, and easy to monitor. It integrates with logging and profiling tools to provide detailed insights into the training process.

### 🚀 Launcher Modules

The Launcher Modules in Luna 16 serve as the entry points for training and fine-tuning machine learning models. They encapsulate the logic required to set up, train, and evaluate models, making it easier to manage the entire training lifecycle.

#### 🚀 LunaClassificationLauncher

The `LunaClassificationLauncher` is designed to handle the training of classification models. It integrates various services and components to streamline the training process. Key responsibilities include:

- **Initialization**: Sets up the training environment, including logging and batch iterators.
- **Model Training**: Provides methods to train models from scratch (`fit`) or continue training from a saved state (`load_fit`).
- **Hyperparameter Tuning**: Supports distributed hyperparameter tuning using Ray (`tune_parameters`).

#### 🚀 LunaMalignantClassificationLauncher

The `LunaMalignantClassificationLauncher` is similar to the `LunaClassificationLauncher` but is specifically tailored for training models that classify malignant nodules in medical images. It shares the same core functionalities but is configured to handle the specific requirements of malignant classification tasks.

#### 📚 Usage in CLI

In the CLI, these launcher modules act as "endpoints" that bring together various services and components to execute training tasks. For example, the `train_luna_classification` command uses the `LunaClassificationLauncher` to initiate the training process.

### 🛠️ Tooling

#### 🖼️ Augmentations

Augmentations are techniques used to artificially expand the size and variability of a training dataset by applying random transformations to the input data. This helps improve the robustness and generalization of machine learning models by exposing them to a wider range of variations during training.

### 🖼️ Augmentations Used in Luna 16

- **Flip**: Randomly flips the image along specified dimensions to introduce variability in orientation.
- **Offset**: Applies random translations to the image, shifting it by a specified offset to simulate different positions.
- **Scale**: Randomly scales the image by a factor to introduce variability in size.
- **Rotate**: Rotates the image by a random angle to simulate different orientations.
- **Noise**: Adds random noise to the image to simulate variations in image quality and improve model robustness.

These augmentations help ensure that the models trained in Luna 16 are more resilient to variations in the input data, leading to better performance on unseen data.

#### 🔄 Batch Iterators

### 🔄 Batch Iterators

#### 🏗️ BaseIteratorProvider

The `BaseIteratorProvider` class defines the interface for batch iterators in the Luna 16 project.

Key method:
- `enumerate_batches`: An abstract method that must be implemented by any subclass to iterate over batches of data, providing the necessary logging and handling during the iteration process.

#### 🔄 BatchIteratorProvider

The `BatchIteratorProvider` class is responsible for managing the iteration over batches of data during the training and validation processes. It logs the start and end of batch processing, as well as periodic updates during the iteration. This class ensures that the training loop is well-documented and that progress can be monitored effectively.

#### 🛠️ Services

### 🛠️ Service Container

The `ServiceContainer` is a core component in Luna 16 that manages the lifecycle of various services used throughout the project.

#### ⚙️ How It Works

The `ServiceContainer` class allows you to register services directly or through creator functions. It also supports registering cleanup functions that are called when the services are closed. This ensures that resources are properly managed and released when they are no longer needed.

Key methods include:

- `register_service`: Registers a service instance directly.
- `register_creator`: Registers a function that creates a service instance.
- `call_all_creators`: Calls all registered creator functions to instantiate services.
- `get_service`: Retrieves a registered service instance.
- `close_all_services`: Calls all registered cleanup functions to close services.

By using the `ServiceContainer`, Luna 16 ensures that services are managed in a consistent and efficient manner, reducing the risk of resource leaks and making the codebase easier to maintain.

## 📊 Data Processing

`CtCutoutService` is a service class responsible for creating cutouts from CT scans for candidate nodules.
It manages the process of reading CT scan data, extracting relevant cutouts, and saving them for further analysis.

The `CtCutoutService` uses the `Ct` class to read CT scans and extract cutouts for each candidate nodule. It then saves these cutouts to disk and updates the list of processed candidates.

### 🗂️ Datasets

#### 🗂️ CutoutsDataset

The `CutoutsDataset` class is a custom dataset designed for handling cutouts of CT scans. These cutouts are small, focused sections of larger CT scans that contain candidate nodules. The primary purpose of this dataset is to facilitate the training and validation of machine learning models by providing a structured way to access and manipulate these cutouts.

- **Initialization**: The dataset is initialized with parameters such as the ratio of positive to negative samples, whether the dataset is for training or validation, and any transformations or filters to be applied to the data.
- **Data Splitting**: The dataset splits the candidates into nodules and non-nodules, as well as malignant and non-malignant categories. This allows for balanced sampling during training and validation.
- **Data Loading**: The `__getitem__` method retrieves a specific cutout based on the index, applies any specified transformations and filters, and returns the processed data along with the corresponding labels.

#### 🗂️ MalignantCutoutsDataset

The `MalignantCutoutsDataset` class extends the `CutoutsDataset` to specifically handle datasets focused on malignant nodules. It overrides methods to ensure that the dataset includes a balanced mix of malignant, benign, and non-nodule samples.

- **Initialization**: Inherits initialization parameters from `CutoutsDataset` and ensures that the dataset is tailored for malignant classification tasks.
- **Data Loading**: The `__getitem__` method is overridden to handle the specific requirements of malignant classification, ensuring that the correct labels are assigned to each cutout.

#### 📚 Usage of Files Generated by CtCutoutService

Both `CutoutsDataset` and `MalignantCutoutsDataset` utilize the cutout files generated by the `CtCutoutService`. The `CtCutoutService` processes CT scans to create small, focused cutouts around candidate nodules and saves these cutouts as `.npz` files. These files contain the CT chunk data and metadata such as the center coordinates of the cutout.

1) **File Loading**: When a dataset instance retrieves a cutout, it loads the corresponding `.npz` file generated by the `CtCutoutService`.
2) **Data Processing**: The loaded data is then processed through any specified transformations and filters to prepare it for model training or validation.

## 📓 Notebooks

TBD

## 📈 Monitoring & Logging

Monitoring and logging are essential components of the Luna 16 project, providing visibility into the training process, model performance, and system metrics.

### 📩 Message Handler

The `BaseMessageHandler` and `MessageHandler` classes in Luna 16 are used to log, monitor, and track metrics during the training and validation processes. These handlers facilitate communication between different components of the system through a structured messaging protocol.

#### 📩 BaseMessageHandler

The `BaseMessageHandler` is a protocol that defines the essential method `handle_message` which any message handler implementation should have. This method takes a `message` of type `Message` and a `registry` of type `services.ServiceContainer`. The protocol ensures that all message handlers adhere to a consistent interface, making it easier to manage and extend the logging and monitoring functionalities.

#### 📩 MessageHandler

The `MessageHandler` class implements the `BaseMessageHandler` protocol and provides the concrete logic for handling messages. The `handle_message` method processes incoming messages by invoking the appropriate handlers based on the message type.

#### ⚙️ How It Works

1. **Message Dataclasses**: The project defines various message dataclasses in `messages.py`, such as `LogMetrics`, `LogStart`, `LogEpoch`, `LogBatchStart`, `LogBatch`, `LogBatchEnd`, `LogResult`, `LogParams`, and `LogModel`. These dataclasses encapsulate different types of information that need to be logged and monitored during the training process.
2. **Logging and Monitoring**: When a specific event occurs (e.g., the start of training, the end of a batch, or the computation of metrics), an instance of the corresponding message dataclass is created and passed to the `handle_message` method of the `MessageHandler`. The handler then processes the message by invoking the appropriate registered handlers, which log the information to the desired logging and monitoring tools (e.g., MLFlow, TensorBoard).
3. **Usage in Models**: In the `BaseModel` classes, the `MessageHandler` is used to log metrics and results during training and validation. For example, the `log_metrics` method creates instances of `LogMetrics` and `LogResult` messages and passes them to the `handle_message` method of the `MessageHandler`. This ensures that all relevant information is logged and monitored consistently throughout the training process.

By using the `MessageHandler` classes, Luna 16 ensures that logging and monitoring are modular, extensible, and easy to manage. The structured messaging protocol allows for clear and consistent communication between different components, facilitating effective tracking of metrics and system performance.

### 📊 Dashboards

Luna 16 provides integrated dashboards for monitoring and visualizing training metrics, model performance, and system statistics. These dashboards leverage tools such as MLFlow and TensorBoard to provide real-time insights into the training process.

## 📦 Installation

To install Luna16, you can use `uv`:

For CPU version:

```sh
uv sync --extra cpu
```

For CUDA version:

```sh
uv sync --extra cu124
```

## 🧪 Running Tests

To run the tests, use the following command:

```sh
PYTHONPATH=. uv run pytest luna16/tests/
```

## 📘 Usage

Luna 16 provides a command-line interface (CLI) for running various tasks related to model training, evaluation, and experimentation.

### Creating Cutouts from CT images

TBD

```sh
PYTHONPATH=. uv run python -m luna16.cli create_cutouts
```

### 🏋️ Training & Continuing Training

Luna 16 supports training models from scratch as well as continuing training from a saved state. The training process can be customized with parameters such as the number of epochs, batch size, and learning rate.

#### 🏋️ Training from Scratch

To train a model (set to version 0.0.1) from scratch, use the following command:

```sh
PYTHONPATH=. uv run python -m luna16.cli \
                train_luna_classification \
                0.0.1 \
                --epochs=5 \
                --batch-size=64
```

The same variant can be used for malignant classification:

```sh
PYTHONPATH=. uv run python -m luna16.cli \
                train_luna_malignant_classification \
                0.0.1 \
                --epochs=5 \
                --batch-size=64
```

#### 🏋️ Continuing Training

To continue training a model (set to version 0.0.2) from a saved state (file loader will load classification model with version 0.0.1), use the following command:

```sh
PYTHONPATH=. uv run python -m luna16.cli \
                load_train_luna_classification \
                0.0.1 \
                file \
                classification \
                0.0.1 \
                --epochs=5 \
                --batch-size=64
```

The same variant can be used for malignant classification:

```sh
PYTHONPATH=. uv run python -m luna16.cli \
                load_train_luna_malignant_classification \
                0.0.1 \
                file \
                classification \
                0.0.1 \
                --epochs=5 \
                --batch-size=64
```


### 📊 Monitoring & Dashboards

Luna 16 provides integrated monitoring and dashboards for tracking training metrics, model performance, and system statistics. The dashboards can be accessed through tools such as MLFlow and TensorBoard.

#### 📊 MLFlow

To start the MLFlow server and access the tracking UI, use the following command:

```sh
make mlflow-migrate
make mlflow-server
```

#### 📊 TensorBoard

To start the TensorBoard server and access the visualization dashboard, use the following command:

```sh
make tensorboard-classification
```

The same variant can be used for malignant classification:

```sh
make tensorboard-malignant-classification
```

### 📈 Profiling

Luna 16 supports profiling the training process to optimize performance and resource utilization. The profiling tools can be used to identify bottlenecks, optimize hyperparameters, and improve training efficiency.

#### 📈 Profiling with PyTorch Profiler

To profile the training process using PyTorch Profiler, use the following command:

```sh
PYTHONPATH=. uv run python -m luna16.cli \
                train_luna_classification \
                0.0.0-profile \
                --epochs=10 \
                --batch-size=64 \
                --profile
```

The same variant can be used for malignant classification:

```sh
PYTHONPATH=. uv run python -m luna16.cli \
                train_luna_malignant_classification \
                0.0.0-profile \
                --epochs=10 \
                --batch-size=64 \
                --profile
```

### 🎛️ Hyperparameter Tuning

Hyperparameter tuning using Ray is not yet fully functional.

In this project, I can optimize several hyperparameters. 

Here are some key hyperparameters:

- **Learning Rate:** The step size used by the optimizer to update the model weights.
- **Batch Size**: The number of samples processed before the model is updated.
- **Number of Epochs**: The number of times the entire dataset is passed through the model.
- **Number of Convolutional Blocks (n_blocks)**: The number of LunaBlock layers in the model.
- **Number of Convolutional Channels (conv_channels)**: The number of output channels for the convolutional layers.
- **Dropout Rate**: The probability of dropping out neurons during training to prevent overfitting.
- **Optimizer Type**: The type of optimizer used (e.g., Adam, SGD).

## ☁️ Cloud Experiment

TBD: Write notes on experiment running on RunPod.

### 🧪 Methodology

TBD

### 📊 Results

TBD

# 📚 Sources

TBD: Mention Books and Papers

- Dive into Deep Learning
- Designing Machine Learning Systems
- Deep Learning with PyTorch
