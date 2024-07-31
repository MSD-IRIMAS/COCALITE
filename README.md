# Project Overview

The objective of this project is to enhance performance and maintain computational efficiency by integrating engineered feature vectors with features learned from deep learning models. The project explores various methodologies, and evaluates the impact of these strategies on time series classification tasks.

## Requirements

Please refer to the `requirements.txt` file and install the required packages using:

```bash
pip install -r requirements.txt
```

## Configuration

The project requires a YAML configuration file named `config.yaml`, which includes the following parameters:

- `datasets`: List of dataset names to process.
- `dataset_repo`: Path to the dataset repository.
- `results_directory`: Directory to save results.
- `features_directory`: Directory for feature files.
- `models_directory`: Directory for model files.
- `pretrained_models_directory`: Directory for pre-trained models.
- `approach`: Approach to use (options: `"concatenation"`, `"finetuning"`, `"from_scratch"`).
- `task`: Task to perform (options: `"training"`, `"evaluation"`, `"concatenate_features"`).
- `tiny_lite`: Boolean indicating if using the tiny version of the model.
- `use_catch22`: Boolean indicating if using Catch22 features.
- `use_ensemble`: Boolean indicating if using ensemble methods.

Ensure that the `config.yaml` file is properly configured to suit your needs. This file directs the script on which datasets to use, their locations, and the approach and tasks to perform.

## Usage

To start processing datasets according to the specified configuration, execute the main script:

```bash
python main.py
```
## Approach Descriptions
### Concatenation
**Purpose** This approach combines features from different sources to create a unified feature set.

#### Task: concatenate_features

Combines features from the specified datasets using pre-trained models along with pre-computed Catch22 features. It trains and evaluates models based on the value of use_ensemble, which determines whether to ensemble the five models or compute the mean accuracy across models.

### Fine-Tuning
**Purpose**: This approach fine-tunes pre-trained models by concatenating Catch22 features with the latent space of the model and then resuming training.

#### Tasks:

training: Trains the models on the specified datasets.<br>
evaluation: Evaluates the performance of fine-tuned models on the datasets.

### From Scratch
**Purpose** This approach involves training models from scratch, without utilizing pre-trained weight, starting with initial model parameters.

#### Tasks:

training: Trains models from scratch using the specified datasets.<br>
evaluation: Evaluates the performance of models trained from scratch on the datasets.

## Contributing
Contributions are welcome! Please submit issues, pull requests, or suggestions. Ensure that contributions align with the project's goals and adhere to the established coding standards.
