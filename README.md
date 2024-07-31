# Project Overview

The objective of this project is to enhance performance and maintain computational efficiency by integrating engineered feature vectors with features learned from deep learning models. The project explores various methodologies, and evaluates the impact of these strategies on time series classification tasks.

## Requirements

Please ensure the following Python packages are installed:

- PyYAML==5.4.1
- pandas==2.0.3
- tensorflow==2.12.0
- scikit-learn==1.4.0
- aeon==0.10.0
- numpy==1.23.5
- matplotlib==3.9.1


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
