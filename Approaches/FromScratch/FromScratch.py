import os
import logging
import numpy as np
import tensorflow as tf

from utils.utils import load_and_prepare_data, evaluate_models
from Approaches.FromScratch.lite_catch22 import LITE_CF
from Approaches.FromScratch.lite import LITE

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def print_conv_layer_weights(model: tf.keras.Model, layer_index: int):
    """
    Print and return the weights of a specified Conv1D layer in a Keras model.

    Parameters:
    - model (tf.keras.Model): The Keras model containing the layer.
    - layer_index (int): Index of the layer to inspect.

    """
    if layer_index < len(model.layers):
        layer = model.layers[layer_index]
        if isinstance(layer, tf.keras.layers.Conv1D):
            weights = layer.get_weights()
            logging.info(f"Weights of Conv1D layer {layer_index}: {weights}")
            return weights
        else:
            logging.error(f"Layer {layer_index} is not a Conv1D layer.")
            return None
    else:
        logging.error(f"Layer index {layer_index} is out of range for the model.")
        return None

def compare_weights(model1: tf.keras.Model, model2: tf.keras.Model, layer_index: int):
    """
    Compare weights of the specified Conv1D layer between two Keras models.

    Parameters:
    - model1 (tf.keras.Model): The first Keras model.
    - model2 (tf.keras.Model): The second Keras model.
    - layer_index (int): The index of the layer to compare.
    """
    weights1 = print_conv_layer_weights(model1, layer_index)
    weights2 = print_conv_layer_weights(model2, layer_index)
    if weights1 is not None and weights2 is not None:
        if np.allclose(weights1, weights2):
            logging.info(f"Weights of layer {layer_index} are identical.")
        else:
            logging.info(f"Weights of layer {layer_index} differ between models.")
    else:
        logging.error(f"Could not compare weights for layer {layer_index}.")

class FromScratch:
    def __init__(self, models_directory: str, features_directory: str, dataset_repo: str, tiny_lite: bool):
        """
        Initialize the FromScratch class with necessary directories and settings.

        Parameters:
        - models_directory (str): Path to the directory for saving models.
        - features_directory (str): Path to the directory with additional features.
        - dataset_repo (str): Path to the repository with dataset information.
        - tiny_lite (bool): Flag to determine the use of a tiny LITE model (16 filters).
        """
        self.models_directory = models_directory
        self.features_directory = features_directory
        self.dataset_repo = dataset_repo
        self.tiny_lite = tiny_lite

    def training(self, dataset_name: str):
        """
        Train models using the specified dataset.

        Parameters:
        - dataset_name (str): Name of the dataset to be used for training.
        """
        try:
            with tf.device('/gpu:0'):
                # Prepare directories for saving models
                dataset_dir_lite = os.path.join(self.models_directory, 'LITE_models', dataset_name)
                dataset_dir_lite_cf = os.path.join(self.models_directory, 'LITE_Catch22_models', dataset_name)
                os.makedirs(dataset_dir_lite, exist_ok=True)
                os.makedirs(dataset_dir_lite_cf, exist_ok=True)

                # Load and prepare data
                data = load_and_prepare_data(dataset_name, self.features_directory, self.dataset_repo)
                
                X_train, y_train, _, _, train_features, _ = data
                dim = train_features.shape[1]
                logging.info("Dimension of custom features: %d", dim)
                
                seeds = [42, 123, 987, 555, 789]
                for i, seed in enumerate(seeds):
                    # Train LITE model
                    model = LITE(
                        output_directory=os.path.join(dataset_dir_lite, dataset_name),
                        run_nbr=i,
                        length_TS=int(X_train.shape[1]),
                        n_classes=len(np.unique(y_train)),
                        seed=seed,
                        tiny_lite=self.tiny_lite
                    )
                    model.fit(xtrain=X_train, ytrain=y_train)

                    # Train LITE_CF model with Catch22 features
                    model_cf = LITE_CF(
                        output_directory=os.path.join(dataset_dir_lite_cf, dataset_name),
                        run_nbr=i,
                        length_TS=int(X_train.shape[1]),
                        n_classes=len(np.unique(y_train)),
                        dim=dim,
                        seed=seed,
                        tiny_lite=self.tiny_lite
                    )
                    model_cf.fit(xtrain=X_train, ytrain=y_train, custom_features_train=train_features)

        except Exception as e:
            logging.error(f"Error during training on dataset {dataset_name}: {e}")

    def evaluate_models(self, dataset_name: str, use_ensemble: str, use_catch22: bool):
        """
        Evaluate models using the specified evaluation type and Catch22 features.

        Parameters:
        - dataset_name (str): Name of the dataset to be evaluated.
        - use_ensemble (str): Evaluation method ('ensemble' or 'mean').
        - use_catch22 (bool): Flag indicating whether to use Catch22 features.

        """
        try:
            return evaluate_models(dataset_name, self.models_directory, self.features_directory, use_ensemble, self.dataset_repo, use_catch22)
        except Exception as e:
            logging.error(f"An error occurred during evaluation for {dataset_name}: {e}")
            return None
