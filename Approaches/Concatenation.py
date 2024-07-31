import os
import numpy as np
import logging
from typing import List, Tuple, Optional, Union

from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, BatchNormalization, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from utils.utils import (
    load_and_prepare_data,
    load_latent_models,
    concatenate_features,
    evaluate_and_record_results
)

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Concatenation:
    def __init__(self, pretrained_models_directory: str, models_directory: str, features_directory: str, dataset_repo: str):
        """
        Initialize the Concatenation class with directories for pretrained models, model saving, features, and dataset.

        Parameters:
        - pretrained_models_directory (str): Directory path containing pretrained models.
        - models_directory (str): Directory path to save trained models.
        - features_directory (str): Directory path for additional features.
        - dataset_repo (str): Repository path for datasets.
        """
        self.pretrained_models_directory = pretrained_models_directory
        self.models_directory = models_directory
        self.features_directory = features_directory
        self.dataset_repo = dataset_repo

    def create_classifier(self, num_classes: int, input_dim: int, X_train: np.ndarray, y_train_categorical: np.ndarray, filepath: str) -> Sequential:
        """
        Create and train a classifier with callbacks.

        Parameters:
        - num_classes (int): Number of output classes.
        - input_dim (int): Dimension of the input features.
        - X_train (np.ndarray): Training feature data.
        - y_train_categorical (np.ndarray): Categorical training labels.
        - filepath (str): Path to save the best model weights.

        """
        classifier = Sequential([
            InputLayer(input_shape=(input_dim,)),
            BatchNormalization(),
            Dense(num_classes, activation='softmax')
        ])
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001),
            ModelCheckpoint(filepath=filepath, monitor='loss', save_best_only=True)
        ]

        classifier.fit(X_train, y_train_categorical, epochs=1500, batch_size=64, verbose=1, callbacks=callbacks)
        classifier.load_weights(filepath)
        return classifier

    def concatenate_features(self, dataset_name: str, use_ensemble: bool) -> Union[float, List[Union[str, float]]]:
        """
        Train classifiers and evaluate their performance.

        Parameters:
        - dataset_name (str): Name of the dataset.
        - use_ensemble (bool): Whether to use ensemble methods.

        """

        data = load_and_prepare_data(dataset_name, self.features_directory, self.dataset_repo)
        
        X_train, y_train, X_test, y_test, train_features, test_features = data
        y_train_categorical = to_categorical(y_train)
        n_classes = len(np.unique(y_train))
        accuracies = []
        probabilities = []  # Used if ensembling

        models = load_latent_models(self.pretrained_models_directory, dataset_name)

        for i, model in enumerate(models):
            try:
                latent_space_train = model.predict(X_train)
                latent_space_test = model.predict(X_test)
                concatenated_features_train = concatenate_features([latent_space_train, train_features])
                concatenated_features_test = concatenate_features([latent_space_test, test_features])
                file_name = f'{dataset_name}_best_model_{i}.hdf5'
                file_path = os.path.join(self.models_directory, dataset_name, file_name)
                classifier = self.create_classifier(
                    n_classes,
                    concatenated_features_train.shape[1],
                    concatenated_features_train,
                    y_train_categorical,
                    file_path
                )

                model_probabilities = classifier.predict(concatenated_features_test)
                
                if use_ensemble:
                    probabilities.append(model_probabilities)
                else:
                    y_pred = np.argmax(model_probabilities, axis=1)
                    accuracy = accuracy_score(y_test, y_pred) 
                    accuracies.append(accuracy)

            except Exception as e:
                logger.error(f"Error during model prediction for {dataset_name}, model {i}: {e}")
                continue

        if use_ensemble:
            final_predictions = np.mean(probabilities, axis=0)
            y_final_pred = np.argmax(final_predictions, axis=1)
            return evaluate_and_record_results(y_test, y_final_pred, dataset_name)

        else:
            mean_accuracy = np.mean(accuracies)
            logger.info(f"Mean accuracy of classifiers: {mean_accuracy:.4f}")
            return [dataset_name, mean_accuracy]
