import os
import numpy as np
import tensorflow as tf
import logging
from typing import Optional, Union

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical

from utils.utils import evaluate_models, load_latent_models, load_and_prepare_data

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FineTuning:
    def __init__(self, pretrained_models_directory: str, models_directory: str, features_directory: str, dataset_repo: str):
        """
        Initialize FineTuning with directories for models, features, and datasets.

        Parameters:
        - pretrained_models_directory (str): Path to the directory with pretrained models.
        - models_directory (str): Path to the directory to save fine-tuned models.
        - features_directory (str): Path to the directory with additional features.
        - dataset_repo (str): Path to the repository with dataset information.
        """
        self.pretrained_models_directory = pretrained_models_directory
        self.models_directory = models_directory
        self.features_directory = features_directory
        self.dataset_repo = dataset_repo

    def training(self, dataset_name: str):
        """
        Fine-tune models using the specified dataset.

        Parameters:
        - dataset_name (str): The name of the dataset.

        """
        try:
            data = load_and_prepare_data(dataset_name, self.features_directory, self.dataset_repo)

            X_train, y_train, _, _, train_features, _ = data
            y_train_categorical = to_categorical(y_train)
            input_dim = train_features.shape[1]
            n_classes = len(np.unique(y_train))

            models = load_latent_models(self.pretrained_models_directory, dataset_name)

            for i, model in enumerate(models):
                try:
                    file_name = f'{dataset_name}_best_model_{i}.hdf5'
                    file_path = os.path.join(self.models_directory, dataset_name, file_name)

                    callbacks = [
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor="loss", factor=0.5, patience=50, min_lr=0.0001
                        ),
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=file_path,
                            monitor="loss",
                            save_best_only=True
                        )
                    ]

                    custom_features = Input(shape=(input_dim,), name='custom_features')
                    concatenated_output = Concatenate(name='concatenate_LS_custom_features')([model.output, custom_features])
                    concatenated_output = BatchNormalization(name='batch_normalization_LS_custom_features')(concatenated_output)
                    classifier_output = Dense(n_classes, activation='softmax')(concatenated_output)

                    final_model = Model(inputs=[model.input, custom_features], outputs=classifier_output)
                    final_model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
                    final_model.fit([X_train, train_features], y_train_categorical, epochs=750, batch_size=64, callbacks=callbacks)

                except Exception as e:
                    logger.error(f"An error occurred during training for model {i}: {e}")

        except Exception as e:
            logger.error(f"An error occurred during data preparation for {dataset_name}: {e}")

    def evaluate_models(self, dataset_name: str, use_ensemble: str, use_catch22: bool = True):
        """
        Evaluate models using specified methods and features.

        Parameters:
        - dataset_name (str): The name of the dataset.
        - use_ensemble (str): The evaluation method ('ensemble' or 'mean').
        - use_catch22 (bool): Whether to include Catch22 features in the evaluation.

        """
        try:
            results = evaluate_models(dataset_name, self.models_directory, self.features_directory, use_ensemble, self.dataset_repo, use_catch22)
            return results
        except Exception as e:
            logger.error(f"An error occurred during evaluation for {dataset_name}: {e}")
            return None
