import os
import csv
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from aeon.transformations.collection.feature_based import Catch22

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesPreprocessor:
    """
    A class to preprocess time series data, including loading, normalization, and label encoding.
    """

    def __init__(self, dataset_repo: str):
        """
        Initialize the TimeSeriesPreprocessor with the path to the dataset repository.

        Parameters:
        - dataset_repo (str): Path to the repository containing dataset files.
        """
        self.dataset_repo = dataset_repo
        self.encoder = LabelEncoder()

    def load_data(self, dataset_name: str) -> tuple:
        """
        Load time series data from TSV files.

        Parameters:
        - dataset_name (str): Name of the dataset to load.

        Returns:
        - tuple: Contains training and test data (X_train, y_train, X_test, y_test).
        """
        folder_path = os.path.join(self.dataset_repo, dataset_name)
        train_path = os.path.join(folder_path, f"{dataset_name}_TRAIN.tsv")
        test_path = os.path.join(folder_path, f"{dataset_name}_TEST.tsv")

        if not os.path.exists(test_path):
            logger.error("File not found: %s", test_path)
            return None, None, None, None

        train = np.loadtxt(train_path, dtype=np.float64)
        test = np.loadtxt(test_path, dtype=np.float64)
        y_train, y_test = train[:, 0], test[:, 0]
        X_train, X_test = np.delete(train, 0, axis=1), np.delete(test, 0, axis=1)

        return X_train, y_train, X_test, y_test

    def z_normalisation(self, x: np.ndarray) -> np.ndarray:
        """
        Z-normalize the input data.

        Parameters:
        - x (np.ndarray): Input data to be normalized.

        Returns:
        - np.ndarray: Z-normalized data.
        """
        stds = np.std(x, axis=1, keepdims=True)
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds

    def encode_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Encode labels using LabelEncoder.

        Parameters:
        - y (np.ndarray): Array of labels to encode.

        Returns:
        - np.ndarray: Encoded labels.
        """
        return self.encoder.fit_transform(y)

    def preprocess(self, dataset_name: str) -> tuple:
        """
        Preprocess the dataset by loading, normalizing, and encoding.

        Parameters:
        - dataset_name (str): Name of the dataset to preprocess.

        Returns:
        - tuple: Contains processed training and test data (X_train, y_train, X_test, y_test).
        """
        try:
            X_train, y_train, X_test, y_test = self.load_data(dataset_name)
            X_train, X_test = self.z_normalisation(X_train), self.z_normalisation(X_test)
            X_train, X_test = np.expand_dims(X_train, axis=2), np.expand_dims(X_test, axis=2)
            return X_train, self.encode_labels(y_train), X_test, self.encode_labels(y_test)
        except Exception as e:
            logger.error(f"An error occurred during preprocessing: {e}")
            return None, None, None, None

    def calculate_catch22(self, dataset_name: str, base_output_dir: str) -> None:
        """
        Compute and save Catch22 features for a given dataset.

        Parameters:
        - dataset_name (str): Name of the dataset.
        - base_output_dir (str): Directory where the Catch22 features will be saved.
        """
        X_train, _, X_test, _ = self.preprocess(dataset_name)
        if X_train is None or X_test is None:
            return

        tnf = Catch22(replace_nans=True)
        scaler = StandardScaler()

        catch22_train = tnf.fit_transform(X_train)
        catch22_train_scaled = scaler.fit_transform(catch22_train)
        catch22_test = tnf.transform(X_test)
        catch22_test_scaled = scaler.transform(catch22_test)

        output_dir = os.path.join(base_output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        catch22_feature_names = [
            "DN_HistogramMode_5", "DN_HistogramMode_10", "SB_BinaryStats_mean_longstretch1",
            "DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd", "CO_f1ecac",
            "CO_FirstMin_ac", "CO_HistogramAMI_even_2_5", "IN_AutoMutualInfoStats_40_gaussian_fmmi",
            "MD_hrv_classic_pnn40", "SB_BinaryStats_diff_longstretch0", "SB_MotifThree_quantile_hh",
            "FC_LocalSimple_mean1_tauresrat", "CO_Embed2_Dist_tau_d_expfit_meandiff", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SP_Summaries_welch_rect_area_5_1",
            "SP_Summaries_welch_rect_centroid", "FC_LocalSimple_mean3_stderr", "CO_trev_1_num",
            "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff"
        ]

        df_train = pd.DataFrame(catch22_train_scaled, columns=catch22_feature_names)
        df_test = pd.DataFrame(catch22_test_scaled, columns=catch22_feature_names)
        df_train.to_csv(os.path.join(output_dir, 'features_train.csv'), index=False)
        df_test.to_csv(os.path.join(output_dir, 'features_test.csv'), index=False)

        print(f"Catch22 features saved for {dataset_name}:")
        print(f" - Training: {os.path.join(output_dir, 'features_train.csv')}")
        print(f" - Test: {os.path.join(output_dir, 'features_test.csv')}")


def concatenate_features(feature_sets: list) -> np.ndarray:
    """
    Concatenate multiple feature sets along the feature axis.

    Parameters:
    - feature_sets (list): List of feature arrays to concatenate.

    Returns:
    - np.ndarray: Concatenated feature array.
    """
    return np.concatenate(feature_sets, axis=1)

def write_results(results: list, headers: list, filepath: str) -> None:
    """
    Write evaluation results to a CSV file.

    Parameters:
    - results (list): Evaluation results to write.
    - headers (list): List of column headers.
    - filepath (str): Path to the CSV file where results will be saved.
    """
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)

def load_features(dataset_name: str, features_directory: str) -> tuple:
    """
    Load feature data from CSV files.

    Parameters:
    - dataset_name (str): Name of the dataset.
    - features_directory (str): Directory containing the feature files.

    Returns:
    - tuple: Contains training and test feature arrays.
    """
    train_features_path = os.path.join(features_directory, dataset_name, 'features_train.csv')
    test_features_path = os.path.join(features_directory, dataset_name, 'features_test.csv')
    return pd.read_csv(train_features_path).values, pd.read_csv(test_features_path).values

def load_and_prepare_data(dataset_name: str, features_directory: str, dataset_repo: str) -> tuple:
    """
    Load and preprocess data for a given dataset.

    Parameters:
    - dataset_name (str): Name of the dataset.
    - features_directory (str): Directory containing feature files.
    - dataset_repo (str): Path to the repository containing dataset files.

    Returns:
    - tuple: Contains preprocessed data and features (X_train, y_train, X_test, y_test, train_features, test_features).
    """
    preprocessor = TimeSeriesPreprocessor(dataset_repo)
    try:
        X_train, y_train, X_test, y_test = preprocessor.preprocess(dataset_name)
        train_features, test_features = load_features(dataset_name, features_directory)
        return X_train, y_train, X_test, y_test, train_features, test_features
    except Exception as e:
        logger.error(f"Error loading or preparing data for {dataset_name}: {e}")
        return None

def load_models(models_directory: str, dataset_name: str) -> list:
    """
    Load models from the specified directory.

    Parameters:
    - models_directory (str): Directory containing the model files.
    - dataset_name (str): Name of the dataset used in model filenames.

    Returns:
    - list: List of loaded Keras models.
    """
    model_paths = [os.path.join(models_directory, dataset_name, f"{dataset_name}_best_model_{i}.hdf5") for i in range(5)]
    models = []
    for path in model_paths:
        try:
            models.append(tf.keras.models.load_model(path))
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
    return models

def load_latent_models(models_directory: str, dataset_name: str) -> list:
    """
    Load models and extract their latent layers.

    Parameters:
    - models_directory (str): Directory containing the model files.
    - dataset_name (str): Name of the dataset used in model filenames.

    Returns:
    - list: List of Keras models with latent layers as outputs.
    """
    models = load_models(models_directory, dataset_name)
    return [tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output) for model in models]

def evaluate_models(dataset_name: str, models_directory: str, features_directory: str, use_ensemble: str, dataset_repo: str, use_catch22: bool) -> list:
    """
    Evaluate models using the test dataset.

    Parameters:
    - dataset_name (str): Name of the dataset.
    - models_directory (str): Directory containing the model files.
    - features_directory (str): Directory containing feature files.
    - use_ensemble (str): Method to use for evaluation ('ensemble' or 'mean').
    - dataset_repo (str): Path to the repository containing dataset files.
    - use_catch22 (bool): Flag indicating whether to use Catch22 features.

    Returns:
    - list: Evaluation results including accuracy, F1 score, precision, and recall.
    """

    def prepare_test_data() -> tuple:
        """
        Prepare test data for evaluation.

        Returns:
        - tuple: Contains test data (X_test, y_test, test_features).
        """
        data = load_and_prepare_data(dataset_name, features_directory, dataset_repo)
        if data is None:
            return None, None, None
        _, _, X_test, y_test, _, test_features = data
        return X_test, y_test, test_features if use_catch22 else None

    def predict_with_models(models: list, X_test: np.ndarray, test_features: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using multiple models.

        Parameters:
        - models (list): List of Keras models.
        - X_test (np.ndarray): Test data.
        - test_features (np.ndarray): Test features if using Catch22.

        Returns:
        - np.ndarray: Combined predicted probabilities.
        """
        probabilities = [model.predict([X_test, test_features]) if use_catch22 else model.predict(X_test) for model in models]
        return combine_probabilities(probabilities)

    X_test, y_test, test_features = prepare_test_data()
    if X_test is None:
        logger.error(f"Failed to load or prepare data for dataset: {dataset_name}")
        return None
    
    models = load_models(models_directory, dataset_name)
    num_classes = len(np.unique(y_test))
    y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

    if use_ensemble:
        averaged_probabilities = predict_with_models(models, X_test, test_features)
        return evaluate_and_record_results(y_test, averaged_probabilities, dataset_name)
        
    else:
        if use_catch22:
            mean_accuracy = np.mean([model.evaluate([X_test, test_features], y_test_one_hot, verbose=0)[1] for model in models])
        else:
            mean_accuracy = np.mean([model.evaluate(X_test, y_test_one_hot, verbose=0)[1] for model in models])
        
        return [dataset_name, mean_accuracy]


def combine_probabilities(probabilities: list) -> np.ndarray:
    """
    Combine probabilities from multiple models by averaging.

    Parameters:
    - probabilities (list): List of probability arrays from different models.

    Returns:
    - np.ndarray: Averaged probability array.
    """
    mean_probabilities = np.mean(probabilities, axis=0)
    return np.argmax(mean_probabilities, axis=1)

def evaluate_and_record_results(y_true: np.ndarray, predictions: np.ndarray, dataset_name: str) -> list:
    """
    Evaluate the predictions and record the results.

    Parameters:
    - y_true (np.ndarray): True labels.
    - predictions (np.ndarray): Predicted labels.
    - dataset_name (str): Name of the dataset.

    Returns:
    - list: Evaluation metrics including accuracy, F1 score, precision, and recall.
    """
    return [
        dataset_name,
        accuracy_score(y_true, predictions),
        f1_score(y_true, predictions, average='weighted'),
        precision_score(y_true, predictions, average='weighted'),
        recall_score(y_true, predictions, average='weighted')
    ]
