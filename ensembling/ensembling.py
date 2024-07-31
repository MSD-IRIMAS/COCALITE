import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from itertools import combinations
from typing import Optional, List, Tuple

from utils.utils import (
    load_and_prepare_data,
    combine_probabilities,
    load_features,
    load_models
)

def predict_with_model(model: tf.keras.Model, X_test: np.ndarray,
                       test_features: Optional[np.ndarray] = None,
                       use_features: bool = False):
    """
    Predict the output for the test set using a single model.

    Parameters:
    - model (tf.keras.Model): Trained model.
    - X_test (np.ndarray): Test data features.
    - test_features (Optional[np.ndarray]): Additional test features (optional).
    - use_features (bool): Whether to use additional test features.

    Returns:
    - np.ndarray: Predictions from the model.
    """
    if use_features:
        return model.predict([X_test, test_features])
    return model.predict(X_test)


def calculate_combined_accuracy(probabilities_list: List[np.ndarray], y_true: np.ndarray):
    """
    Calculate the accuracy by combining predictions from multiple models.

    Parameters:
    - probabilities_list (List[np.ndarray]): List of probabilities from models.
    - y_true (np.ndarray): True labels.

    Returns:
    - float: Accuracy of the combined predictions.
    """
    combined_pred = combine_probabilities(probabilities_list)
    return accuracy_score(y_true, combined_pred)

def calculate_mean_accuracy_of_combinations(probabilities_list_1: List[np.ndarray], 
                                            probabilities_list_2: List[np.ndarray], 
                                            combination_size: int, 
                                            y_true: np.ndarray):
    """
    Calculate the mean accuracy for a given combination size of models.

    Parameters:
    - probabilities_list_1 (List[np.ndarray]): List of probabilities from models without Catch22.
    - probabilities_list_2 (List[np.ndarray]): List of probabilities from models with Catch22.
    - combination_size (int): The number of models to combine.
    - y_true (np.ndarray): True labels.

    Returns:
    - float: Mean accuracy for the given combination.
    """
    accuracies = []
    for combo in combinations(range(len(probabilities_list_1)), combination_size):
        combined_probs = combine_probabilities(
            [probabilities_list_1[i] for i in combo] + [probabilities_list_2[i] for i in combo]
        )
        accuracies.append(accuracy_score(y_true, combined_probs))
    return np.mean(accuracies)

def evaluate_ensemble(models_directory: str, models_directory_catch22: str,
                      dataset_name: str, features_directory: str, dataset_repo: str):
    """
    Evaluate an ensemble of models using various configurations.

    Parameters:
    - models_directory (str): Directory containing models without Catch22 features.
    - models_directory_catch22 (str): Directory containing models with Catch22 features.
    - dataset_name (str): The name of the dataset.
    - features_directory (str): Directory containing additional features.
    - dataset_repo (str): Directory where data files are located.

    Returns:
    - mean_combined_accuracy_without_catch22 (float): Mean accuracy of all possible pairs of models without Catch22.
    - mean_combined_accuracy_with_catch22 (float): Mean accuracy of all possible pairs of models with Catch22.
    - COCALITE (float): Mean accuracy of paired ensembling (1 without and 1 with Catch22 with same initialization).
    - COCALITE_5 (float): Accuracy for all 10 models combined (5 without and 5 with Catch22).
    - COCALITE_4 (float): Mean accuracy for combinations of 8 models (4 without and 4 with Catch22).
    - COCALITE_3 (float): Mean accuracy for combinations of 6 models (3 without and 3 with Catch22).
    - COCALITE_2 (float): Mean accuracy for combinations of 4 models (2 without and 2 with Catch22).
    - LITETime (float): Accuracy of ensembling 5 models without Catch22.
    - LITETime_Catch22 (float): Accuracy of ensembling 5 models with Catch22.
    - LITE (float): Mean accuracy of individual models without Catch22.
    - LITE_Catch22 (float): Mean accuracy of individual models with Catch22.
    """
    data = load_and_prepare_data(dataset_name, features_directory, dataset_repo)

    X_train, y_train, X_test, y_test, _, test_features = data

    # Load models
    models_without_catch22 = load_models(models_directory, dataset_name)
    models_with_catch22 = load_models(models_directory_catch22, dataset_name)

    # Predict with individual models
    all_probabilities_without_catch22 = [predict_with_model(model, X_test) for model in models_without_catch22]
    all_probabilities_with_catch22 = [predict_with_model(model, X_test, test_features=test_features, use_features=True) for model in models_with_catch22]

    # Calculate individual model accuracies
    LITE = np.mean([accuracy_score(y_test, np.argmax(prob, axis=1)) for prob in all_probabilities_without_catch22])
    LITE_Catch22 = np.mean([accuracy_score(y_test, np.argmax(prob, axis=1)) for prob in all_probabilities_with_catch22])

    # Calculate paired ensembling accuracy
    paired_accuracies = []
    for i in range(len(models_without_catch22)):
        paired_probabilities = combine_probabilities([all_probabilities_without_catch22[i], all_probabilities_with_catch22[i]])
        paired_accuracies.append(accuracy_score(y_test, paired_probabilities))
    COCALITE = np.mean(paired_accuracies)

    # Calculate mean accuracy for all possible pairs of models without Catch22
    combined_accuracies_without_catch22 = [calculate_combined_accuracy([all_probabilities_without_catch22[i], all_probabilities_without_catch22[j]], y_test) for i, j in combinations(range(len(models_without_catch22)), 2)]
    mean_combined_accuracy_without_catch22 = np.mean(combined_accuracies_without_catch22)

    # Calculate mean accuracy for all possible pairs of models with Catch22
    combined_accuracies_with_catch22 = [calculate_combined_accuracy([all_probabilities_with_catch22[i], all_probabilities_with_catch22[j]], y_test) for i, j in combinations(range(len(models_with_catch22)), 2)]
    mean_combined_accuracy_with_catch22 = np.mean(combined_accuracies_with_catch22)

    # Accuracy for all models combined
    combined_pred_all = combine_probabilities(all_probabilities_without_catch22 + all_probabilities_with_catch22)
    COCALITE_5 = accuracy_score(y_test, combined_pred_all)

    # Mean accuracy for combinations of 8 models (4 without and 4 with Catch22)
    COCALITE_4 = calculate_mean_accuracy_of_combinations(all_probabilities_without_catch22, all_probabilities_with_catch22, 4, y_test)

    # Mean accuracy for combinations of 6 models (3 without and 3 with Catch22)
    COCALITE_3 = calculate_mean_accuracy_of_combinations(all_probabilities_without_catch22, all_probabilities_with_catch22, 3, y_test)

    # Mean accuracy for combinations of 4 models (2 without and 2 with Catch22)
    COCALITE_2 = calculate_mean_accuracy_of_combinations(all_probabilities_without_catch22, all_probabilities_with_catch22, 2, y_test)

    # Calculate overall accuracies for specific configurations
    LITETime = accuracy_score(y_test, combine_probabilities(all_probabilities_without_catch22))
    LITETime_Catch22 = accuracy_score(y_test, combine_probabilities(all_probabilities_with_catch22))

    return (mean_combined_accuracy_without_catch22, mean_combined_accuracy_with_catch22, COCALITE,
            COCALITE_5, COCALITE_4, COCALITE_3, COCALITE_2, LITETime, LITETime_Catch22, LITE, LITE_Catch22)
