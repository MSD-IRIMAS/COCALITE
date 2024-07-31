import os
import yaml
import pandas as pd
import logging
from pathlib import Path
from Approaches.FineTuning import FineTuning
from Approaches.Concatenation import Concatenation
from Approaches.FromScratch.FromScratch import FromScratch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """
    Load the configuration file.
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

def validate_config(config):
    """
    Validate the configuration parameters.
    """
    required_keys = ['datasets', 'dataset_repo', 'results_directory', 'features_directory',
                     'models_directory', 'pretrained_models_directory', 'approach', 'task']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    if config['approach'] not in ['concatenation', 'finetuning', 'from_scratch']:
        raise ValueError(f"Invalid approach: {config['approach']}")
    
    if config['task'] not in ['training', 'evaluation', 'concatenate_features']:
        raise ValueError(f"Invalid task: {config['task']}")
    
    if not isinstance(config['use_catch22'], bool) or not isinstance(config['use_ensemble'], bool):
        raise ValueError("Parameters 'use_catch22' and 'use_ensemble' must be boolean.")

def append_results_to_csv(filepath, headers, results):
    """
    Append results to a CSV file with the appropriate headers.
    """
    df = pd.DataFrame(results, columns=headers)
    mode = 'a' if os.path.exists(filepath) else 'w'
    header = not os.path.exists(filepath)
    df.to_csv(filepath, index=False, mode=mode, header=header)

def handle_concatenation(concat, dataset_name, task, results_path, headers, use_ensemble):
    """
    Handle the concatenation approach.
    """
    if task == "concatenate_features":
        results = concat.concatenate_features(dataset_name, use_ensemble)
        if results:
            append_results_to_csv(results_path, headers, [results])

def handle_finetuning(finetuning, dataset_name, task, results_path, headers, use_ensemble, use_catch22):
    """
    Handle the fine-tuning approach.
    """
    if task == "training":
        finetuning.training(dataset_name)
    elif task == "evaluation":
        results = finetuning.evaluate_models(dataset_name, use_ensemble, use_catch22)
        if results:
            append_results_to_csv(results_path, headers, [results])

def handle_from_scratch(from_scratch, dataset_name, task, results_path, headers, use_ensemble, use_catch22):
    """
    Handle the from-scratch approach.
    """
    if task == "training":
        from_scratch.training(dataset_name)
    elif task == "evaluation":
        results = from_scratch.evaluate_models(dataset_name, use_ensemble, use_catch22)
        if results:
            append_results_to_csv(results_path, headers, [results])

def main():
    """
    Main function to load configuration, process datasets, and handle results.
    """
    # Load and validate configuration
    config = load_config('config.yaml')
    validate_config(config)
    
    # Extract parameters from the config
    datasets = config['datasets']
    dataset_repo = Path(config['dataset_repo'])
    results_directory = Path(config['results_directory'])
    features_directory = Path(config['features_directory'])
    models_directory = Path(config['models_directory'])
    pretrained_models_directory = Path(config['pretrained_models_directory'])
    approach = config['approach']
    task = config['task']
    tiny_lite = config['tiny_lite']
    use_catch22 = config['use_catch22']
    use_ensemble = config['use_ensemble']
    
    # Create results directory if it doesn't exist
    results_directory.mkdir(parents=True, exist_ok=True)
    results_path = results_directory / 'evaluation_results.csv'
    
    # Define headers based on the configuration
    headers = ["Dataset", "Accuracy", "F1", "Precision", "Recall"] if use_ensemble else ["Dataset", "Accuracy"]

    # Process each dataset according to the approach and task
    for dataset_name in datasets:
        logging.info(f"Processing dataset: {dataset_name}")

        if approach == "concatenation":
            concat = Concatenation(pretrained_models_directory, models_directory, features_directory, dataset_repo)
            handle_concatenation(concat, dataset_name, task, results_path, headers, use_ensemble)
        
        elif approach == "finetuning":
            finetuning = FineTuning(pretrained_models_directory, models_directory, features_directory, dataset_repo)
            handle_finetuning(finetuning, dataset_name, task, results_path, headers, use_ensemble, use_catch22)
        
        elif approach == "from_scratch":
            from_scratch = FromScratch(models_directory, features_directory, dataset_repo, tiny_lite)
            handle_from_scratch(from_scratch, dataset_name, task, results_path, headers, use_ensemble, use_catch22)

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
