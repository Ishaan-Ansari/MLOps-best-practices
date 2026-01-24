import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from logger import loggerMT as logger
import yaml

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except FileNotFoundError:
        logger.error(f"Parameters file not found: {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise   
    except pd.errors.EmptyDataError as e:
        logger.error(f"Error reading CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, model_params: dict) -> RandomForestClassifier:
    """Train a RandomForestClassifier model."""
    try:
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("Training data is empty.")
        elif X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Mismatch in number of samples between X_train and y_train.")
        
        model = RandomForestClassifier(n_estimators=model_params.get('n_estimators', 100), random_state=model_params.get('random_state', 42))
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        return model
    except ValueError as e:
        logger.error(f"Error during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str)-> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    try:
        # Load parameters from YAML file
        params = load_params('params.yaml')['model_building']
        train_data = load_data('./data/processed/train_fe.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train the model
        model = train_model(X_train, y_train, params)

        model_save_path = 'models/model.pkl'

        # Save the trained model
        save_model(model, model_save_path)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
