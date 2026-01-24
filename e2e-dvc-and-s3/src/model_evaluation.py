import os 
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from logger import loggerME as logger
from dvclive import Live
import yaml

def load_params(params_path: str)-> dict:
    """Load model parameters from a pickle file."""
    try:
        with open(params_path, 'rb') as file:
            params = yaml.safe_load(file)
        logger.info(f"Model parameters loaded successfully from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"Parameters file not found at {params_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model parameters: {e}")
        raise

def load_model(model_path: str):
    """Load a trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_data(data_path: str)-> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series)-> dict:
    """Evaluate the model and return performance metrics."""
    try:
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        logger.info("Model evaluation completed successfully")
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: dict, output_path: str)-> None:
    """Save evaluation metrics to a text file."""
    try:
        with open(output_path, 'w') as file:
            for key, value in metrics.items():
                file.write(f"{key}: {value}\n")
        logger.info(f"Metrics saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        model = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_fe.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(model, X_test, y_test)

        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", metrics['accuracy'])
            live.log_metric("precision", metrics['precision']) 
            live.log_metric("recall", metrics['recall'])

            live.log_params(params)

        save_metrics(metrics, output_path='./reports/metrics.json')
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
    