import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from logger import loggerFE as logger
import yaml

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"No data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

def apply_tfidf_vectorization(train_df: pd.DataFrame, test_df: pd.DataFrame, max_features: int) -> tuple:
    """Apply TF-IDF to the data"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        X_train = train_df['text'].values
        y_train = train_df['target'].values
        X_test = test_df['text'].values
        y_test = test_df['target'].values

        X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = vectorizer.transform(X_test).toarray()

        logger.info("TF-IDF vectorization applied successfully.")

        train_df = pd.DataFrame(X_train_tfidf, columns=vectorizer.get_feature_names_out())
        train_df['target'] = y_train

        test_df = pd.DataFrame(X_test_tfidf, columns=vectorizer.get_feature_names_out())
        test_df['target'] = y_test

        logger.info(f"Train DataFrame shape after TF-IDF: {train_df.shape}")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error applying TF-IDF vectorization: {e}")
        raise


