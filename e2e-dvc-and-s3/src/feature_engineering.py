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

        train_df['text'] = train_df['text'].fillna('')  
        test_df['text'] = test_df['text'].fillna('')
        
        X_train = train_df['text'].values
        y_train = train_df['target'].values
        X_test = test_df['text'].values
        y_test = test_df['target'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        logger.info("TF-IDF vectorization applied successfully.")

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['target'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['target'] = y_test

        logger.info(f"Train DataFrame shape after TF-IDF: {train_df.shape}")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error applying TF-IDF vectorization: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str)->None:
    """Save DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def main():
    try:
        params_path = load_params('params.yaml')
        fe_params = params_path['feature_engineering']['max_features']

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf_vectorization(train_data, test_data, fe_params)

        save_data(train_df, os.path.join('./data/processed/', 'train_fe.csv'))
        save_data(test_df, os.path.join('./data/processed/', 'test_fe.csv'))
    except Exception as e:
        logger.error(f"Feature engineering process failed: {e}")
        raise

if __name__ == "__main__":
    main()
