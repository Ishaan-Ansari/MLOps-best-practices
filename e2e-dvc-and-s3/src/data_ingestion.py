import os
import yaml
import pandas as pd
from logger import loggerDI as logger
from sklearn.model_selection import train_test_split

def load_params(config_path: str) -> dict:
    """Load parameters from a YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded successfully from {config_path}")
        return params
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise 
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise 
    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {e}")
        raise
        
def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file"""
    try:
        data = pd.read_csv(data_path)
        logger.debug(f"Data loaded successfully from {data_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise 
    except pd.errors.EmptyDataError:
        logger.error(f"No data: {data_path} is empty")
        raise 
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data"""
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test data to CSV files."""
    try:
        raw_data_pth = os.path.join(data_path, 'raw_data')
        os.makedirs(raw_data_pth, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_pth, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_pth, 'test.csv'), index=False)
        logger.debug(f"Train and test data saved successfully in {raw_data_pth}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        params = load_params(config_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_pth = "https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv"
        df = load_data(data_path=data_pth)
        final_df = preprocess_data(df)
        train_df, test_df = train_test_split(final_df, test_size=test_size, random_state=42)

        save_data(train_data=train_df, test_data=test_df, data_path='./data')
        logger.info("Data ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return
    
if __name__ == "__main__":
    main()

