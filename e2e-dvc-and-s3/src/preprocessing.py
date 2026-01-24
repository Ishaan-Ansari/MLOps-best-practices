import os
import pandas as pd
from logger import loggerPR as logger
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string 
import nltk

nltk.download('stopwords')
nltk.download('punkt')


def transform_text(text: str) -> str:
    """Transform text by lowering case, tokenizing, removing stopwords and punctuation and stemming."""
    try:
        # Lowercase
        text = text.lower()
        
        # Tokenization
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        
        # Stemming
        stemmer = nltk.PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        
        transformed_text = ' '.join(stemmed_tokens)
        return transformed_text
    except Exception as e:
        logger.error(f"Error transforming text: {e}")
        raise

def preprocess_df(df: pd.DataFrame, text_column: str, target_column='target'):
    """Preprocess the DataFrame by encoding target column, removing duplicates and transforming text."""
    try:
        # Encode target column
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        
        # Remove duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"DataFrame shape after removing duplicates: {df.shape}")

        # Transform text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        
        return df
    except Exception as e:
        logger.error(f"Error preprocessing DataFrame: {e}")
        raise

def main(text_column: str = 'text', target_column: str = 'target'):
    """Main function to load, preprocess and save the DataFrame."""
    try:
        train_data = pd.read_csv("./data/raw_data/train.csv")
        test_data = pd.read_csv("./data/raw_data/test.csv")

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Save the processed data
        data_pth = os.path.join("./data", "interim")
        os.makedirs(data_pth, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_pth, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_pth, "test_processed.csv"), index=False)

        logger.info("Preprocessing completed and files saved.")
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")    
    except pd.errors.EmptyDataError as ede:
        logger.error(f"Empty data error: {ede}")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}")
        raise


if __name__ == "__main__":
    main()