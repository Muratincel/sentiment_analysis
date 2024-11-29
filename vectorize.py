import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle  # To save the vectorizer for future use

def vectorize_text(df, save_vectorizer=True):
    """
    Vectorizes the text column in the DataFrame using TF-IDF.
    
    Args:
        df (pd.DataFrame): DataFrame with 'text' and 'label' columns.
        save_vectorizer (bool): Whether to save the TfidfVectorizer for later use.
    
    Returns:
        X (sparse matrix): The TF-IDF matrix.
        y (array): The labels as a numpy array.
    """
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    
    # Transform the text data
    X = vectorizer.fit_transform(df['text'])
    y = df['label'].values  # Labels
    
    # Save the vectorizer for reuse (optional)
    if save_vectorizer:
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    
    return X, y

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv('preprocessed_data.csv')
    
    # Vectorize the text
    X, y = vectorize_text(df)
    
    print("TF-IDF vectorization complete.")
    print(f"Shape of TF-IDF matrix: {X.shape}")
    print(f"Number of labels: {len(y)}")
