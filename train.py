import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os
import re

def clean_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters but keep apostrophes
    text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_imdb_data():
    """
    Load IMDb dataset subset (5,000 samples).
    This uses the real IMDb dataset subset.
    """
    try:
        # Load the subset dataset
        data = pd.read_csv('imdb_reviews_subset.csv')
        
        # Clean the text data
        print("Cleaning text data...")
        data['review_clean'] = data['review'].apply(clean_text)
        
        # Verify the data structure
        print(f"Loaded IMDb subset: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Sentiment distribution:")
        print(data['sentiment'].value_counts())
        
        return data
        
    except FileNotFoundError:
        print("Error: IMDB_Dataset_Subset.csv not found!")
        print("Please ensure the subset file exists.")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def train_model():
    """Train the sentiment analysis model with improved parameters"""
    print("Loading data...")
    data = load_imdb_data()
    
    print(f"Dataset shape: {data.shape}")
    print(f"Positive reviews: {len(data[data['sentiment'] == 'positive'])}")
    print(f"Negative reviews: {len(data[data['sentiment'] == 'negative'])}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data['review_clean'], data['sentiment'], 
        test_size=0.2, random_state=42, stratify=data['sentiment']
    )
    
    print("Vectorizing text with enhanced TF-IDF...")
    # Create enhanced TF-IDF vectorizer with better parameters
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased features
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams
        min_df=1,            # Include more rare words
        max_df=0.95,         # Remove very common words
        sublinear_tf=True,   # Apply sublinear scaling
        use_idf=True,
        smooth_idf=True
    )
    
    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    print("Training enhanced Logistic Regression model...")
    # Train Logistic Regression model with better parameters
    model = LogisticRegression(
        random_state=42, 
        max_iter=2000,      # More iterations
        C=1.0,              # Regularization strength
        solver='liblinear', # Better for binary classification
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Fit the model
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
    print("Training completed! Model and vectorizer saved.")
    print("Files created:")
    print("- sentiment_model.pkl")
    print("- tfidf_vectorizer.pkl")

if __name__ == "__main__":
    train_model()
