import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

def preprocess_text(text):
    # Text cleanup for reviews (Convert to lowercase, remove HTML tags, remove special characters except apostrophes, remove extra whitespace)
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_imdb_data():
    try:
        data = pd.read_csv('imdb_reviews_subset.csv')
        print("Dataset loaded. Preprocessing text data...")
        data['review_clean'] = data['review'].apply(preprocess_text)
        
        print(f"Dataset size: {data.shape[0]} reviews")
        print(f"Sentiment split:\n{data['sentiment'].value_counts()}")
        return data
        
    except FileNotFoundError:
        print("Error: dataset not found! Please ensure the file exists.")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def train_model():
    print("Loading data...")
    data = load_imdb_data()
    
    print("Analyzing data...")
    print(f"Positive reviews: {len(data[data['sentiment'] == 'positive'])}")
    print(f"Negative reviews: {len(data[data['sentiment'] == 'negative'])}")
    
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        data['review_clean'], data['sentiment'], 
        test_size=0.2, random_state=42, stratify=data['sentiment']
    )
    
    print("Vectorizing text with TF-IDF...")
    # TF-IDF setup - trigrams and a decent vocab size with sublinear scaling
    tfidf= TfidfVectorizer(
        max_features=8000, 
        stop_words='english',
        ngram_range=(1, 3),  
        min_df=1,            
        max_df=0.95,         
        sublinear_tf=True,   
        use_idf=True,
        smooth_idf=True
    )
    
    #Transofrm text to vectors
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    # Train Logistic Regression model 
    model = LogisticRegression(
        random_state=42, 
        max_iter=2000,      
        C=1.0,              
        solver='liblinear', 
        class_weight='balanced'  
    )
    print("Training model...")
    model.fit(X_train_vec, y_train)
    
    # Evaluate Model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    joblib.dump(model, 'sentiment_classifier.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print("Training completed! Model and vectorizer saved.")
    

if __name__ == "__main__":
    train_model()
