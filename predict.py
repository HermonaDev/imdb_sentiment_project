import sys
import joblib
import numpy as np
import re

def clean_text(text):
    """Clean and preprocess text data (same as in train.py)"""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters but keep apostrophes
    text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found. Please run 'python train.py' first.")
        sys.exit(1)

def predict_sentiment(review_text, model, vectorizer):
    """Predict sentiment for a given review text"""
    # Clean the input text
    clean_review = clean_text(review_text)
    
    # Vectorize the input text
    text_vectorized = vectorizer.transform([clean_review])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    
    # Get confidence scores
    confidence_scores = model.predict_proba(text_vectorized)[0]
    
    # Get confidence for the predicted class
    if prediction == 'positive':
        confidence = confidence_scores[1]  # positive class probability
    else:
        confidence = confidence_scores[0]  # negative class probability
    
    return prediction, confidence

def main():
    """Main function to handle command line prediction"""
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your review text here\"")
        print("Example: python predict.py \"I loved this movie!\"")
        sys.exit(1)
    
    # Get review text from command line arguments
    review_text = sys.argv[1]
    
    print(f"Review: {review_text}")
    print("Analyzing sentiment...")
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    # Make prediction
    sentiment, confidence = predict_sentiment(review_text, model, vectorizer)
    
    # Output results
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
