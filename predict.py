import sys
import joblib
import numpy as np
import re

def preprocess_text(text):
    # Text cleanup for reviews (Convert to lowercase, remove HTML tags, remove special characters except apostrophes, remove extra whitespace)
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model_and_vectorizer():
    try:
        model = joblib.load('sentiment_classifier.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found. Please run 'python train.py' first.")
        sys.exit(1)

def predict_sentiment(review_text, model, vectorizer):
    clean_review = preprocess_text(review_text)
    text_vectorized = vectorizer.transform([clean_review])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    confidence = model.predict_proba(text_vectorized)[0]
    confidence_score = confidence[1] if prediction == 'positive' else confidence[0]
    return prediction, confidence_score

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your review text here\"")
        print("Example: python predict.py \"I loved this movie!\"")
        sys.exit(1)
    
    # Get review text from command line arguments
    review_text = sys.argv[1]    
    print(f"Review: {review_text}")
    print("Analyzing sentiment...")
    model, vectorizer = load_model_and_vectorizer()
    sentiment, confidence = predict_sentiment(review_text, model, vectorizer)
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence*100:.1f}%")

if __name__ == "__main__":
    main()
