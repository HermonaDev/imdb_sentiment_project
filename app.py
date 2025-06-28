from flask import Flask, request, jsonify
import joblib
import numpy as np
import re

app = Flask(__name__)

def preprocess_text(text):
    # Clean up review text
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s\']', ' ', text)
    text = ' '.join(text.split())
    return text

def load_model():
    # Load trained model and vectorizer
    try:
        model = joblib.load('sentiment_classifier.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        raise FileNotFoundError("Can't find model files. Run train.py first.")

# Load model at startup
try:
    model, vectorizer = load_model()
except Exception as e:
    print(f"Model load failed: {e}")
    model, vectorizer = None, None

def analyze_sentiment(review_text):
    # Predict sentiment for a review
    clean_review = preprocess_text(review_text)
    review_vec = vectorizer.transform([clean_review])
    sentiment = model.predict(review_vec)[0]
    confidence = model.predict_proba(review_vec)[0]
    confidence_score = confidence[1] if sentiment == 'positive' else confidence[0]
    return sentiment, confidence_score

@app.route('/')
def home():
    # API info endpoint
    return jsonify({
        "message": "IMDb Sentiment Analysis API",
        "endpoints": ["/health (GET)", "/predict (POST)", "/predict/<review> (GET)"],
        "example": "POST /predict with {'review': 'This movie was great!'}"
    })

@app.route('/health')
def health():
    # Check API status
    return jsonify({"status": "up", "model_ready": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    # Predict sentiment from JSON input
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        review_text = data.get('review', '').strip()
        if not review_text:
            return jsonify({"error": "Missing or empty review"}), 400

        sentiment, confidence = analyze_sentiment(review_text)
        return jsonify({
            "review": review_text,
            "sentiment": sentiment.capitalize(),
            "confidence": f"{confidence * 100:.1f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<path:review_text>', methods=['GET'])
def predict_get(review_text):
    # Predict sentiment from URL
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        sentiment, confidence = analyze_sentiment(review_text)
        return jsonify({
            "review": review_text,
            "sentiment": sentiment.capitalize(),
            "confidence": f"{confidence * 100:.1f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting sentiment analysis API...")
    app.run(debug=True, host='0.0.0.0', port=5000)