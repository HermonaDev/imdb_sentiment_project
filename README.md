# IMDb Sentiment Analysis Mini-Project

A Python-based sentiment analysis system that classifies movie reviews as positive or negative using TF-IDF vectorization and Logistic Regression.

## Project Overview

This project demonstrates a complete machine learning pipeline for sentiment analysis:
- **Dataset**: 5,000 real IMDb movie reviews (2,500 positive, 2,500 negative) - created by taking random balanced samples from the actual 50K IMDb dataset
- **Vectorization**: TF-IDF with n-gram features
- **Model**: Logistic Regression classifier
- **Output**: Sentiment prediction (positive/negative) with confidence score

## Features

- ✅ Real IMDb dataset integration
- ✅ TF-IDF text vectorization
- ✅ Logistic Regression classification
- ✅ Command-line prediction interface
- ✅ Model persistence with joblib
- ✅ Confidence scoring
- ✅ Easy-to-use API
- ✅ **Bonus: Flask REST API endpoint**

## Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model

First, train the sentiment analysis model:

```bash
python train.py
```

This will:
- Load the real IMDb dataset subset
- Vectorize text using TF-IDF
- Train a Logistic Regression model
- Save the model and vectorizer to disk
- Display training metrics

### 2. Make Predictions

#### Command Line Interface

Use the trained model to predict sentiment for new reviews:

```bash
python predict.py "I loved this movie!"
```

**Example outputs:**
```
Review: I loved this movie!
Analyzing sentiment...
Sentiment: positive
Confidence: 81.3%
```

```bash
python predict.py "This movie was terrible and boring."
```

```
Review: This movie was terrible and boring.
Analyzing sentiment...
Sentiment: negative
Confidence: 96.3%
```

#### REST API (Optional Bonus)

Start the Flask API server:

```bash
python app.py
```

**Available endpoints:**

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict sentiment (JSON)
- `GET /predict/<review>` - Predict sentiment (URL)

**Example API usage:**

```powershell
# POST request with JSON (PowerShell)
Invoke-WebRequest -Uri "http://localhost:5000/predict" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"review": "This movie was absolutely fantastic!"}'

# GET request with URL parameter (PowerShell)
Invoke-WebRequest -Uri "http://localhost:5000/predict/This%20movie%20was%20fantastic!"
```

**Alternative: Using curl (if available in Git Bash or WSL):**
```bash
# POST request with JSON
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"review": "This movie was absolutely fantastic!"}'

# GET request with URL parameter
curl "http://localhost:5000/predict/This%20movie%20was%20fantastic!"
```

**API Response:**
```json
{
  "review": "This movie was absolutely fantastic!",
  "sentiment": "positive",
  "confidence": 81.3,
  "confidence_percentage": "81.3%"
}
```

## Project Structure

```
imdb_sentiment_project/
├── train.py                  # Training script
├── predict.py                # Prediction script
├── app.py                    # Flask API server (optional bonus)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── imdb_reviews_subset.csv   # Subset for training (5K samples, balanced)
├── sentiment_classifier.pkl  # Trained model (created after training)
└── tfidf_vectorizer.pkl      # TF-IDF vectorizer (created after training)
```

## Technical Details

### Model Architecture
- **Text Vectorization**: TF-IDF with 8000 max features
- **N-grams**: Unigrams, bigrams, and trigrams (1-3 word combinations)
- **Stop Words**: English stop words removed
- **Classifier**: Logistic Regression with L2 regularization
- **Text Preprocessing**: HTML tag removal, lowercase conversion, special character cleaning

### Performance
- **Accuracy**: ~87% on real IMDb test set
- **Training Time**: < 30 seconds
- **Prediction Time**: < 1 second

## Dependencies

- `scikit-learn==1.3.0` - Machine learning library
- `pandas==2.0.3` - Data manipulation
- `numpy==1.24.3` - Numerical computing
- `joblib==1.3.2` - Model persistence

## Troubleshooting

### Common Issues

1. **"imdb_reviews_subset.csv not found" error**
   - Solution: Ensure the subset file exists in the project directory

2. **"Model files not found" error**
   - Solution: Run `python train.py` first to create the model files

3. **Import errors**
   - Solution: Install dependencies with `pip install -r requirements.txt`

## Future Enhancements

- [ ] Web API with Flask/FastAPI
- [ ] Docker containerization
- [ ] Model performance monitoring

## License

This project is created for demonstration purposes.

---

**Note**: The dataset used (`imdb_reviews_subset.csv`) was created by taking random balanced samples from the actual IMDb dataset (50,000 samples). This ensures a fair and representative training set for sentiment analysis.
