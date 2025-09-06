# train_model.py
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# File paths (consistent with app.py)
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DATA_PATH = "data/fake_or_real_news.csv"

def train_and_save_model(data_path=DATA_PATH, sample=5000):
    """
    Train a TF-IDF + Logistic Regression model on fake news dataset
    and save the model + vectorizer to disk.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ Dataset not found at {data_path}. Please make sure it exists."
        )

    print("🔄 Loading dataset...")
    df = pd.read_csv(data_path)

    # Optional sampling for faster training
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)

    print(f"✅ Dataset loaded: {len(df)} rows. Starting training...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    preds = model.predict(X_test_tfidf)
    print("\n📊 Model Performance:")
    print(classification_report(y_test, preds))

    # Save artifacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"✅ Vectorizer saved to {VECTORIZER_PATH}")

if __name__ == "__main__":
    # Train on a subset of 5000 rows by default
    # Set sample=None for full dataset
    train_and_save_model(sample=5000)
