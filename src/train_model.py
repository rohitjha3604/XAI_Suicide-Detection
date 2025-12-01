"""
Train a text classification model for suicide detection.
Uses TF-IDF vectorization and Logistic Regression.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Column name constants
TEXT_COL = "text"
LABEL_COL = "class"  # Dataset uses 'class' column for labels

def load_and_prepare_data(data_path: str) -> tuple:
    """
    Load the dataset and prepare it for training.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Tuple of (texts, labels)
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    if TEXT_COL not in df.columns:
        raise ValueError(f"Text column '{TEXT_COL}' not found. Available: {list(df.columns)}")
    
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found. Available: {list(df.columns)}")
    
    # Clean missing rows
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    print(f"Shape after dropping NaN: {df.shape}")
    
    # Convert labels to binary (1 = suicide, 0 = non-suicide)
    df["label_binary"] = df[LABEL_COL].apply(
        lambda x: 1 if str(x).lower() in ["suicide", "suicidal", "1"] else 0
    )
    
    print(f"\nLabel distribution:")
    print(df["label_binary"].value_counts())
    
    return df[TEXT_COL].values, df["label_binary"].values

def train_model(X_train, y_train) -> Pipeline:
    """
    Train the TF-IDF + Logistic Regression pipeline.
    
    Args:
        X_train: Training texts
        y_train: Training labels
        
    Returns:
        Trained pipeline
    """
    print("\nTraining model...")
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs"
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    print("Model training complete.")
    
    return pipeline

def evaluate_model(pipeline: Pipeline, X_test, y_test):
    """
    Evaluate the trained model and print metrics.
    """
    print("\nEvaluating model...")
    
    y_pred = pipeline.predict(X_test)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Suicide", "Suicide"]))

def save_model(pipeline: Pipeline, model_path: str):
    """
    Save the trained model to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")

def main():
    # Set up paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "suicide_watch.csv")
    model_path = os.path.join(project_root, "models", "suicide_text_model.pkl")
    
    # Load and prepare data
    texts, labels = load_and_prepare_data(data_path)
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=0.2, 
        random_state=42,
        stratify=labels
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    pipeline = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(pipeline, X_test, y_test)
    
    # Save model
    save_model(pipeline, model_path)
    
    print("\nTraining pipeline complete!")

if __name__ == "__main__":
    main()
