"""
Explainable AI using LIME (Local Interpretable Model-agnostic Explanations).
Provides word-level contributions for model predictions.
"""

import os
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer
from typing import Tuple, List, Dict

# Class names for the explainer
CLASS_NAMES = ["Non-Suicide", "Suicide"]

def load_model(model_path: str = None):
    """
    Load the trained model pipeline.
    
    Args:
        model_path: Path to the model file. If None, uses default path.
        
    Returns:
        Loaded model pipeline
    """
    if model_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "models", "suicide_text_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    return joblib.load(model_path)

def explain_text_with_lime(
    text: str,
    model=None,
    num_features: int = 10,
    num_samples: int = 500
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate LIME explanation for a given text.
    
    Args:
        text: Input text to explain
        model: Trained model pipeline (loads default if None)
        num_features: Number of top features to return
        num_samples: Number of samples for LIME perturbation
        
    Returns:
        Tuple of:
            - probs: Prediction probabilities [non-suicide, suicide]
            - word_contributions: List of dicts with word, weight, and sentiment
    """
    if model is None:
        model = load_model()
    
    # Create LIME explainer
    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        split_expression=r'\W+',
        random_state=42
    )
    
    # Get prediction probabilities
    probs = model.predict_proba([text])[0]
    
    # Generate explanation for the suicide class (label 1)
    explanation = explainer.explain_instance(
        text,
        model.predict_proba,
        num_features=num_features,
        num_samples=num_samples,
        labels=[1]  # Explain the suicide class
    )
    
    # Extract word contributions
    word_contributions = []
    for word, weight in explanation.as_list(label=1):
        word_contributions.append({
            "word": word,
            "weight": weight,
            "contribution": "positive" if weight > 0 else "negative"
        })
    
    return probs, word_contributions

def get_explanation_summary(
    text: str,
    model=None
) -> Dict:
    """
    Get a complete explanation summary for a text.
    
    Args:
        text: Input text to analyze
        model: Trained model pipeline
        
    Returns:
        Dictionary containing prediction and explanation details
    """
    if model is None:
        model = load_model()
    
    probs, word_contributions = explain_text_with_lime(text, model)
    
    predicted_class = CLASS_NAMES[np.argmax(probs)]
    confidence = float(max(probs))
    suicide_probability = float(probs[1])
    
    # Separate positive and negative contributors
    positive_words = [w for w in word_contributions if w["contribution"] == "positive"]
    negative_words = [w for w in word_contributions if w["contribution"] == "negative"]
    
    return {
        "text": text,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "suicide_probability": suicide_probability,
        "probabilities": {
            "non_suicide": float(probs[0]),
            "suicide": float(probs[1])
        },
        "word_contributions": word_contributions,
        "positive_contributors": positive_words,
        "negative_contributors": negative_words
    }

def format_explanation_text(explanation: Dict) -> str:
    """
    Format the explanation as human-readable text.
    
    Args:
        explanation: Dictionary from get_explanation_summary
        
    Returns:
        Formatted string explanation
    """
    lines = [
        f"Prediction: {explanation['predicted_class']}",
        f"Suicide Probability: {explanation['suicide_probability']:.2%}",
        "",
        "Top Contributing Words (toward suicide prediction):"
    ]
    
    for word_info in explanation["positive_contributors"][:5]:
        lines.append(f"  + {word_info['word']}: {word_info['weight']:.4f}")
    
    lines.append("")
    lines.append("Top Protective Words (against suicide prediction):")
    
    for word_info in explanation["negative_contributors"][:5]:
        lines.append(f"  - {word_info['word']}: {word_info['weight']:.4f}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    # Test the explainer
    test_texts = [
        "I feel so hopeless, I don't want to live anymore.",
        "I had a great day at work and feel happy about my life."
    ]
    
    model = load_model()
    
    for text in test_texts:
        print("=" * 60)
        print(f"Text: {text[:50]}...")
        print()
        
        explanation = get_explanation_summary(text, model)
        print(format_explanation_text(explanation))
        print()
