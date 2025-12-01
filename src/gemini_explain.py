"""
Natural language explanation using Google Gemini API.
Converts LIME outputs into human-friendly explanations.
"""

import os
from typing import List, Dict, Optional

def get_gemini_model():
    """
    Initialize and return the Gemini model.
    Requires GEMINI_API_KEY environment variable.
    
    Returns:
        Configured Gemini model or None if API key not set
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return None

def explain_with_gemini(
    text: str,
    predicted_label: str,
    probability: float,
    contributing_words: List[Dict],
    model=None
) -> str:
    """
    Generate a natural language explanation using Gemini API.
    
    Args:
        text: Original input text
        predicted_label: Model's prediction ("Suicide" or "Non-Suicide")
        probability: Probability score for the prediction
        contributing_words: List of word contribution dicts from LIME
        model: Optional pre-initialized Gemini model
        
    Returns:
        Human-friendly explanation string
    """
    # Safety message to always include
    safety_message = (
        "\n\n⚠️ **Important Notice**: This is not a clinical diagnosis. "
        "This AI system is for informational purposes only. "
        "If you or someone you know is struggling with thoughts of suicide, "
        "please reach out to professional help immediately:\n"
        "- **National Suicide Prevention Lifeline**: 988 (US)\n"
        "- **Crisis Text Line**: Text HOME to 741741\n"
        "- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/"
    )
    
    # If Gemini is not available, provide a template-based explanation
    if model is None:
        model = get_gemini_model()
    
    if model is None:
        return generate_fallback_explanation(
            text, predicted_label, probability, contributing_words
        ) + safety_message
    
    try:
        # Prepare word lists for the prompt
        positive_words = [w["word"] for w in contributing_words if w.get("contribution") == "positive"][:5]
        negative_words = [w["word"] for w in contributing_words if w.get("contribution") == "negative"][:5]
        
        prompt = f"""You are a mental health AI assistant explaining suicide risk detection results.
Be empathetic, clear, and non-judgmental. Never provide medical advice.

Given this analysis:
- Text analyzed (first 200 chars): "{text[:200]}..."
- Model prediction: {predicted_label}
- Confidence: {probability:.1%}
- Words suggesting risk: {', '.join(positive_words) if positive_words else 'none identified'}
- Protective words: {', '.join(negative_words) if negative_words else 'none identified'}

Provide a brief (2-3 sentences), empathetic explanation of why the model made this prediction.
Focus on the linguistic patterns, not the person.
Start with "This message was classified as {predicted_label} because..."

Do NOT provide crisis resources (they will be added separately).
Do NOT make clinical judgments about the person.
"""
        
        response = model.generate_content(prompt)
        explanation = response.text.strip()
        
        return explanation + safety_message
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return generate_fallback_explanation(
            text, predicted_label, probability, contributing_words
        ) + safety_message

def generate_fallback_explanation(
    text: str,
    predicted_label: str,
    probability: float,
    contributing_words: List[Dict]
) -> str:
    """
    Generate a template-based explanation when Gemini is unavailable.
    
    Args:
        text: Original input text
        predicted_label: Model's prediction
        probability: Probability score
        contributing_words: Word contributions from LIME
        
    Returns:
        Template-based explanation string
    """
    positive_words = [w["word"] for w in contributing_words if w.get("contribution") == "positive"][:3]
    negative_words = [w["word"] for w in contributing_words if w.get("contribution") == "negative"][:3]
    
    if predicted_label == "Suicide":
        if positive_words:
            word_list = ", ".join(f"'{w}'" for w in positive_words)
            explanation = (
                f"This message was classified as potentially concerning (probability: {probability:.1%}) "
                f"because it contains language patterns associated with distress. "
                f"Words like {word_list} contributed to this assessment."
            )
        else:
            explanation = (
                f"This message was classified as potentially concerning (probability: {probability:.1%}) "
                f"based on the overall language patterns detected in the text."
            )
    else:
        if negative_words:
            word_list = ", ".join(f"'{w}'" for w in negative_words)
            explanation = (
                f"This message was classified as non-concerning (probability: {1-probability:.1%}) "
                f"because it contains positive or neutral language patterns. "
                f"Words like {word_list} suggest a lower risk level."
            )
        else:
            explanation = (
                f"This message was classified as non-concerning (probability: {1-probability:.1%}) "
                f"based on the overall positive or neutral language patterns."
            )
    
    return explanation

def is_gemini_available() -> bool:
    """
    Check if Gemini API is available and configured.
    
    Returns:
        True if Gemini is available, False otherwise
    """
    return os.environ.get("GEMINI_API_KEY") is not None

if __name__ == "__main__":
    # Test the explanation generator
    test_words = [
        {"word": "hopeless", "weight": 0.45, "contribution": "positive"},
        {"word": "suicide", "weight": 0.38, "contribution": "positive"},
        {"word": "happy", "weight": -0.25, "contribution": "negative"},
    ]
    
    explanation = explain_with_gemini(
        text="I feel so hopeless and don't want to go on.",
        predicted_label="Suicide",
        probability=0.87,
        contributing_words=test_words
    )
    
    print(explanation)
