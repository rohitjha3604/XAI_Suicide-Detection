"""
Streamlit Web Application for XAI Suicide Detection and Prevention System.
Provides an interactive interface for text analysis with explainable predictions.
"""

import streamlit as st
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from xai_lime import load_model, get_explanation_summary
from gemini_explain import explain_with_gemini, is_gemini_available

# Page configuration
st.set_page_config(
    page_title="XAI Suicide Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background-color: #0e0f12 !important;
        color: #e0e0e0 !important;
    }
    
    /* Main container */
    .main .block-container {
        background-color: #0e0f12 !important;
        color: #e0e0e0 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1b1d22 !important;
        border-right: 1px solid #2a2d33 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
    }
    
    /* Paragraphs and text */
    p, span, label, .stMarkdown {
        color: #c5c5c5 !important;
    }
    
    /* Text input and text area */
    .stTextInput input, .stTextArea textarea {
        background-color: #1b1d22 !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2d33 !important;
        border-radius: 5px !important;
    }
    
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #888888 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1b1d22 !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2d33 !important;
        border-radius: 5px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #2a2d33 !important;
        border-color: #3399ff !important;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #3399ff !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #2277dd !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1b1d22 !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2d33 !important;
        border-radius: 5px !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1b1d22 !important;
        border: 1px solid #2a2d33 !important;
        border-top: none !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #c5c5c5 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #3bd671 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #3399ff !important;
    }
    
    .stProgress > div {
        background-color: #2a2d33 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3399ff !important;
    }
    
    /* Divider */
    hr {
        border-color: #2a2d33 !important;
    }
    
    /* Alert boxes - Warning */
    .warning-box {
        background-color: rgba(255, 191, 71, 0.15) !important;
        border: 1px solid #ffbf47 !important;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        color: #ffbf47 !important;
    }
    
    .warning-box strong {
        color: #ffbf47 !important;
    }
    
    .warning-box br + * {
        color: #e0e0e0 !important;
    }
    
    /* Alert boxes - Danger */
    .danger-box {
        background-color: rgba(255, 75, 75, 0.15) !important;
        border: 1px solid #ff4b4b !important;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        color: #ff4b4b !important;
    }
    
    .danger-box strong {
        color: #ff4b4b !important;
    }
    
    /* Alert boxes - Success */
    .success-box {
        background-color: rgba(59, 214, 113, 0.15) !important;
        border: 1px solid #3bd671 !important;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        color: #3bd671 !important;
    }
    
    .success-box strong {
        color: #3bd671 !important;
    }
    
    /* Alert boxes - Info */
    .info-box {
        background-color: rgba(51, 153, 255, 0.15) !important;
        border: 1px solid #3399ff !important;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        color: #e0e0e0 !important;
    }
    
    .info-box strong {
        color: #3399ff !important;
    }
    
    /* LIME word chips - Risk indicators (high contrast red) */
    .word-positive {
        background-color: #ff4b4b !important;
        color: #ffffff !important;
        padding: 4px 10px;
        border-radius: 4px;
        margin: 3px;
        display: inline-block;
        font-weight: 600;
        font-size: 0.9em;
        border: 1px solid #ff6b6b;
    }
    
    /* LIME word chips - Protective factors (high contrast green) */
    .word-negative {
        background-color: #3bd671 !important;
        color: #000000 !important;
        padding: 4px 10px;
        border-radius: 4px;
        margin: 3px;
        display: inline-block;
        font-weight: 600;
        font-size: 0.9em;
        border: 1px solid #4de682;
    }
    
    /* Streamlit native alerts */
    .stAlert {
        background-color: #1b1d22 !important;
        border: 1px solid #2a2d33 !important;
        color: #e0e0e0 !important;
    }
    
    /* Success alert */
    [data-testid="stAlert"][data-baseweb="notification"] {
        background-color: #1b1d22 !important;
    }
    
    /* Info boxes in sidebar */
    .stInfo, .stWarning, .stSuccess, .stError {
        background-color: #1b1d22 !important;
        color: #e0e0e0 !important;
    }
    
    /* Card panels */
    .card-panel {
        background-color: #1b1d22 !important;
        border: 1px solid #2a2d33 !important;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }
    
    /* Links */
    a {
        color: #3399ff !important;
    }
    
    a:hover {
        color: #66b3ff !important;
    }
    
    /* Lists */
    ul, ol {
        color: #c5c5c5 !important;
    }
    
    li {
        color: #c5c5c5 !important;
    }
    
    /* Code blocks */
    code {
        background-color: #1b1d22 !important;
        color: #3bd671 !important;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1b1d22;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2a2d33;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3a3d43;
    }
</style>
""", unsafe_allow_html=True)

def display_header():
    """Display the application header with disclaimers."""
    st.title("🧠 XAI Suicide Detection & Prevention System")
    
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important Disclaimer</strong><br>
        <span style="color: #e0e0e0;">This tool is for <strong style="color: #ffbf47;">educational and research purposes only</strong>. 
        It is NOT a substitute for professional mental health evaluation or crisis intervention.
        If you or someone you know is in crisis, please contact emergency services or a crisis hotline immediately.</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📞 Crisis Resources"):
        st.markdown("""
        - **National Suicide Prevention Lifeline (US)**: 988
        - **Crisis Text Line**: Text HOME to 741741
        - **International Association for Suicide Prevention**: [Crisis Centres](https://www.iasp.info/resources/Crisis_Centres/)
        - **UK Samaritans**: 116 123
        - **Canada Crisis Services**: 1-833-456-4566
        """)

def display_sidebar():
    """Display sidebar with information about the system."""
    with st.sidebar:
        st.header("About This System")
        
        st.markdown("""
        ### How It Works
        
        1. **Text Classification**: Uses TF-IDF + Logistic Regression trained on Reddit data
        2. **LIME Explanation**: Shows which words influenced the prediction
        3. **Natural Language**: Gemini AI provides human-friendly explanations
        
        ### Model Information
        - **Dataset**: Suicide Watch (Reddit posts)
        - **Algorithm**: TF-IDF + Logistic Regression
        - **XAI Method**: LIME TextExplainer
        """)
        
        st.divider()
        
        st.markdown("""
        ### Gemini API Status
        """)
        
        if is_gemini_available():
            st.success("✅ Gemini API Connected")
        else:
            st.warning("⚠️ Gemini API not configured. Using template explanations.")
            st.info("Set GEMINI_API_KEY environment variable to enable AI explanations.")

@st.cache_resource
def get_model():
    """Load and cache the model. Downloads from Google Drive if not available locally."""
    try:
        with st.spinner("Loading model... (downloading from Google Drive if needed)"):
            return load_model()
    except FileNotFoundError as e:
        st.error(f"Model not found: {e}")
        st.info("Please run `python src/train_model.py` first to train the model.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def display_results(explanation: dict):
    """Display the analysis results."""
    
    # Prediction result
    st.subheader("📊 Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Probability gauge
        suicide_prob = explanation["suicide_probability"]
        
        if suicide_prob >= 0.7:
            st.markdown(f"""
            <div class="danger-box">
                <strong>⚠️ High Risk Detected</strong><br>
                <span style="color: #e0e0e0;">Suicide Probability: <strong style="color: #ff4b4b;">{suicide_prob:.1%}</strong></span>
            </div>
            """, unsafe_allow_html=True)
        elif suicide_prob >= 0.4:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚡ Moderate Concern</strong><br>
                <span style="color: #e0e0e0;">Suicide Probability: <strong style="color: #ffbf47;">{suicide_prob:.1%}</strong></span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Low Risk</strong><br>
                <span style="color: #e0e0e0;">Suicide Probability: <strong style="color: #3bd671;">{suicide_prob:.1%}</strong></span>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar for probability
        st.progress(suicide_prob)
    
    with col2:
        st.metric(
            label="Predicted Class",
            value=explanation["predicted_class"],
            delta=f"{explanation['confidence']:.1%} confidence"
        )

def display_lime_explanation(explanation: dict):
    """Display LIME word contributions."""
    st.subheader("🔍 LIME Explanation - Word Contributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 Risk Indicators** (increase suicide probability)")
        positive_words = explanation.get("positive_contributors", [])
        if positive_words:
            for word_info in positive_words:
                weight = word_info["weight"]
                st.markdown(
                    f'<span class="word-positive">{word_info["word"]}</span> '
                    f'(+{weight:.4f})',
                    unsafe_allow_html=True
                )
        else:
            st.info("No significant risk indicators found.")
    
    with col2:
        st.markdown("**🟢 Protective Factors** (decrease suicide probability)")
        negative_words = explanation.get("negative_contributors", [])
        if negative_words:
            for word_info in negative_words:
                weight = word_info["weight"]
                st.markdown(
                    f'<span class="word-negative">{word_info["word"]}</span> '
                    f'({weight:.4f})',
                    unsafe_allow_html=True
                )
        else:
            st.info("No significant protective factors found.")

def display_gemini_explanation(text: str, explanation: dict):
    """Display the Gemini natural language explanation."""
    st.subheader("💬 AI Explanation")
    
    with st.spinner("Generating explanation..."):
        gemini_explanation = explain_with_gemini(
            text=text,
            predicted_label=explanation["predicted_class"],
            probability=explanation["suicide_probability"],
            contributing_words=explanation["word_contributions"]
        )
    
    st.markdown(f"""
    <div class="info-box">
        <span style="color: #e0e0e0;">{gemini_explanation}</span>
    </div>
    """, unsafe_allow_html=True)

def display_footer():
    """Display footer with ethical considerations."""
    st.divider()
    
    st.markdown("""
    <div class="warning-box">
        <strong>🔒 Ethical Considerations & Privacy</strong><br>
        <ul style="color: #e0e0e0;">
            <li>This system does not store or transmit any text you enter</li>
            <li>All analysis is performed locally (except Gemini API calls if enabled)</li>
            <li>This tool should never replace professional mental health assessment</li>
            <li>False positives and negatives are possible - always err on the side of caution</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ---
    **XAI Suicide Detection System** | Built with Streamlit, scikit-learn, LIME, and Gemini AI  
    For educational and research purposes only.
    """)

def main():
    """Main application entry point."""
    display_header()
    display_sidebar()
    
    # Load model
    model = get_model()
    
    if model is None:
        st.stop()
    
    # Main content area
    st.subheader("📝 Enter Text for Analysis")
    
    # Text input
    text_input = st.text_area(
        "Enter the text you want to analyze:",
        height=150,
        placeholder="Type or paste text here...",
        help="The system will analyze this text for potential suicide risk indicators."
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    # Analysis
    if analyze_button:
        if not text_input or not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            try:
                with st.spinner("Analyzing text..."):
                    # Get LIME explanation
                    explanation = get_explanation_summary(text_input, model)
                
                # Display results
                display_results(explanation)
                
                # Display LIME explanation
                display_lime_explanation(explanation)
                
                # Display Gemini explanation
                display_gemini_explanation(text_input, explanation)
                
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                st.info("Please try again or check if the model is properly trained.")
    
    # Footer
    display_footer()

if __name__ == "__main__":
    main()
