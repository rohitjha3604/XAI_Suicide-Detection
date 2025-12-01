# 🧠 XAI Suicide Detection and Prevention System

An Explainable AI system for detecting potential suicide risk in text using machine learning and interpretable AI techniques.

## ⚠️ Important Disclaimer

**This tool is for EDUCATIONAL and RESEARCH purposes ONLY.**

- This is NOT a clinical diagnostic tool
- This should NEVER replace professional mental health evaluation
- If you or someone you know is in crisis, please contact emergency services immediately

### Crisis Resources
- **National Suicide Prevention Lifeline (US)**: 988
- **Crisis Text Line**: Text HOME to 741741
- **International Association for Suicide Prevention**: [Crisis Centres](https://www.iasp.info/resources/Crisis_Centres/)

---

## 📋 Project Overview

This system uses Natural Language Processing (NLP) and Explainable AI (XAI) to:

1. **Classify text** as potentially indicating suicide risk or not
2. **Explain predictions** using LIME (Local Interpretable Model-agnostic Explanations)
3. **Generate human-friendly explanations** using Google Gemini AI (optional)

## 🗂️ Project Structure

```
xai_suicide/
├── data/                      # Dataset storage
│   └── suicide_watch.csv      # Downloaded dataset
├── models/                    # Trained model storage
│   └── suicide_text_model.pkl # Trained pipeline
├── src/
│   ├── download_data.py       # Dataset download script
│   ├── train_model.py         # Model training script
│   ├── xai_lime.py            # LIME explainability
│   └── gemini_explain.py      # Gemini AI explanations
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📊 Dataset

- **Source**: [Kaggle - Suicide Watch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
- **Content**: Reddit posts labeled as suicidal or non-suicidal
- **Size**: ~230,000 posts

## 🤖 Machine Learning Model

- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
  - Max features: 10,000
  - N-gram range: (1, 2)
  - English stop words removed

- **Classifier**: Logistic Regression
  - Balanced class weights
  - Max iterations: 1,000

## 🔍 Explainability Methods

### LIME (Local Interpretable Model-agnostic Explanations)
- Provides word-level contribution scores
- Shows which words increased/decreased suicide probability
- Model-agnostic approach works with any classifier

### Google Gemini AI (Optional)
- Converts technical LIME output to natural language
- Provides empathetic, human-friendly explanations
- Requires API key (set as environment variable)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd XAI_Suicide-main
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API
Create `~/.kaggle/kaggle.json` with your Kaggle credentials:
```json
{"username":"your_username","key":"your_api_key"}
```

### 5. Download Dataset
```bash
python src/download_data.py
```

### 6. Train Model
```bash
python src/train_model.py
```

### 7. (Optional) Set Gemini API Key
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

### 8. Run the Application
```bash
streamlit run app.py
```

## 🖥️ Usage

1. Open the Streamlit app in your browser (usually http://localhost:8501)
2. Enter or paste text in the text area
3. Click "Analyze Text"
4. View:
   - Suicide probability score
   - LIME word contributions
   - Natural language explanation

## 🛡️ Ethical Considerations

- **Privacy**: No text is stored permanently
- **Limitations**: Model may produce false positives/negatives
- **Bias**: Model trained on Reddit data may not generalize to all populations
- **Use Case**: Research and education only, not clinical deployment

## 📄 License

This project is for educational purposes. Please use responsibly.

## 🙏 Acknowledgments

- Dataset by [Nikhileswar Komati](https://www.kaggle.com/nikhileswarkomati)
- LIME library by Marco Tulio Ribeiro
- Streamlit for the web framework
- Google Gemini for natural language explanations
