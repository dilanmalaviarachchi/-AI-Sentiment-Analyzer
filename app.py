import streamlit as st
import pickle
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

# Simple CSS styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .result-positive {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-negative {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-neutral {
        background-color: #e2e3e5;
        border: 1px solid #d6d8db;
        color: #383d41;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stats-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .model-result {
        background-color: #fff;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    models = {}
    vectorizer = None
    
    try:
        with open("D:\\projects\\twitter sentimentel analysis\\models\\vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    model_files = {
        "Naive Bayes": "D:\\projects\\twitter sentimentel analysis\\models\\bnd.pkl",
        "Logistic Regression": "D:\\projects\\twitter sentimentel analysis\\models\\logreg.pkl", 
        "SVM": "D:\\projects\\twitter sentimentel analysis\\models\\svm.pkl"
    }
    
    for name, file in model_files.items():
        try:
            with open(file, "rb") as f:
                models[name] = pickle.load(f)
        except Exception as e:
            st.error(f"{name} model not found: {str(e)}")
    
    return models, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def get_final_sentiment(predictions):
    positive_count = sum(1 for pred in predictions if pred in ['positive', 'good', '1', 1, 'pos'])
    negative_count = sum(1 for pred in predictions if pred in ['negative', 'bad', '0', 0, 'neg'])
    
    if positive_count > negative_count:
        return "😊 POSITIVE", "positive"
    elif negative_count > positive_count:
        return "😞 NEGATIVE", "negative"
    else:
        return "😐 NEUTRAL", "neutral"

def get_sentiment_emoji(prediction):
    if prediction in ['positive', 'good', '1', 1, 'pos']:
        return "😊"
    elif prediction in ['negative', 'bad', '0', 0, 'neg']:
        return "😞"
    else:
        return "😐"

def main():
    # Title
    st.markdown('<h1 class="title">Sentiment Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze the sentiment of any text using machine learning**")
    
    # Load models
    models, vectorizer = load_models()
    
    if not models:
        st.error("No models loaded. Please check your model files.")
        return
    
    # Example texts
    examples = [
        "I love this product! It's amazing!",
        "This is terrible, I'm very disappointed",
        "It's okay, nothing special",
        "The service was average but the food was great"
    ]
    
    # Input section
    st.subheader("Enter Your Text")
    
    selected_example = st.selectbox(
        "Choose an example or select 'Custom text':", 
        ["Custom text"] + examples
    )
    
    if selected_example == "Custom text":
        user_text = st.text_area("Enter your text here:", height=100)
    else:
        user_text = st.text_area("Enter your text here:", value=selected_example, height=100)
    
    # Analyze button
    if st.button("Analyze Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text to analyze")
            return
        
        with st.spinner("Analyzing..."):
            try:
                cleaned_text = clean_text(user_text)
                text_vector = vectorizer.transform([cleaned_text])
                
                predictions = []
                model_results = []
                
                for name, model in models.items():
                    try:
                        pred = model.predict(text_vector)[0]
                        predictions.append(pred)
                        
                        confidence = ""
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(text_vector)[0]
                            confidence = f"{max(proba):.1%}"
                        
                        model_results.append({
                            "model": name,
                            "prediction": pred,
                            "emoji": get_sentiment_emoji(pred),
                            "confidence": confidence
                        })
                        
                    except Exception as e:
                        st.error(f"Error with {name}: {str(e)}")
                
                if predictions:
                    final_sentiment, sentiment_type = get_final_sentiment(predictions)
                    
                    # Show overall result
                    st.subheader("Result")
                    result_class = f"result-{sentiment_type}"
                    st.markdown(f"""
                    <div class="{result_class}">
                        <h3>Overall Sentiment: {final_sentiment}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show statistics
                    st.subheader("Text Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Words", len(cleaned_text.split()))
                    with col2:
                        st.metric("Characters", len(user_text))
                    with col3:
                        st.metric("Models Used", len(models))
                    
                    # Show individual model predictions
                    st.subheader("Individual Model Predictions")
                    for result in model_results:
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{result['model']}**")
                            with col2:
                                st.write(f"{result['emoji']} {result['prediction']}")
                            with col3:
                                st.write(result['confidence'])
                    
                    # Show original text
                    st.subheader("Original Text")
                    st.text_area("", value=user_text, height=80, disabled=True)
                    
                    # Show processed text
                    with st.expander("Show processed text"):
                        st.text(cleaned_text)
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()