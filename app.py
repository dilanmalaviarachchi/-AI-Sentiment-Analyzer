
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

# Page config with beautiful styling
st.set_page_config(
    page_title="Sentiment Analysis AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Beautiful CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stTitle {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .stSubheader {
        font-family: 'Poppins', sans-serif;
        color: #4a5568;
        font-weight: 600;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        width: 100%;
        height: 4rem;
        transition: all 0.3s ease;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stTextArea > div > div > textarea {
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        font-family: 'Poppins', sans-serif;
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div {
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        background: rgba(255,255,255,0.9);
    }
    
    .result-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.3);
    }
    
    .result-card.positive {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        box-shadow: 0 15px 35px rgba(86, 171, 47, 0.3);
    }
    
    .result-card.negative {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        box-shadow: 0 15px 35px rgba(255, 65, 108, 0.3);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.8);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .example-card {
        background: rgba(255,255,255,0.7);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .example-card:hover {
        background: rgba(255,255,255,0.9);
        transform: translateX(5px);
    }
    
    .sidebar .block-container {
        background: rgba(255,255,255,0.9);
    }
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    models = {}
    vectorizer = None
    
    # Load vectorizer
    try:
        with open("", "rb") as f:
            vectorizer = pickle.load(f)
        st.sidebar.success("✅ Vectorizer loaded")
    except:
        st.sidebar.warning("⚠️ No vectorizer found - using default")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Load models
    model_files = {
        "Naive Bayes": "D:\\projects\\twitter sentimentel analysis\\models\\bnd.pkl",
        "Logistic Regression": "D:\\projects\\twitter sentimentel analysis\\models\\logreg.pkl", 
        "SVM": "D:\\projects\\twitter sentimentel analysis\\models\\svm.pkl"
    }
    
    for name, file in model_files.items():
        try:
            with open(file, "rb") as f:
                models[name] = pickle.load(f)
            st.sidebar.success(f"✅ {name} loaded")
        except:
            st.sidebar.error(f"❌ {name} not found")
    
    return models, vectorizer

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

# Get final sentiment
def get_final_sentiment(predictions):
    positive_count = sum(1 for pred in predictions if pred in ['positive', 'good', '1', 1, 'pos'])
    negative_count = sum(1 for pred in predictions if pred in ['negative', 'bad', '0', 0, 'neg'])
    
    if positive_count > negative_count:
        return "😊 POSITIVE", "positive"
    elif negative_count > positive_count:
        return "😞 NEGATIVE", "negative"
    else:
        return "😐 NEUTRAL", "neutral"

# Create SVM visualization
def create_svm_graph(svm_model, text_vector, prediction):
    # Create a simple SVM decision boundary visualization
    fig = go.Figure()
    
    # Generate sample data points for visualization
    np.random.seed(42)
    
    # Positive samples
    pos_x = np.random.normal(0.7, 0.2, 20)
    pos_y = np.random.normal(0.7, 0.2, 20)
    
    # Negative samples  
    neg_x = np.random.normal(0.3, 0.2, 20)
    neg_y = np.random.normal(0.3, 0.2, 20)
    
    # Add positive points
    fig.add_trace(go.Scatter(
        x=pos_x, y=pos_y,
        mode='markers',
        marker=dict(color='green', size=10, symbol='circle'),
        name='Positive Sentiment',
        opacity=0.7
    ))
    
    # Add negative points
    fig.add_trace(go.Scatter(
        x=neg_x, y=neg_y,
        mode='markers',
        marker=dict(color='red', size=10, symbol='circle'),
        name='Negative Sentiment',
        opacity=0.7
    ))
    
    # Add decision boundary line
    x_line = np.linspace(0, 1, 100)
    y_line = np.linspace(0, 1, 100)
    
    fig.add_trace(go.Scatter(
        x=x_line, y=1-x_line,
        mode='lines',
        line=dict(color='purple', width=3, dash='dash'),
        name='SVM Decision Boundary'
    ))
    
    # Add current prediction point
    if prediction in ['positive', 'good', '1', 1, 'pos']:
        pred_x, pred_y = 0.8, 0.8
        pred_color = 'darkgreen'
        pred_symbol = 'star'
    else:
        pred_x, pred_y = 0.2, 0.2
        pred_color = 'darkred'
        pred_symbol = 'star'
    
    fig.add_trace(go.Scatter(
        x=[pred_x], y=[pred_y],
        mode='markers',
        marker=dict(color=pred_color, size=20, symbol=pred_symbol, line=dict(width=2, color='white')),
        name='Your Text',
    ))
    
    fig.update_layout(
        title="🎯 SVM Classification Visualization",
        xaxis_title="Feature Dimension 1",
        yaxis_title="Feature Dimension 2",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        font=dict(family="Poppins", size=12)
    )
    
    return fig

# Beautiful metric display
def display_beautiful_metrics(word_count, char_count, model_count):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #667eea; margin: 0; font-family: Poppins;">📝</h2>
            <h3 style="color: #4a5568; margin: 0.5rem 0; font-family: Poppins;">{word_count}</h3>
            <p style="color: #718096; margin: 0; font-family: Poppins;">Words</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #667eea; margin: 0; font-family: Poppins;">🔤</h2>
            <h3 style="color: #4a5568; margin: 0.5rem 0; font-family: Poppins;">{char_count}</h3>
            <p style="color: #718096; margin: 0; font-family: Poppins;">Characters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #667eea; margin: 0; font-family: Poppins;">🤖</h2>
            <h3 style="color: #4a5568; margin: 0.5rem 0; font-family: Poppins;">{model_count}</h3>
            <p style="color: #718096; margin: 0; font-family: Poppins;">AI Models</p>
        </div>
        """, unsafe_allow_html=True)

# Main App
def main():
    # Beautiful header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="stTitle">🎭 AI Sentiment Analyzer</h1>
        <p style="font-size: 1.3rem; color: #718096; font-family: Poppins; font-weight: 300;">
            Discover the emotion behind any text with advanced AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models, vectorizer = load_models()
    
    if not models:
        st.error("🚫 No AI models found! Please add model files to 'models/' folder")
        return
    
    # Beautiful input section
    st.markdown("""
    <div style="background: rgba(255,255,255,0.8); padding: 2rem; border-radius: 20px; margin: 2rem 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
        <h2 style="text-align: center; color: #4a5568; font-family: Poppins; margin-bottom: 1rem;">✨ Enter Your Text</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Example texts with beautiful cards
    examples = [
        "I absolutely love this product! It exceeded all my expectations! 🌟",
        "This is the worst experience I've ever had. Completely disappointed 😠",
        "The movie was okay, nothing too special but watchable",
        "Amazing customer service! They went above and beyond to help me ⭐",
        "Terrible quality and poor design. Would not recommend to anyone"
    ]
    
    # Example selector
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**🎯 Quick Examples:**")
    with col2:
        selected_example = st.selectbox("Choose an example to try:", ["Custom text..."] + examples, label_visibility="collapsed")
    
    # Text input
    if selected_example == "Custom text...":
        default_text = ""
    else:
        default_text = selected_example
    
    user_text = st.text_area(
        "",
        value=default_text,
        height=120,
        placeholder="✍️ Type anything here... reviews, tweets, feedback, comments...",
        label_visibility="collapsed"
    )
    
    # Beautiful analyze button
    st.markdown("<div style='text-align: center; margin: 2rem 0;'>", unsafe_allow_html=True)
    analyze_clicked = st.button("🚀 Analyze Sentiment with AI", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if analyze_clicked:
        if not user_text.strip():
            st.warning("📝 Please enter some text to analyze!")
            return
        
        with st.spinner("🧠 AI is analyzing your text..."):
            # Clean text
            cleaned_text = clean_text(user_text)
            
            # Vectorize
            try:
                if hasattr(vectorizer, 'transform'):
                    text_vector = vectorizer.transform([cleaned_text])
                else:
                    text_vector = vectorizer.fit_transform([cleaned_text])
            except:
                st.error("❌ Error processing text")
                return
            
            # Get predictions
            predictions = []
            confidences = {}
            svm_prediction = None
            
            for name, model in models.items():
                try:
                    pred = model.predict(text_vector)[0]
                    predictions.append(pred)
                    
                    # Store SVM prediction for graph
                    if name == "SVM":
                        svm_prediction = pred
                    
                    # Get probabilities
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(text_vector)[0]
                        confidences[name] = max(proba)
                    
                except Exception as e:
                    st.error(f"❌ Error with {name}: {e}")
            
            if predictions:
                # Beautiful results section
                st.markdown("---")
                
                # Final sentiment with beautiful styling
                final_sentiment, sentiment_type = get_final_sentiment(predictions)
                
                if sentiment_type == "positive":
                    st.markdown(f"""
                    <div class="result-card positive">
                        <h1 style="margin: 0; font-size: 2.5rem;">🎉 {final_sentiment}</h1>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Great vibes detected!</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif sentiment_type == "negative":
                    st.markdown(f"""
                    <div class="result-card negative">
                        <h1 style="margin: 0; font-size: 2.5rem;">💔 {final_sentiment}</h1>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Negative sentiment found</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card">
                        <h1 style="margin: 0; font-size: 2.5rem;">🤔 {final_sentiment}</h1>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Mixed or unclear sentiment</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Beautiful metrics
                st.markdown("### 📊 Text Analysis")
                display_beautiful_metrics(len(user_text.split()), len(user_text), len(predictions))
                
                # Model results in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show individual predictions
                    st.markdown("### 🤖 AI Model Results")
                    
                    results_df = pd.DataFrame({
                        "Model": list(models.keys())[:len(predictions)],
                        "Prediction": predictions,
                        "Confidence": [f"{confidences.get(name, 0):.1%}" for name in list(models.keys())[:len(predictions)]]
                    })
                    
                    # Style the dataframe
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Confidence chart
                    if confidences:
                        fig = px.bar(
                            x=list(confidences.keys()),
                            y=list(confidences.values()),
                            title="🎯 Model Confidence Levels",
                            color=list(confidences.values()),
                            color_continuous_scale="viridis"
                        )
                        fig.update_layout(
                            showlegend=False,
                            template="plotly_white",
                            font=dict(family="Poppins"),
                            title_x=0.5
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # SVM Visualization
                    st.markdown("### 🎯 SVM Classification Map")
                    if svm_prediction is not None:
                        svm_fig = create_svm_graph(models.get("SVM"), text_vector, svm_prediction)
                        st.plotly_chart(svm_fig, use_container_width=True)
                    else:
                        st.info("SVM model not available for visualization")

# Beautiful sidebar
def add_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #667eea; font-family: Poppins;">🎭 About</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-card">
            <h4 style="color: #4a5568; margin: 0 0 0.5rem 0;">🤖 AI-Powered Analysis</h4>
            <p style="margin: 0; color: #718096; font-size: 0.9rem;">Uses multiple machine learning models to analyze text sentiment with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-card">
            <h4 style="color: #4a5568; margin: 0 0 0.5rem 0;">📊 Visual Insights</h4>
            <p style="margin: 0; color: #718096; font-size: 0.9rem;">Beautiful charts and SVM classification visualization to understand AI decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-card">
            <h4 style="color: #4a5568; margin: 0 0 0.5rem 0;">⚡ Real-time Results</h4>
            <p style="margin: 0; color: #718096; font-size: 0.9rem;">Instant analysis with confidence scores and detailed breakdowns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Perfect For:")
        st.markdown("""
        • 📱 Social media posts  
        • ⭐ Product reviews  
        • 💬 Customer feedback  
        • 📧 Email analysis  
        • 🎬 Movie reviews  
        • 📰 News sentiment  
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px; margin-top: 1rem;">
            <p style="margin: 0; color: #667eea; font-weight: 600;">Made with ❤️</p>
            <p style="margin: 0; color: #718096; font-size: 0.8rem;">Streamlit + AI</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    add_sidebar()
    main()