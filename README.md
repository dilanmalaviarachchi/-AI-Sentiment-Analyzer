# AI-Sentiment-Analyzer

Project Description

This project is an AI-powered Sentiment Analysis Application that identifies emotions expressed in text and classifies them as positive, negative, or neutral. It is implemented using Streamlit for the interface and machine learning models trained on textual data.

The system integrates Natural Language Processing (NLP) techniques to clean and process raw user input. The preprocessing pipeline removes URLs, hashtags, user mentions, punctuation, and unnecessary spaces, and then applies a TF-IDF vectorizer to convert text into numerical features.

To ensure robustness, the application uses multiple models — Naive Bayes, Logistic Regression, and Support Vector Machine (SVM). Each model predicts sentiment individually, and their outputs are compared to produce a final aggregated sentiment. Confidence scores (if available) are also displayed to show how certain each model is about its prediction.

The Streamlit interface is designed to be intuitive and visually engaging. Users can either input their own custom text or choose from predefined examples. After analysis, the application presents results in a structured format:

An overall sentiment classification with emojis and color-coded highlights.

Text statistics such as word count, character count, and number of models used.

Model-by-model predictions, showing the sentiment, emoji, and confidence score of each classifier.

An expandable section to view the processed (cleaned) text used for analysis.

This project demonstrates how machine learning and NLP can be applied to real-world problems such as analyzing tweets, customer reviews, or feedback. The combination of multiple models improves reliability and provides users with greater insight into how different algorithms interpret the same input.

Beyond sentiment analysis, the application serves as a practical framework for deploying AI models in user-friendly web applications. It highlights the end-to-end process of text preprocessing, model integration, prediction, and visualization in a single tool.
