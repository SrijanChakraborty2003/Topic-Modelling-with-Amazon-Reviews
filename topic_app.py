import streamlit as st
import joblib
import pandas as pd
lda_model = joblib.load("lda_model (1).pkl")
vectorizer = joblib.load("vectorizer (1).pkl")
topic_labels = joblib.load("topic_labels.pkl")
st.title("🔍 Product Review Classifier")
st.write("Enter a product review, and the model will predict its category.")
user_review = st.text_area("Enter your review here:", "")
if st.button("Predict Category"):
    if user_review.strip():
        user_vector = vectorizer.transform([user_review])
        topic_id = lda_model.transform(user_vector).argmax(axis=1)[0]
        predicted_category = topic_labels.get(topic_id, "Unknown")
        st.success(f"**Predicted Category:** {predicted_category}")
    else:
        st.warning("Please enter a review before clicking predict.")
