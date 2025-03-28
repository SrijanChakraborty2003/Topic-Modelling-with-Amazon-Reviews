import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)
with open("lda_model.pkl", "rb") as model_file:
    lda_model = pickle.load(model_file)
topic_labels = {
    0: "Clothing & Style",
    1: "Product Quality & Effectiveness",
    2: "Electronics & Performance",
    3: "General Experience & Usability",
    4: "Beauty & Skincare"
}
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)
def predict_topic(text):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    topic_distribution = lda_model.transform(text_vector)
    topic_index = topic_distribution.argmax()
    return topic_labels[topic_index]
st.title("Amazon Reviews Topic Modeling")
st.write("Enter a product review to predict its category.")
user_input = st.text_area("Enter your review:")
if st.button("Predict Topic"):
    if user_input:
        predicted_topic = predict_topic(user_input)
        st.success(f"Predicted Topic: {predicted_topic}")
    else:
        st.warning("Please enter a review before predicting.")
