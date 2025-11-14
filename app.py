import streamlit as st
import pickle
import re


# --- same clean_text you used during training ---
def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Load saved model and vectorizer
tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("models/sentiment_model.pkl", "rb"))


def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    return "Positive 😊" if pred == 1 else "Negative 😞"


# --- Streamlit UI ---
st.title("Movie Review Sentiment Analyzer 🎬")

user_input = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = predict_sentiment(user_input)
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter a review first!")
