import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved model + vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

# Text cleaning
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # keep only letters
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.title("📄 AI Resume Classifier")
st.write("Upload your resume text and get predicted job category.")

resume_text = st.text_area("Paste your Resume Text Here:")

if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning("⚠️ Please paste your resume text.")
    else:
        cleaned = clean_text(resume_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"✅ Predicted Job Category: **{prediction}**")
