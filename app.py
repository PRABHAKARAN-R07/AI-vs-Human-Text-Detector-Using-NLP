import re
from pathlib import Path

import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------
# 1. Page configuration
# -----------------------------

st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI vs Human Text Detector")
st.write("Analyze text using NLP and Machine Learning.")


# -----------------------------
# 2. Text preprocessing
# -----------------------------

@st.cache_resource
def initialize_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    return (
        set(stopwords.words("english")),
        WordNetLemmatizer()
    )


stop_words, lemmatizer = initialize_nltk()


def clean_text(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(words)


# -----------------------------
# 3. Load trained model
# -----------------------------

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "model.joblib is missing from the GitHub repository."
        )

    return joblib.load(model_path)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# -----------------------------
# 4. Text input
# -----------------------------

text = st.text_area(
    "Enter your paragraph",
    height=220,
    placeholder="Paste your text here..."
)

word_count = len(text.split()) if text.strip() else 0
char_count = len(text)

col1, col2 = st.columns(2)

col1.metric("Word Count", word_count)
col2.metric("Character Count", char_count)


# -----------------------------
# 5. Predict
# -----------------------------

if st.button("🔍 Detect Text", use_container_width=True):

    if not text.strip():
        st.warning("Please enter a paragraph first.")

    else:
        try:
            prediction = str(model.predict([text])[0]).lower()

            probabilities = model.predict_proba([text])[0]
            classes = [str(label).lower() for label in model.classes_]

            confidence = float(
                probabilities[classes.index(prediction)]
            ) * 100

            st.subheader("Prediction Result")

            if prediction == "ai":
                st.error("🤖 AI-generated")
            elif prediction == "human":
                st.success("👤 Human-written")
            else:
                st.warning(f"Prediction: {prediction}")

            st.metric("Model Score", f"{confidence:.2f}%")

            st.caption(
                "The model score is an estimated probability, "
                "not proof of authorship."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# -----------------------------
# 6. Prediction history
# -----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Clear Text"):
    st.rerun()

