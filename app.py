import re
from pathlib import Path

import joblib
import nltk
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -------------------------------
# 1. Project configuration
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = Flask(__name__)


# -------------------------------
# 2. Text preprocessing
# Must match train_model.py
# -------------------------------

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(words)


# -------------------------------
# 3. Load trained model
# -------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "model.joblib not found. Train your model first."
    )

model = joblib.load(MODEL_PATH)


# -------------------------------
# 4. Home page
# -------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# 5. Prediction API
# -------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")

        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "error": "Please enter a paragraph."
            }), 400

        # Predict using the trained model
        prediction = str(model.predict([text])[0]).lower()

        # Calculate predicted-class probability
        probabilities = model.predict_proba([text])[0]
        classes = [str(label).lower() for label in model.classes_]

        confidence = float(
            probabilities[classes.index(prediction)]
        ) * 100

        # Return result to JavaScript
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "word_count": len(text.split()),
            "character_count": len(text)
        })

    except Exception:
        app.logger.exception("Prediction error")

        return jsonify({
            "error": "Prediction failed. Check the Flask terminal."
        }), 500


# -------------------------------
# 6. Run Flask application
# -------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

