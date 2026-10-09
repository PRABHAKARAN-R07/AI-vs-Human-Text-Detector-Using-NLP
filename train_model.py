

import re
import nltk
import pandas as pd
import joblib

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# Find the project folder
BASE_DIR = Path(__file__).resolve().parent

# CSV file is inside the data folder
DATA_PATH = BASE_DIR / "data" / "dataset.csv"

# Save the trained model in the project folder
MODEL_PATH = BASE_DIR / "model.joblib"


# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(words)


# Check whether the dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Make sure dataset.csv is inside the data folder."
    )


# Load the CSV dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# Validate required columns
required_columns = {"text", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "CSV must contain these columns: text and label. "
        f"Found: {df.columns.tolist()}"
    )


# Remove missing values
df = df.dropna(subset=["text", "label"]).copy()

# Normalize labels and text
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["text"] = df["text"].astype(str).str.strip()

# Keep only valid labels
df = df[df["label"].isin(["ai", "human"])]

# Remove empty text
df = df[df["text"] != ""]

# Check label counts
print("\nLabel distribution:")
print(df["label"].value_counts())

if df["label"].nunique() != 2:
    raise ValueError(
        "Dataset must contain both AI and Human labels."
    )

if df["label"].value_counts().min() < 2:
    raise ValueError(
        "Each label must contain at least two samples."
    )


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# Build the machine-learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            max_features=10000
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# Train the model
print("\nTraining model...")
model.fit(X_train, y_train)


# Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel evaluation completed!")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# Save the trained model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")
print("Training completed!")