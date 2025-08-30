# train.py (≈90 lines) — trains and saves a tiny spam/ham text classifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Tiny, embedded dataset to keep project self-contained
texts = [
    "WIN a free iPhone now! Click this link",
    "You have been selected for a prize, claim now",
    "Lowest price guaranteed, buy now and save big",
    "Urgent: your account has been compromised, verify immediately",
    "Free vacation and hotel voucher for winners",
    "Get rich quick, invest in this scheme today",
    "Hi, are we still meeting tomorrow at 10am?",
    "Please see attached report and let me know your thoughts",
    "Thanks for your help last week",
    "Lunch at 12 works for me",
    "Reminder: project standup is at 9am",
    "Can you review the latest draft?",
]
labels = [
    "spam","spam","spam","spam","spam","spam",
    "ham","ham","ham","ham","ham","ham"
]

# Build a simple pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train/test split for quick sanity check
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)
pipeline.fit(X_train, y_train)

# Evaluate quickly (prints to console for developer info)
from sklearn.metrics import classification_report
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred, zero_division=0)
print(report)

# Persist the model
import joblib
joblib.dump(pipeline, "model.joblib")
print("Saved model to model.joblib")
