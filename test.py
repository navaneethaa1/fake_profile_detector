
import os, re, joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
model_path = os.path.join("models", "fake_profile_detector.pkl")
feat_path = os.path.join("models", "feature_names.pkl")
if not os.path.exists(model_path):
    if os.path.exists("fake_profile_detector.pkl"):
        model_path = "fake_profile_detector.pkl"
    else:
        raise FileNotFoundError(
            "Model not found. Please run train.py first to create 'models/fake_profile_detector.pkl' "
            "or place the model file in the project root."
        )
model = joblib.load(model_path)
feature_names = None
if os.path.exists(feat_path):
    feature_names = joblib.load(feat_path)
else:
    print("feature_names.pkl not found — test will try to use dataset columns (may cause mismatches).")
df = pd.read_csv("data/clean_profiles.csv")
if "bio_text" in df.columns:
    df["bio_text"] = df["bio_text"].fillna("").astype(str)
    df["bio_length"] = df["bio_text"].apply(len)
    df["bio_word_count"] = df["bio_text"].apply(lambda x: len(x.split()))
    df["bio_has_link"] = df["bio_text"].apply(lambda x: 1 if "http" in x.lower() else 0)
    df["bio_has_digits"] = df["bio_text"].apply(lambda x: 1 if any(c.isdigit() for c in x) else 0)
    df = df.drop("bio_text", axis=1)
if "label" in df.columns:
    y = df["label"].astype(int)
    X = df.drop("label", axis=1)
else:
    y = None
    X = df.copy()
if feature_names is not None:
    X = X.reindex(columns=feature_names, fill_value=0)
preds = model.predict(X)
print("Predictions (first 20):", preds[:20])

if y is not None:
    print("Accuracy on dataset:", accuracy_score(y, preds))
    print(classification_report(y, preds))