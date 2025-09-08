
import os, re, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/clean_profiles.csv")
df["bio_text"] = df.get("bio_text", "").fillna("").astype(str)
df["bio_length"] = df["bio_text"].apply(len)
df["bio_word_count"] = df["bio_text"].apply(lambda x: len(x.split()))
df["bio_has_link"] = df["bio_text"].apply(lambda x: 1 if "http" in x.lower() else 0)
df["bio_has_digits"] = df["bio_text"].apply(lambda x: 1 if any(c.isdigit() for c in x) else 0)
if "bio_text" in df.columns:
    df = df.drop("bio_text", axis=1)
label_enc = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = label_enc.fit_transform(df[col].astype(str))
if "label" not in df.columns:
    raise RuntimeError("No 'label' column found in data/clean_profiles.csv")
X = df.drop("label", axis=1)
y = df["label"].astype(int)

print("Before SMOTE:", y.value_counts())
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("After SMOTE:", y_res.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
model_path = os.path.join("models", "fake_profile_detector.pkl")
feat_path = os.path.join("models", "feature_names.pkl")
joblib.dump(model, model_path)
joblib.dump(X.columns.tolist(), feat_path)
print("Model saved to", model_path)
print("Feature names saved to", feat_path)