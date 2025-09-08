import pandas as pd
df = pd.read_csv("data/profiles.csv")
df = df.drop_duplicates()
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df["bio_text"] = df["bio_text"].fillna("")
df.to_csv("data/clean_profiles.csv", index = False)
print("Cleaned dataset saved to data/clean_profiles.csv")
print("Rows:", df.shape[0], "Columns:",df.shape[1])