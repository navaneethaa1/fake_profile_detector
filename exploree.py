import pandas as pd
import os
print("Current working directory:", os.getcwd())
print("Files in data folder:", os.listdir("data"))
df = pd.read_csv("data/profiles.csv")
print("First 5 rows:\n", df.head())
print("Columns:", df.columns.tolist())
print("Columns:", df.columns.tolist())
