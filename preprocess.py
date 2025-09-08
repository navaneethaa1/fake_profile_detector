import re
import pandas as pd
def preprocess(df):
	df["bio_text"] = df["bio_text"].fillna("")
	df["bio_has_digits"] = df["bio_text"].str.contains(r"\d").astype(int)
	df["bio_has_link"] = df["bio_text"].str.contains(r"http").astype(int)
	df["bio_length"] = df["bio_text"].str.len()
	df["bio_word_count"] = df["bio_text"].str.split().apply(len)
	df = df.drop(columns = ["bio_text"])
	return df
