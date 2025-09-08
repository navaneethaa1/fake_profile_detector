import joblib
import pandas as pd
model = joblib.load("models/fake_profile_detector.pkl")
feature_names = joblib.load("models/feature_names.pkl")
print("Enter profile details:\n")
followers = int(input("Followers: "))
following = int(input("Following: "))
posts = int(input("Posts: "))
avg_likes = int(input("Average likes: "))
avg_comments = int(input("Average Comments: "))
has_bio = int(input("Has Bio (1 = Yes, 0 = No): "))
has_profile_pic = int(input("Has profile Pic (1 = Yes, 0 = No): "))
is_private = int(input("Is private (1 = Yes, 0 = No): "))
account_age_days = int(input("Account Age (in days): "))
bio_text = input("Bio text: ")
df = pd.DataFrame([{
"followers": followers,
"following": following,
"posts": posts,
"avg_likes": avg_likes,
"avg_comments": avg_comments,
"has_bio": has_bio,
"has_profile_pic": has_profile_pic,
"is_private": is_private,
"account_age_days": account_age_days,
"bio_text": bio_text
}])
df["bio_length"] = df["bio_text"].fillna("").astype(str).apply(len)
df["bio_word_count"] = df["bio_text"].fillna("").astype(str).apply(lambda x: len(x.split()))
df["bio_has_link"] = df["bio_text"].fillna("").astype(str).apply(lambda x: 1 if "http" in x.lower() else 0)
df["bio_has_digits"] = df["bio_text"].fillna("").astype(str).apply(lambda x: 1 if any(c.isdigit() for c in x) else 0)
df = df.drop("bio_text", axis = 1)
df = df.reindex(columns=feature_names, fill_value=0)
prediction = model.predict(df)[0]
if prediction == 1:
	print("fake profile detected")
else:
	print("this looks like a real profile")
