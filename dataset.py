import numpy as np
import pandas as pd
rng = np.random.default_rng(42)
N = 2000
followers = rng.integers(0, 80000, N)
following = rng.integers(0, 5000, N)
posts = rng.integers(0, 2000, N)
avg_likes = (followers * rng.uniform(0, 0.06, N)).astype(int)
avg_comments = (avg_likes * rng.uniform(0, 0.25, N)).astype(int)
has_bio = rng.integers(0, 2, N)
has_profile_pic = rng.integers(0, 2, N)
is_private = rng.integers(0, 2, N)
account_age_days = rng.integers(1, 4000, N)
bio_text = rng.choice(["", "photography lover", "student", "enterpreneur", "contact: email@example.com"], N)
engagement_rate = (avg_likes + avg_comments) / np.maximum(followers, 1)
ff_ratio = followers / (following + 1)
followers_per_day = followers / np.maximum(account_age_days, 1)
score = (2.5 * (engagement_rate < 0.01).astype(int) + 1.5 * (ff_ratio < 0.02).astype(int) + 1.2 * (followers_per_day > 300).astype(int) + 1.0 * (has_bio == 0).astype(int) + 1.0 * (has_profile_pic == 0).astype(int) + 0.7 * (posts < 5).astype(int)) + rng.normal(0, 0.7, N)
label = (score > 2.3).astype(int)
df = pd.DataFrame({
"followers": followers,
"following": following,
"posts": posts,
"avg_likes": avg_likes,
"avg_comments": avg_comments,
"has_bio": has_bio,
"has_profile_pic": has_profile_pic,
"is_private": is_private,
"account_age_days": account_age_days,
"bio_text": bio_text,
"label": label
})
df.to_csv("data/profiles.csv", index = False)
print("saved synthetic dataset to data/profiles.csv")