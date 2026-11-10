import numpy as np
import pandas as pd

np.random.seed(42)

total_donors = 100

# Number of gifts per donor: most give a few times, a few give often.
donation_count = np.clip(np.random.geometric(p=0.15, size=total_donors), 1, 30)

# Average gift size (~$50 typical, long tail), so total grows with count.
avg_gift = np.clip(np.random.lognormal(mean=np.log(50), sigma=0.6, size=total_donors), 10, 300)
total_donation = np.maximum(10, np.round(donation_count * avg_gift / 10) * 10).astype(int)

# Frequent donors tend to have given more recently.
days_since_last_donation = np.clip(
    np.random.exponential(scale=300 / np.sqrt(donation_count)), 1, 500
).astype(int)

df = pd.DataFrame({
    "donor_id": [f"D{i:03d}" for i in range(1, total_donors + 1)],
    "total_donation": total_donation,
    "donation_count": donation_count,
    "days_since_last_donation": days_since_last_donation,
})

# First rows match the reference table: (total_donation, donation_count, days_since_last_donation)
reference = [
    (1000, 10, 10),   # D001
    (50, 1, 5),       # D002
    (500, 5, 200),    # D003
    (20, 1, 400),     # D004
    (2000, 20, 15),   # D005
    (200, 4, 30),     # D006
]
df.loc[: len(reference) - 1, ["total_donation", "donation_count", "days_since_last_donation"]] = reference

df.to_csv("data/donors.csv", index=False)

print(df.head(10))
print("Total donors:", len(df))
