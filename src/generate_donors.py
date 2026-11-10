import pandas as pd
import numpy as np

np.random.seed(42)

total_donors = 100

data = {
    "donor_id": [
        f"D{i:03d}"
        for i in range(1, total_donors + 1)
    ],
    "total_donation": np.random.randint(
        20, 3000, total_donors
    ),
    "donation_count": np.random.randint(
        1, 30, total_donors
    ),
    "days_since_last_donation": np.random.randint(
        1, 500, total_donors
    )
}

df = pd.DataFrame(data)

df.to_csv("data/donors.csv", index=False)

print(df.head())
print("Total donors:", len(df))
