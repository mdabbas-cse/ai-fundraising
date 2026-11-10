import sys

import numpy as np

from donor_kpis import INACTIVE_DAYS, load_donors

# Lifetime giving at or above this amount makes a donor High Value.
HIGH_VALUE_MIN = 1000

SEGMENTS = ["High Value Donor", "Regular Donor", "New Donor", "Inactive Donor"]


def segment(df):
    """Assign one segment per donor. Rules are checked top to bottom; first match wins."""
    conditions = [
        df["days_since_last_donation"] > INACTIVE_DAYS,  # no gift in over a year
        df["total_donation"] >= HIGH_VALUE_MIN,          # big lifetime giving
        df["donation_count"] == 1,                       # only one gift so far
    ]
    choices = ["Inactive Donor", "High Value Donor", "New Donor"]
    return np.select(conditions, choices, default="Regular Donor")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/donors.csv"
    out_path = "data/donor_segments.csv"

    df = load_donors(path)
    df["segment"] = segment(df)
    df.to_csv(out_path, index=False)

    for donor_id, seg in zip(df["donor_id"], df["segment"]):
        print(f"{donor_id} → {seg}")

    counts = df["segment"].value_counts().reindex(SEGMENTS, fill_value=0)
    print("\nSegment summary:")
    for seg, n in counts.items():
        value = df.loc[df["segment"] == seg, "total_donation"].sum()
        print(f"  {seg:<17} {n:>3} donors  ${value:>10,.0f}")

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
