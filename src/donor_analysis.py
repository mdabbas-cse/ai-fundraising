"""Donor analysis: synthetic data, KPI report, and rule-based segmentation.

Usage:
    python src/donor_analysis.py                       # report on data/donors.csv
    python src/donor_analysis.py path/to/donors.csv    # report on another file
    python src/donor_analysis.py --generate            # rebuild data/donors.csv first
    python src/donor_analysis.py --out data/donor_segments.csv   # also save segments
"""

import argparse

import numpy as np
import pandas as pd

DEFAULT_PATH = "data/donors.csv"

# A donor with no gift in this many days counts as inactive (lapsed).
INACTIVE_DAYS = 365

# Lifetime giving at or above this amount makes a donor High Value.
HIGH_VALUE_MIN = 1000

SEGMENTS = ["High Value Donor", "Regular Donor", "New Donor", "Inactive Donor"]

# First donors match a small hand-written reference table:
# (total_donation, donation_count, days_since_last_donation)
REFERENCE_DONORS = [
    (1000, 10, 10),   # D001
    (50, 1, 5),       # D002
    (500, 5, 200),    # D003
    (20, 1, 400),     # D004
    (2000, 20, 15),   # D005
    (200, 4, 30),     # D006
]


def generate_donors(n=100, seed=42):
    """Build a reproducible synthetic donor dataset. No real people are involved."""
    rng = np.random.RandomState(seed)

    # Number of gifts per donor: most give a few times, a few give often.
    donation_count = np.clip(rng.geometric(p=0.15, size=n), 1, 30)

    # Average gift size (~$50 typical, long tail), so total grows with count.
    avg_gift = np.clip(rng.lognormal(mean=np.log(50), sigma=0.6, size=n), 10, 300)
    total_donation = np.maximum(10, np.round(donation_count * avg_gift / 10) * 10).astype(int)

    # Frequent donors tend to have given more recently.
    days_since_last_donation = np.clip(
        rng.exponential(scale=300 / np.sqrt(donation_count)), 1, 500
    ).astype(int)

    df = pd.DataFrame({
        "donor_id": [f"D{i:03d}" for i in range(1, n + 1)],
        "total_donation": total_donation,
        "donation_count": donation_count,
        "days_since_last_donation": days_since_last_donation,
    })
    cols = ["total_donation", "donation_count", "days_since_last_donation"]
    df.loc[: len(REFERENCE_DONORS) - 1, cols] = REFERENCE_DONORS
    return df


def load_donors(path=DEFAULT_PATH):
    """Load donors.csv in either the raw or the display format."""
    df = pd.read_csv(path)

    # Display format: Donor, Total Donation ($1,000), Donations, Last Donation (10 days ago)
    if "Donor" in df.columns:
        df = pd.DataFrame({
            "donor_id": df["Donor"],
            "total_donation": df["Total Donation"].str.replace(r"[$,]", "", regex=True).astype(float),
            "donation_count": df["Donations"].astype(int),
            "days_since_last_donation": df["Last Donation"].str.extract(r"(\d+)")[0].astype(int),
        })

    return df


def segment(df):
    """Assign one segment per donor. Rules are checked top to bottom; first match wins."""
    conditions = [
        df["days_since_last_donation"] > INACTIVE_DAYS,  # no gift in over a year
        df["total_donation"] >= HIGH_VALUE_MIN,          # big lifetime giving
        df["donation_count"] == 1,                       # only one gift so far
    ]
    choices = ["Inactive Donor", "High Value Donor", "New Donor"]
    return np.select(conditions, choices, default="Regular Donor")


def print_kpis(df):
    total_donors = len(df)
    total_amount = df["total_donation"].sum()
    total_gifts = df["donation_count"].sum()

    top = df.loc[df["total_donation"].idxmax()]

    inactive = df[df["days_since_last_donation"] > INACTIVE_DAYS]
    inactive = inactive.sort_values("total_donation", ascending=False)

    print(f"Total Donors:      {total_donors}")
    print(f"Total Donations:   ${total_amount:,.0f}  ({total_gifts:,} gifts)")
    print(
        f"Average Donation:  ${total_amount / total_donors:,.2f} per donor"
        f"  |  ${total_amount / total_gifts:,.2f} per gift"
    )
    print(
        f"Highest Donor:     {top['donor_id']}  ${top['total_donation']:,.0f}"
        f"  ({top['donation_count']} gifts, last {top['days_since_last_donation']} days ago)"
    )
    print(
        f"Inactive Donors:   {len(inactive)} of {total_donors}"
        f" ({len(inactive) / total_donors:.0%}) with no gift in {INACTIVE_DAYS}+ days"
    )

    if not inactive.empty:
        print(f"\nInactive donors (lost value ${inactive['total_donation'].sum():,.0f}):")
        print(inactive.to_string(index=False))


def print_segments(df):
    counts = df["segment"].value_counts().reindex(SEGMENTS, fill_value=0)
    print("\nSegment summary:")
    for seg, n in counts.items():
        value = df.loc[df["segment"] == seg, "total_donation"].sum()
        print(f"  {seg:<17} {n:>3} donors  ${value:>10,.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="donor CSV file")
    parser.add_argument("--generate", action="store_true",
                        help="write a fresh synthetic dataset to PATH before the report")
    parser.add_argument("--out", help="save donors with a segment column to this CSV")
    args = parser.parse_args()

    if args.generate:
        generate_donors().to_csv(args.path, index=False)
        print(f"Generated {args.path}\n")

    df = load_donors(args.path)
    print_kpis(df)

    df["segment"] = segment(df)
    print_segments(df)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
