import sys

import pandas as pd

# A donor with no gift in this many days counts as inactive (lapsed).
INACTIVE_DAYS = 365


def load_donors(path):
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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/donors.csv"
    df = load_donors(path)

    total_donors = len(df)
    total_amount = df["total_donation"].sum()
    total_gifts = df["donation_count"].sum()

    avg_per_donor = total_amount / total_donors
    avg_per_gift = total_amount / total_gifts

    top = df.loc[df["total_donation"].idxmax()]

    inactive = df[df["days_since_last_donation"] > INACTIVE_DAYS]
    inactive = inactive.sort_values("total_donation", ascending=False)

    print(f"Total Donors:      {total_donors}")
    print(f"Total Donations:   ${total_amount:,.0f}  ({total_gifts:,} gifts)")
    print(f"Average Donation:  ${avg_per_donor:,.2f} per donor  |  ${avg_per_gift:,.2f} per gift")
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


if __name__ == "__main__":
    main()
