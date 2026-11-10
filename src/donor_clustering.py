"""Donor clustering: group donors with K-Means on recency, frequency, and monetary value.

Usage:
    python src/donor_clustering.py                      # 4 clusters on data/donors.csv
    python src/donor_clustering.py --k 3                # choose another cluster count
    python src/donor_clustering.py --out data/donor_clusters.csv   # also save labels
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from donor_analysis import DEFAULT_PATH, SEGMENTS, load_donors, segment

# Recency, frequency, monetary (RFM).
FEATURES = ["days_since_last_donation", "donation_count", "total_donation"]

# Four clusters lines up with the four rule-based segments, which makes the two
# easy to compare. On the synthetic data, silhouette scores for k = 2 to 5 are
# close together (about 0.33 to 0.39), so no k is a clear statistical winner.
DEFAULT_K = 4
K_RANGE = range(2, 9)
SEED = 42


def rfm_features(df):
    """Log-scale the skewed RFM columns, then standardize so each counts equally."""
    return StandardScaler().fit_transform(np.log1p(df[FEATURES]))


def fit_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)


def score_k(X, k_range=K_RANGE):
    """Inertia (elbow method) and silhouette score for each candidate k."""
    rows = []
    for k in k_range:
        model = fit_kmeans(X, k)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, model.labels_),
        })
    return pd.DataFrame(rows).set_index("k")


def cluster(df, k=DEFAULT_K):
    """Return cluster labels, numbered so cluster 1 has the highest median giving."""
    labels = fit_kmeans(rfm_features(df), k).labels_
    median_giving = df.groupby(labels)["total_donation"].median()
    rank = median_giving.rank(ascending=False, method="first").astype(int)
    return pd.Series(labels, index=df.index).map(rank)


def profile(df):
    """One row per cluster: size, typical RFM values, and share of total giving."""
    groups = df.groupby("cluster")
    return pd.DataFrame({
        "donors": groups.size(),
        "median_days_since_last": groups["days_since_last_donation"].median(),
        "median_gifts": groups["donation_count"].median(),
        "median_total": groups["total_donation"].median(),
        "share_of_giving": groups["total_donation"].sum() / df["total_donation"].sum(),
    })


def compare_with_segments(df):
    """Donor counts per cluster and rule-based segment."""
    table = pd.crosstab(df["cluster"], df["segment"])
    return table.reindex(columns=SEGMENTS, fill_value=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="donor CSV file")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="number of clusters")
    parser.add_argument("--out", help="save donors with cluster and segment columns to this CSV")
    args = parser.parse_args()

    df = load_donors(args.path)

    print("Choosing k (lower inertia is tighter; higher silhouette is better separated):")
    print(score_k(rfm_features(df)).round(3).to_string())

    df["cluster"] = cluster(df, args.k)
    df["segment"] = segment(df)

    print(f"\nCluster profiles (k={args.k}, cluster 1 = highest median giving):")
    print(profile(df).to_string(formatters={
        "median_days_since_last": "{:.0f}".format,
        "median_gifts": "{:.0f}".format,
        "median_total": "${:,.0f}".format,
        "share_of_giving": "{:.0%}".format,
    }))

    print("\nClusters vs rule-based segments (donor counts):")
    print(compare_with_segments(df).to_string())

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
