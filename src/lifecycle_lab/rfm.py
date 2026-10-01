"""RFM scoring, rule-based segments and k-means clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEGMENT_ORDER = ["Champions", "Loyal", "Promising", "Needs attention", "At risk", "Hibernating"]


def _quintile(series: pd.Series, ascending: bool = True) -> pd.Series:
    """1 to 5 score by rank (ties broken by order), so every quintile is populated."""
    ranks = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def rfm_scores(feats: pd.DataFrame) -> pd.DataFrame:
    out = feats.copy()
    out["r_score"] = _quintile(out["recency_days"], ascending=False)  # recent buyers score 5
    out["f_score"] = _quintile(out["frequency"])
    out["m_score"] = _quintile(out["monetary"])
    out["rfm_segment"] = np.select(
        [
            (out.r_score >= 4) & (out.f_score >= 4),
            (out.r_score >= 3) & (out.f_score >= 3),
            (out.r_score >= 4) & (out.f_score <= 2),
            (out.r_score <= 2) & (out.f_score >= 3),
            (out.r_score <= 2) & (out.f_score <= 2),
        ],
        ["Champions", "Loyal", "Promising", "At risk", "Hibernating"],
        default="Needs attention",
    )
    return out


def segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Size, value and next-90-day repurchase rate for each rule-based segment."""
    g = scored.groupby("rfm_segment")
    out = pd.DataFrame(
        {
            "customers": g.size(),
            "revenue_to_date": g["monetary"].sum(),
            "avg_orders": g["frequency"].mean(),
            "avg_recency_days": g["recency_days"].mean(),
            "repurchase_rate_90d": g["repurchased_90d"].mean(),
            "revenue_next_6m": g["revenue_6m"].sum(),
        }
    ).reindex(SEGMENT_ORDER).dropna(how="all")
    out["customer_share"] = out["customers"] / out["customers"].sum()
    out["revenue_share"] = out["revenue_to_date"] / out["revenue_to_date"].sum()
    return out.reset_index().rename(columns={"rfm_segment": "segment"})


CLUSTER_FEATURES = ["recency_days", "frequency", "monetary"]


def _matrix(df: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(np.log1p(df[CLUSTER_FEATURES].to_numpy(dtype=float)))


def choose_k(df: pd.DataFrame, k_range=range(3, 8), seed: int = 0, sample: int = 4000) -> pd.DataFrame:
    x = _matrix(df)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(sample, len(x)), replace=False)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(x[idx], km.labels_[idx])})
    return pd.DataFrame(rows)


def cluster_customers(df: pd.DataFrame, k: int, seed: int = 0) -> tuple[pd.DataFrame, float]:
    """Cluster on log recency, frequency and monetary value; also report agreement with the hidden segments."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(_matrix(df))
    out = df.copy()
    out["cluster"] = km.labels_
    ari = adjusted_rand_score(out["true_segment"], out["cluster"])
    return out, float(ari)


def cluster_profiles(clustered: pd.DataFrame) -> pd.DataFrame:
    g = clustered.groupby("cluster")
    prof = pd.DataFrame(
        {
            "customers": g.size(),
            "avg_recency_days": g["recency_days"].mean(),
            "avg_frequency": g["frequency"].mean(),
            "avg_monetary": g["monetary"].mean(),
            "dominant_true_segment": g["true_segment"].agg(lambda s: s.value_counts().idxmax()),
            "purity": g["true_segment"].agg(lambda s: s.value_counts(normalize=True).max()),
        }
    )
    return prof.reset_index()
