import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Amazon Watch Reviews EDA", layout="wide")


# ---------- data loading ----------
def parse(filename):
    f = gzip.open(filename, "rt")
    entry = {}
    for l in f:
        l = l.strip()
        colon = l.find(":")
        if colon == -1:
            yield entry
            entry = {}
            continue
        entry[l[:colon]] = l[colon + 2:]
    yield entry


DATA_PATH = Path(__file__).parent / "Watches.txt.gz"


@st.cache_data
def load_data(filename=DATA_PATH):
    if not Path(filename).exists():
        st.error(
            f"Data file not found at `{filename}`.\n\n"
            "Make sure `Watches.txt.gz` is committed to the repo, in the same "
            "folder as `app.py`."
        )
        st.stop()
    df = pd.DataFrame(list(parse(filename)))
    df = df[df["product/productId"].notna()]
    df.columns = [c.split("/")[-1] for c in df.columns]

    df["score"] = df["score"].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s")
    df["price"] = pd.to_numeric(df["price"].replace("unknown", np.nan), errors="coerce")

    hv = df["helpfulness"].str.split("/", expand=True).astype(int)
    df["help_votes"], df["help_total"] = hv[0], hv[1]
    df["help_ratio"] = np.where(df["help_total"] > 0, df["help_votes"] / df["help_total"], np.nan)

    df["text_len"] = df["text"].str.len()
    df["year"] = df["time"].dt.year
    df["brand"] = df["title"].str.split().str[0].str.title()
    return df


st.title("Amazon watch-review exploratory dashboard")

df = load_data()

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
brands_all = df["brand"].value_counts().head(20).index.tolist()
picked = st.sidebar.multiselect("Brands", brands_all, default=brands_all[:6])
yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
yr = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

mask = df["year"].between(*yr)
if picked:
    mask &= df["brand"].isin(picked)
d = df[mask]
active_brands = picked or brands_all[:6]

# ---------- headline metrics ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(d):,}")
c2.metric("Products", f"{d['productId'].nunique():,}")
c3.metric("Mean score", f"{d['score'].mean():.2f}")
c4.metric("Price known", f"{d['price'].notna().mean():.0%}")

if d.empty:
    st.warning("No reviews match the current filters.")
    st.stop()

# ================= 1. score distribution =================
st.subheader("1 · Score distribution")
fig, ax = plt.subplots(figsize=(7, 3))
d["score"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
ax.set_xlabel("star rating"); ax.set_ylabel("count")
st.pyplot(fig)

# ================= 2. reviews per quarter =================
st.subheader("2 · Reviews per quarter")
fig, ax = plt.subplots(figsize=(9, 3))
d.set_index("time").resample("QE").size().plot(ax=ax)
ax.set_ylabel("count")
st.pyplot(fig)

# ================= 3. mean rating over time by brand =================
st.subheader("3 · Mean rating over time by brand (years with ≥10 reviews)")
yearly = d.groupby(["year", "brand"])["score"].agg(["mean", "size"]).reset_index()
yearly = yearly[yearly["size"] >= 10]
fig, ax = plt.subplots(figsize=(9, 4))
for b in active_brands:
    sub = yearly[yearly["brand"] == b]
    if len(sub):
        ax.plot(sub["year"], sub["mean"], marker="o", ms=3, label=b)
ax.set_ylim(1, 5); ax.set_ylabel("mean score"); ax.legend(ncol=2, fontsize=8)
st.pyplot(fig)

# ================= 4. score distribution by brand (ridgeline-ish) =================
st.subheader("4 · Score distribution by brand")
bplot = [b for b in active_brands if (d["brand"] == b).sum() > 0][:8]
if bplot:
    fig, axes = plt.subplots(len(bplot), 1, figsize=(8, 1.1 * len(bplot)), sharex=True)
    if len(bplot) == 1:
        axes = [axes]
    for ax, b in zip(axes, bplot):
        s = d.loc[d["brand"] == b, "score"]
        ax.hist(s, bins=np.arange(0.5, 6, 1), density=True, color="steelblue", edgecolor="white")
        ax.set_yticks([])
        ax.set_ylabel(f"{b}\n(n={len(s)})", rotation=0, ha="right", va="center", fontsize=8)
        ax.axvline(s.mean(), color="crimson", lw=1.5)
    axes[-1].set_xlabel("star rating  (red line = brand mean)")
    st.pyplot(fig)
else:
    st.info("No brands selected to break down.")

# ================= 5. helpfulness vs rating (hexbin / J-curve) =================
st.subheader("5 · Helpfulness vs. rating (reviews with ≥5 votes)")
v = d[d["help_total"] >= 5]
if len(v) > 10:
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(v["score"], v["help_ratio"], gridsize=20, cmap="viridis", mincnt=1)
    m = v.groupby("score")["help_ratio"].mean()
    ax.plot(m.index, m.values, "r-o", lw=2, label="mean per star")
    ax.set_xlabel("star rating"); ax.set_ylabel("helpfulness ratio")
    ax.legend(); fig.colorbar(hb, label="count")
    st.pyplot(fig)
else:
    st.info(f"Only {len(v)} reviews with ≥5 votes in this filter — not enough to plot.")

# ================= 6. review length by rating (violins) =================
st.subheader("6 · Review length distribution by rating")
data = [np.log10(d.loc[d["score"] == s, "text_len"].dropna() + 1) for s in [1, 2, 3, 4, 5]]
data_present = [(s, arr) for s, arr in zip([1, 2, 3, 4, 5], data) if len(arr) > 1]
if data_present:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.violinplot([arr for _, arr in data_present], showmedians=True)
    ax.set_xticks(range(1, len(data_present) + 1))
    ax.set_xticklabels([s for s, _ in data_present])
    ax.set_xlabel("star rating"); ax.set_ylabel("log10(review length + 1)")
    st.pyplot(fig)
else:
    st.info("Not enough text to show length distributions.")

# ================= 7. price tiers =================
st.subheader("7 · Mean rating by price tier")
priced = d[d["price"].notna()].copy()
if len(priced) > 20:
    priced["tier"] = pd.qcut(priced["price"], 6, duplicates="drop")
    g = priced.groupby("tier", observed=True)["score"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(g)), g["mean"], color="teal")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(i) for i in g.index], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(3.5, 5); ax.set_ylabel("mean score")
    for i, (mn, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(i, mn + 0.01, f"n={n}", ha="center", fontsize=8)
    st.pyplot(fig)
else:
    st.info("Not enough priced rows in the current filter to bin into tiers.")

# ================= 8 & 9. TF-IDF distinctive terms (gated) =================
st.subheader("8 · Distinctive words by rating (TF-IDF)")
st.caption("Heavier to compute — runs on the current filter when enabled.")
if st.checkbox("Run text analysis"):
    docs = d.groupby("score")["text"].apply(lambda s: " ".join(s.dropna()))
    if len(docs) >= 2:
        tfidf = TfidfVectorizer(max_features=3000, stop_words="english", min_df=5)
        X = tfidf.fit_transform(docs)
        terms = np.array(tfidf.get_feature_names_out())

        lines = []
        for i, s in enumerate(docs.index):
            topi = X[i].toarray().ravel().argsort()[::-1][:12]
            lines.append(f"**{int(s)}★** — " + ", ".join(terms[topi]))
        st.markdown("\n\n".join(lines))

        # heatmap of union of each rating's top terms
        top_terms = set()
        for i in range(X.shape[0]):
            top_terms.update(terms[X[i].toarray().ravel().argsort()[::-1][:8]])
        top_terms = sorted(top_terms)
        idx = [list(terms).index(t) for t in top_terms]
        mat = pd.DataFrame(
            X[:, idx].toarray(),
            index=[f"{int(s)}★" for s in docs.index],
            columns=top_terms,
        ).T
        st.subheader("9 · TF-IDF of distinctive terms by rating")
        fig, ax = plt.subplots(figsize=(6, max(3, 0.32 * len(top_terms))))
        im = ax.imshow(mat.values, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(top_terms))); ax.set_yticklabels(top_terms, fontsize=8)
        ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns)
        fig.colorbar(im, shrink=0.5)
        st.pyplot(fig)
    else:
        st.info("Need at least two rating groups with text for TF-IDF.")

# ================= 10. top products table =================
st.subheader("10 · Most-reviewed products (current filter)")
top = (d.groupby("productId")
         .agg(n=("score", "size"), mean_score=("score", "mean"), title=("title", "first"))
         .sort_values("n", ascending=False).head(15).round(2))
st.dataframe(top, use_container_width=True)
