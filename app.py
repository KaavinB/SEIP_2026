import gzip
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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


@st.cache_data
def load_data(filename="Watches.txt.gz"):
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


st.title("Hello World!")
st.caption("Amazon watch-review exploratory dashboard")

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

# ---------- headline metrics ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(d):,}")
c2.metric("Products", f"{d['productId'].nunique():,}")
c3.metric("Mean score", f"{d['score'].mean():.2f}")
c4.metric("Price known", f"{d['price'].notna().mean():.0%}")

# ---------- score distribution ----------
st.subheader("Score distribution")
fig, ax = plt.subplots(figsize=(7, 3))
d["score"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
ax.set_xlabel("star rating"); ax.set_ylabel("count")
st.pyplot(fig)

# ---------- reviews over time ----------
st.subheader("Reviews per quarter")
fig, ax = plt.subplots(figsize=(9, 3))
d.set_index("time").resample("QE").size().plot(ax=ax)
ax.set_ylabel("count")
st.pyplot(fig)

# ---------- mean rating over time by brand ----------
st.subheader("Mean rating over time by brand (years with ≥10 reviews)")
yearly = d.groupby(["year", "brand"])["score"].agg(["mean", "size"]).reset_index()
yearly = yearly[yearly["size"] >= 10]
fig, ax = plt.subplots(figsize=(9, 4))
for b in (picked or brands_all[:6]):
    sub = yearly[yearly["brand"] == b]
    if len(sub):
        ax.plot(sub["year"], sub["mean"], marker="o", ms=3, label=b)
ax.set_ylim(1, 5); ax.set_ylabel("mean score"); ax.legend(ncol=2, fontsize=8)
st.pyplot(fig)

# ---------- price tiers ----------
st.subheader("Mean rating by price tier")
priced = d[d["price"].notna()].copy()
if len(priced) > 20:
    priced["tier"] = pd.qcut(priced["price"], 6, duplicates="drop")
    g = priced.groupby("tier", observed=True)["score"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(g)), g["mean"], color="teal")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(i) for i in g.index], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(3.5, 5); ax.set_ylabel("mean score")
    st.pyplot(fig)
else:
    st.info("Not enough priced rows in the current filter to bin into tiers.")

# ---------- top products table ----------
st.subheader("Most-reviewed products (current filter)")
top = (d.groupby("productId")
         .agg(n=("score", "size"), mean_score=("score", "mean"), title=("title", "first"))
         .sort_values("n", ascending=False).head(15).round(2))
st.dataframe(top, use_container_width=True)
