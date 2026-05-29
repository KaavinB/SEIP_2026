import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Watch Reviews EDA", page_icon="⌚", layout="wide")

# ---------- global plot style ----------
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.size": 9,
})

ACCENT = "#2b6cb0"
PALETTE = plt.cm.tab10.colors


def show(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


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


# ---------- header ----------
st.title("⌚ Amazon Watch Reviews")
st.caption("Exploratory analysis of 68k watch reviews (1998–2013). Use the sidebar to filter by brand and year.")

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

# stable color per brand, shared across every chart
brand_color = {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(active_brands)}

# ---------- headline metrics ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Reviews", f"{len(d):,}")
m2.metric("Products", f"{d['productId'].nunique():,}")
m3.metric("Mean score", f"{d['score'].mean():.2f}")
m4.metric("Price known", f"{d['price'].notna().mean():.0%}")

if d.empty:
    st.warning("No reviews match the current filters.")
    st.stop()

tab_overview, tab_brands, tab_quality, tab_price, tab_text, tab_products = st.tabs(
    ["Overview", "Brands", "Helpfulness & length", "Price", "Text", "Products"]
)

# ===================== OVERVIEW =====================
with tab_overview:
    left, right = st.columns(2)

    with left:
        st.markdown("**Score distribution**")
        counts = d["score"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar([f"{int(s)}★" for s in counts.index], counts.values, color=ACCENT)
        ax.set_ylabel("reviews")
        for x, v in enumerate(counts.values):
            ax.text(x, v, f"{v/counts.sum():.0%}", ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.12)
        show(fig)
        st.caption("Heavily skewed to 5★ — typical of product-review data.")

    with right:
        st.markdown("**Reviews per quarter**")
        ts = d.set_index("time").resample("QE").size()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.fill_between(ts.index, ts.values, color=ACCENT, alpha=0.18)
        ax.plot(ts.index, ts.values, color=ACCENT, lw=1.5)
        ax.set_ylabel("reviews")
        show(fig)
        st.caption("Review volume over time (quarterly).")

# ===================== BRANDS =====================
with tab_brands:
    st.markdown("**Mean rating over time** — years with ≥10 reviews per brand")
    yearly = d.groupby(["year", "brand"])["score"].agg(["mean", "size"]).reset_index()
    yearly = yearly[yearly["size"] >= 10]
    fig, ax = plt.subplots(figsize=(10, 4))
    for b in active_brands:
        sub = yearly[yearly["brand"] == b]
        if len(sub):
            ax.plot(sub["year"], sub["mean"], marker="o", ms=4, lw=1.8,
                    color=brand_color[b], label=b)
    ax.set_ylim(1, 5); ax.set_ylabel("mean score")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.35))
    show(fig)

    st.markdown("**Score distribution by brand** — red line marks each brand's mean")
    bplot = [b for b in active_brands if (d["brand"] == b).sum() > 0][:8]
    if bplot:
        fig, axes = plt.subplots(len(bplot), 1, figsize=(8, 1.0 * len(bplot)), sharex=True)
        if len(bplot) == 1:
            axes = [axes]
        for ax, b in zip(axes, bplot):
            s = d.loc[d["brand"] == b, "score"]
            ax.hist(s, bins=np.arange(0.5, 6, 1), density=True,
                    color=brand_color[b], edgecolor="white")
            ax.set_yticks([]); ax.grid(False)
            ax.set_ylabel(f"{b}\nn={len(s):,}", rotation=0, ha="right", va="center", fontsize=8)
            ax.axvline(s.mean(), color="crimson", lw=1.5)
        axes[-1].set_xlabel("star rating")
        axes[-1].set_xticks([1, 2, 3, 4, 5])
        show(fig)
    else:
        st.info("No brands selected to break down.")

# ===================== HELPFULNESS & LENGTH =====================
with tab_quality:
    left, right = st.columns(2)

    with left:
        st.markdown("**Helpfulness vs. rating** — reviews with ≥5 votes")
        v = d[d["help_total"] >= 5]
        if len(v) > 10:
            fig, ax = plt.subplots(figsize=(5, 4))
            hb = ax.hexbin(v["score"], v["help_ratio"], gridsize=18, cmap="Blues", mincnt=1)
            m = v.groupby("score")["help_ratio"].mean()
            ax.plot(m.index, m.values, "o-", color="crimson", lw=2, label="mean per star")
            ax.set_xlabel("star rating"); ax.set_ylabel("helpfulness ratio")
            ax.legend(fontsize=8, frameon=False)
            fig.colorbar(hb, label="reviews")
            show(fig)
            st.caption("The 'J-curve': extreme reviews tend to be voted most helpful.")
        else:
            st.info(f"Only {len(v)} reviews with ≥5 votes here — not enough to plot.")

    with right:
        st.markdown("**Review length by rating**")
        groups = [(s, np.log10(d.loc[d["score"] == s, "text_len"].dropna() + 1))
                  for s in [1, 2, 3, 4, 5]]
        groups = [(s, a) for s, a in groups if len(a) > 1]
        if groups:
            fig, ax = plt.subplots(figsize=(5, 4))
            parts = ax.violinplot([a for _, a in groups], showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(ACCENT); pc.set_alpha(0.5)
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([f"{s}★" for s, _ in groups])
            ax.set_ylabel("log₁₀(length + 1)")
            show(fig)
            st.caption("Critical reviews tend to run longer than praise.")
        else:
            st.info("Not enough text to show length distributions.")

# ===================== PRICE =====================
with tab_price:
    st.markdown("**Mean rating by price tier** — known prices only")
    priced = d[d["price"].notna()].copy()
    if len(priced) > 20:
        priced["tier"] = pd.qcut(priced["price"], 6, duplicates="drop")
        g = priced.groupby("tier", observed=True)["score"].agg(["mean", "size"])
        labels = [f"${int(iv.left)}–{int(iv.right)}" for iv in g.index]
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(range(len(g)), g["mean"], color=ACCENT)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(3.5, 5); ax.set_ylabel("mean score")
        for i, (mn, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(i, mn + 0.01, f"{mn:.2f}\nn={n}", ha="center", va="bottom", fontsize=7)
        show(fig)
        st.caption(f"{len(priced):,} of {len(d):,} reviews have a known price.")
    else:
        st.info("Not enough priced rows in the current filter to bin into tiers.")

# ===================== TEXT =====================
with tab_text:
    st.markdown("**Distinctive words by rating (TF-IDF)**")
    st.caption("Heavier to compute — runs on the current filter when enabled.")
    if st.checkbox("Run text analysis"):
        docs = d.groupby("score")["text"].apply(lambda s: " ".join(s.dropna()))
        if len(docs) >= 2:
            tfidf = TfidfVectorizer(max_features=3000, stop_words="english", min_df=5)
            X = tfidf.fit_transform(docs)
            terms = np.array(tfidf.get_feature_names_out())

            cols = st.columns(len(docs))
            for col, i, s in zip(cols, range(len(docs)), docs.index):
                topi = X[i].toarray().ravel().argsort()[::-1][:12]
                col.markdown(f"**{int(s)}★**")
                col.markdown("\n".join(f"- {t}" for t in terms[topi]))

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
            st.markdown("**TF-IDF heatmap of distinctive terms**")
            fig, ax = plt.subplots(figsize=(6, max(3, 0.30 * len(top_terms))))
            im = ax.imshow(mat.values, aspect="auto", cmap="magma")
            ax.set_yticks(range(len(top_terms))); ax.set_yticklabels(top_terms, fontsize=8)
            ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns)
            ax.grid(False)
            fig.colorbar(im, shrink=0.5, label="tf-idf")
            show(fig)
            st.caption("Note: much negative-review vocabulary is about shipping/returns, "
                       "not the watches themselves.")
        else:
            st.info("Need at least two rating groups with text for TF-IDF.")

# ===================== PRODUCTS =====================
with tab_products:
    st.markdown("**Most-reviewed products** (current filter)")
    top = (d.groupby("productId")
             .agg(reviews=("score", "size"), mean_score=("score", "mean"),
                  title=("title", "first"))
             .sort_values("reviews", ascending=False).head(20).round(2))
    st.dataframe(
        top[["title", "reviews", "mean_score"]],
        use_container_width=True,
        column_config={
            "title": "Product",
            "reviews": st.column_config.NumberColumn("Reviews"),
            "mean_score": st.column_config.NumberColumn("Mean ★", format="%.2f"),
        },
    )
