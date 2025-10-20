# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "matplotlib==3.10.6",
#     "numpy==2.2.6",
#     "pandas==2.3.2",
#     "pyarrow==21.0.0",
#     "scikit-learn==1.7.2",
#     "umap-learn==0.5.9.post2",
# ]
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    return mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 2D-Embedding der Antworten (SVD → t-SNE/UMAP)

    **Ziel:** Aus ausgewählten Fragen (Gruppe) wird ein kategoriales Feature-Set gebaut, per **One-Hot** kodiert, dann via **Truncated SVD** auf \(k\) Dimensionen reduziert und schließlich mit **t-SNE** oder **UMAP** in 2D projiziert.  
    Die resultierenden **Komponenten (`comp1`, `comp2`) werden als Spalten** zurück in den Original-DataFrame geschrieben.  
    Die **Brush-Auswahl** zeigt am Ende **alle Spalten** der selektierten Zeilen.
    """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/Antworten.csv")
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Gruppierung der Fragen (F-Spalten)

    Wir parsen alle Spalten, die mit `F…` beginnen, und gruppieren sie in Themenblöcke.  
    Das Dropdown **“Fragegruppe für Embedding”** steuert, welche F-Spalten für das Embedding verwendet werden.
    """
    )
    return


@app.cell
def _(df):
    import re

    def parse_f(col):
        m = re.match(r"^F(\d+)([A-Za-z_].*)?$", str(col))
        return (int(m.group(1)), m.group(2) or "") if m else None

    fcols = [c for c in df.columns if str(c).startswith("F")]
    num_to_cols = {}
    for c in fcols:
        p = parse_f(c)
        if not p:
            continue
        n, _ = p
        num_to_cols.setdefault(n, []).append(c)

    group_specs_numeric = [
        ("Aktuelle Wohnsituation (F1–F5)",        1,   5),
        ("Zukünftige Wohnsituation (F6–F10)",     6,  10),
        ("Sicherheit (F11)",                      11, 11),
        ("Haushalt (F12–F20)",                    12, 20),
        ("Mobilität & Freizeit (F21–F31)",        21, 31),
        ("Seniorenpolitik (F32–F34)",             32, 34),
        ("Gesundheit (F35–F38)",                  35, 38),
        ("Unterstützung für andere (F39–F41)",    39, 41),
        ("Finanzielle Situation (F41–F45)",       41, 45),
    ]

    group_to_cols = {}
    for label, a, b in group_specs_numeric:
        cols = []
        for n in range(a, b + 1):
            cols.extend(num_to_cols.get(n, []))
        cols = [c for c in df.columns if c in set(cols)]
        if cols:
            group_to_cols[label] = cols

    color_candidates = list(df.columns)
    return color_candidates, group_to_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Steuerung & Parameter""")
    return


@app.cell
def _(color_candidates, group_to_cols, mo):
    default_group = next(iter(group_to_cols.keys())) if group_to_cols else None

    group_dropdown = mo.ui.dropdown(
        options=list(group_to_cols.keys()),
        value=default_group,
        label="Fragegruppe für Embedding (eine wählen)"
    )

    color_by = mo.ui.dropdown(
        options=color_candidates,
        value=("Jahr" if "Jahr" in color_candidates else (color_candidates[0] if color_candidates else None)),
        label="Farbkodierung (Color by)"
    )

    dimred_dropdown = mo.ui.dropdown(
        options=["UMAP", "t-SNE"],
        value="t-SNE",
        label="Dimensionality Reduction"
    )

    svd_dims = mo.ui.slider(5, 200, value=50, label="SVD Embedding Dimensionality (k)")
    umap_neighbors = mo.ui.slider(5, 200, value=30, label="UMAP n_neighbors")
    umap_mindist = mo.ui.slider(0, 99, value=10, label="UMAP min_dist (x/100)")
    tsne_perplexity = mo.ui.slider(5, 100, value=35, label="t-SNE perplexity")
    tsne_iter = mo.ui.slider(250, 4000, value=1000, label="t-SNE n_iter")

    mo.vstack((
        group_dropdown,
        mo.hstack((dimred_dropdown, svd_dims)),
        mo.hstack((umap_neighbors, umap_mindist)),
        mo.hstack((tsne_perplexity, tsne_iter)),
        color_by,
    ), gap="1rem")
    return (
        color_by,
        dimred_dropdown,
        group_dropdown,
        svd_dims,
        tsne_iter,
        tsne_perplexity,
        umap_mindist,
        umap_neighbors,
    )


@app.cell
def _(group_dropdown, group_to_cols):
    chosen_group_value = group_dropdown.value
    want = []
    seen = set()
    if chosen_group_value and chosen_group_value in group_to_cols:
        for d in group_to_cols[chosen_group_value]:
            if d not in seen:
                seen.add(d)
                want.append(d)
    if not want:
        for _grp_cols in group_to_cols.values():
            if _grp_cols:
                want = _grp_cols
                break
    return (want,)


@app.cell(hide_code=True)
def _(group_dropdown, mo):
    mo.md(
        f"""
    ### Aktive Fragegruppe
    **{group_dropdown.value}** — die folgenden F-Spalten werden einbezogen.
    """
    )
    return


@app.cell
def _(df, mo, want):
    mo.as_html(df[want])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Pipeline: One-Hot → SVD(k) → 2D-Projektion (Komponenten werden zurückgeschrieben)""")
    return


@app.cell
def _(
    df,
    dimred_dropdown,
    np,
    svd_dims,
    tsne_iter,
    tsne_perplexity,
    umap_mindist,
    umap_neighbors,
    want,
):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE

    # One-hot encode selected F-columns as strings
    X_subset = df[want].copy()
    ohe = OneHotEncoder(handle_unknown="ignore")
    X_ohe = ohe.fit_transform(X_subset.astype(str))

    # Truncated SVD to k dims
    max_k = max(2, X_ohe.shape[1] - 1)
    k_dims = int(min(max(2, svd_dims.value), max_k))
    svd = TruncatedSVD(n_components=k_dims, random_state=42)
    Z = svd.fit_transform(X_ohe)

    # 2D projection (UMAP or t-SNE)
    n_samples = Z.shape[0]
    if dimred_dropdown.value == "UMAP":
        import umap
        umap_model = umap.UMAP(
            n_neighbors=int(max(2, min(umap_neighbors.value, n_samples - 1))),
            min_dist=float(min(max(0.0, umap_mindist.value / 100.0), 0.99)),
            n_components=2,
            random_state=42,
            metric="euclidean",
        )
        XY = umap_model.fit_transform(Z)
        projection_desc = f"UMAP (n_neighbors={umap_model.n_neighbors}, min_dist={umap_model.min_dist:.2f})"
    else:
        safe_perp = float(max(5, min(tsne_perplexity.value, max(5, n_samples - 5))))
        tsne_model = TSNE(
            n_components=2,
            perplexity=safe_perp,
            n_iter_without_progress=tsne_iter.value,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        XY = tsne_model.fit_transform(Z)
        projection_desc = f"t-SNE (perplexity={safe_perp:.0f}, n_iter={int(tsne_iter.value)})"

    explained = getattr(svd, "explained_variance_ratio_", None)

    # Return a copy of the original df with components added
    df_embed = df.copy()
    df_embed["comp1"] = XY[:, 0]
    df_embed["comp2"] = XY[:, 1]
    # Optional: stable row id for debugging/joins
    df_embed["_row_id"] = np.arange(len(df_embed))
    return df_embed, explained, projection_desc


@app.cell
def _(color_by, df_embed, explained, mo, pd, projection_desc):
    import altair as alt

    # Decide color encoding (numeric vs categorical)
    color_field = color_by.value
    df_embed_copy = df_embed.copy()

    title_text = f"2D map of respondents — {projection_desc} • Color: {color_field}"
    subtitle = None
    if explained is not None:
        top = min(10, len(explained))
        subtitle = "SVD explained variance (top {}): {}".format(
            top, ", ".join(f"{x:.3f}" for x in explained[:top])
        )

    if pd.api.types.is_numeric_dtype(df_embed_copy[color_field]):
        color_enc = alt.Color(f"{color_field}:Q", title=color_field)
        tooltip = [
            alt.Tooltip("comp1:Q", title="Component 1"),
            alt.Tooltip("comp2:Q", title="Component 2"),
            alt.Tooltip(f"{color_field}:Q", title=color_field),
        ]
    else:
        # keep legend readable: top 12 cats, rest "Other"
        cats = df_embed[color_field].astype("object")
        counts = cats.value_counts(dropna=False)
        top_levels = set(map(str, counts.index[:12].tolist()))
        df_embed_copy["_color_cat"] = cats.map(
            lambda v: ("Missing" if pd.isna(v) else str(v))
        ).map(lambda s: s if s in top_levels else "Other")
        color_enc = alt.Color("_color_cat:N", title=color_field)
        tooltip = [
            alt.Tooltip("comp1:Q", title="Component 1"),
            alt.Tooltip("comp2:Q", title="Component 2"),
            alt.Tooltip("_color_cat:N", title=color_field),
            alt.Tooltip(f"{color_field}:N", title="Raw value"),
        ]

    brush = alt.selection_interval(name="brush")

    chart = (
        alt.Chart(df_embed_copy)
        .mark_circle(size=30, opacity=0.85)
        .encode(
            x=alt.X("comp1:Q", axis=alt.Axis(title="Component 1")),
            y=alt.Y("comp2:Q", axis=alt.Axis(title="Component 2")),
            color=color_enc,
            tooltip=tooltip,
        )
        .properties(
            width=600,
            height=520,
            title={"text": title_text, **({"subtitle": subtitle} if subtitle else {})},
        )
        .add_params(brush)
    )

    mo_chart = mo.ui.altair_chart(chart)
    mo_chart
    return (mo_chart,)


@app.cell
def _(df_embed, mo, mo_chart):
    # Show the brushed selection with ALL columns (including comp1/comp2)
    sel = mo_chart.value
    # Fallback: if nothing selected yet, show empty frame with correct columns
    if sel is None or (hasattr(sel, "empty") and sel.empty):
        sel = df_embed.iloc[0:0]  # empty with schema
    mo.md("### Brush-Auswahl: vollständiger Datensatz (alle Spalten)")
    mo.as_html(sel)
    return


if __name__ == "__main__":
    app.run()
