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
    # 2D-Embedding der Antworten (Ordinal/Binär/Nominal → SVD → t-SNE/UMAP)

    **Ziel:** Aus ausgewählten Fragen (Gruppe) werden **ordinale/binä­re** Items sinnvoll als **Scores** kodiert (Reihenfolge bleibt erhalten), **nominale** Items bleiben **one-hot**. Danach **Truncated SVD** auf \(k\) Dimensionen und 2D-Projektion via **t-SNE** oder **UMAP**.  
    Die resultierenden **Komponenten (`comp1`, `comp2`)** werden in den Original-DataFrame zurückgeschrieben. Die **Brush-Auswahl** zeigt am Ende **alle Spalten** der selektierten Zeilen.
    """
    )
    return


@app.cell
def _(pd):
    df = pd.read_csv("data/Antworten.csv")
    return (df,)


@app.cell
def _(pd):
    meta = pd.read_csv("data/Variablen_Befragung_55_plus_2023.csv")
    # normalize
    meta["FrageName"] = meta["FrageName"].astype(str).str.strip()
    meta["Format"] = meta["Format"].astype(str).str.strip()
    return (meta,)


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
    for fcol in fcols:
        p = parse_f(fcol)
        if not p:
            continue
        n, _ = p
        num_to_cols.setdefault(n, []).append(fcol)

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
def _(color_candidates, group_to_cols, meta, mo):
    default_group = next(iter(group_to_cols.keys())) if group_to_cols else None

    group_dropdown = mo.ui.dropdown(
        options=list(group_to_cols.keys()),
        value=default_group,
        label="Fragegruppe für Embedding (eine wählen)"
    )

    # --- Build restricted color-by options from metadata ---
    label_map = dict(zip(meta["FrageName"], meta["FrageLabel"]))

    allowed_color_vars = [
        "Methode","S1","S2","S3","S4","S5","S6","S7","WV",
        "Wohndauer_Adresse","Wohndauer_Quartier","Wohndauer_Kanton",
        "Wohnflaeche","Zimmerzahl","F1",
    ]

    valid_color_vars = [v for v in allowed_color_vars if v in color_candidates]

    # options as (label, value) tuples
    color_options = [(label_map.get(v, v), v) for v in valid_color_vars]

    # default must be one of the actual option *tuples*
    def _default_pair(target_value: str, options):
        for pair in options:
            if len(pair) == 2 and pair[1] == target_value:
                return pair
        return options[0] if options else None

    default_color_pair = _default_pair("S3", color_options)

    color_by = mo.ui.dropdown(
        options=color_options,
        value=default_color_pair,
        label="Farbkodierung (Color by – Fragen mit Personenbezug)"
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
    already_seen = set()
    if chosen_group_value and chosen_group_value in group_to_cols:
        for d in group_to_cols[chosen_group_value]:
            if d not in already_seen:
                already_seen.add(d)
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
    mo.md(r"""## Pipeline: Gemischtes Encoding → SVD(k) → 2D-Projektion (Komponenten werden zurückgeschrieben)""")
    return


@app.cell
def _(
    df,
    dimred_dropdown,
    meta,
    np,
    pd,
    svd_dims,
    tsne_iter,
    tsne_perplexity,
    umap_mindist,
    umap_neighbors,
    want,
):
    from sklearn.impute import SimpleImputer

    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE

    # ---- Hardcoded Format → Antwortordnung / Typ (nur für Dim-Red-Spalten) ----
    # type: 'ordinal', 'binary', or 'nominal'
    F = {
        "altersgruppe.": dict(type="ordinal", order=["55-64 Jahre","65-74 Jahre",">74 Jahre"]),
        "dauer_adr.":    dict(type="ordinal", order=["0 bis 2 Jahre","3 bis 5 Jahre","6 bis 10 Jahre","Mehr als 10 Jahre"]),
        "dauer_kt.":     dict(type="ordinal", order=["0 bis 10 Jahre","11 bis 20 Jahre","21 bis 35 Jahre","Mehr als 35 Jahre"]),
        "dauer_pflege.": dict(type="ordinal", order=["0 Stunden","1 bis 5 Stunden","6 bis 10 Stunden","11 bis 15 Stunden","16 bis 20 Stunden","Mehr als 20 Stunden"]),
        "dauer_qrt.":    dict(type="ordinal", order=["0 bis 4 Jahre","5 bis 8 Jahre","9 bis 15 Jahre","Mehr als 15 Jahre"]),
        "geld.":         dict(type="ordinal", order=["Geld reicht nicht","Brauche/Mache ich nicht","Geld reicht"]),  # sinnvoll: schlecht < neutral < gut
        "gerne.":        dict(type="ordinal", order=["Sehr ungerne","Eher ungerne","Eher gerne","Sehr gerne"], na={"Weiss nicht"}),
        "geschlecht.":   dict(type="binary",  map={"Männlich":0,"Weiblich":1}),
        "gut.":          dict(type="ordinal", order=["Gar nicht gut","Eher nicht gut","Eher gut","Sehr gut"], na={"Weiss nicht"}),
        "haeufig_dwm.":  dict(type="ordinal", order=["Nie","Selten","Mindestens ein Mal pro Monat","Mindestens ein Mal pro Woche","Mindestens ein Mal pro Tag"], na={"Nicht möglich / Nicht vorhanden"}),
        "haeufig_einsam.": dict(type="ordinal", order=["Nie","Manchmal","Ziemlich häufig","Sehr häufig"], na={"Weiss nicht"}),
        "haeufig_regelm.": dict(type="ordinal", order=["Nie","Selten","Hin und wieder","Regelmässig"], na={"Nicht möglich / Nicht vorhanden"}),
        "haeufig_spez.": dict(type="ordinal", order=["Nie","Seltener","2- bis 3-mal im Jahr","2- bis 3-mal im Monat","Einmal pro Woche","Mehrmals pro Woche"]),
        "hauptverkehrsmittel.": dict(type="nominal"),  # echte Nennungen → one-hot
        "haushalt.":     dict(type="binary",  map={"Einpersonenhaushalt":0,"Mehrpersonenhaushalt":1}),
        "hheinkommen.":  dict(type="ordinal", order=["<5 000","5 000-7 499",">7 500"], na={"Keine Antwort","Weiss nicht"}),
        "janein.":       dict(type="binary",  map={"Nein":0,"Ja":1}),
        "janein_gesund.":dict(type="ordinal", order=["Nein","Ja, bis zu einem gewissen Grad","Ja, sehr"], na={"Keine Antwort"}),
        "kenneich.":     dict(type="binary",  map={"Kenne ich nicht":0,"Kenne ich":1}),
        "methode.":      dict(type="nominal"),
        "schulabschluss_aggr.": dict(type="ordinal", order=["Obligatorische Schule","Sekundarstufe II","Höhere Berufsbildung","Hochschule"]),
        "seniorenfreundlich.":  dict(type="ordinal", order=["Gar nicht seniorenfreundlich","Eher weniger seniorenfreundlich","Eher seniorenfreundlich","Sehr seniorenfreundlich"], na={"Weiss nicht"}),
        "sicher.":       dict(type="ordinal", order=["Sehr unsicher","Eher unsicher","Eher sicher","Sehr sicher"], na={"Trifft nicht auf mich zu","Weiss nicht"}),
        "staatsang.":    dict(type="nominal"),
        "teiz.":         dict(type="ordinal", order=["Nein","Ja, weniger als 50%","Ja, zwischen 50 und 89%"]),
        "vorstellbar.":  dict(type="ordinal", order=["Gar nicht vorstellbar","Eher nicht vorstellbar","Eher vorstellbar","Gut vorstellbar"], na={"Weiss nicht"}),
        "wahlkreis.":    dict(type="nominal"),
        "wichtig.":      dict(type="ordinal", order=["Sehr unwichtig","Eher unwichtig","Eher wichtig","Sehr wichtig"], na={"Weiss nicht"}),
        "wohnflaeche_rg.": dict(type="ordinal", order=["70m2 oder weniger","71 bis 80 m2","81 bis 100 m2","101 bis 120 m2","121 bis 160 m2","Mehr als 160 m2"]),
        "wohnkosten.":   dict(type="ordinal", order=["< 1 000 Fr.","1 000-1 499 Fr.","1 500-1 999 Fr.","2 000-2 499 Fr.","2 500-2 999 Fr.","> 3 000 Fr."]),
        "wohnviertel.":  dict(type="nominal"),
        "zimmerzahl.":   dict(type="ordinal", order=["1 bis 2,5 Zimmer","3 bis 3,5 Zimmer","4 bis 4,5 Zimmer","5 Zimmer und mehr"]),
        "zufrieden.":    dict(type="ordinal", order=["Sehr unzufrieden","Eher unzufrieden","Eher zufrieden","Sehr zufrieden"], na={"Weiss nicht","**OTHER**"}),
        "zustimmen.":    dict(type="ordinal", order=["Stimmt gar nicht","Stimmt eher nicht","Stimmt eher","Stimmt genau"], na={"Weiss nicht"}),
    }

    # ---- helper: encode a single column by format spec ----
    def encode_series(col: pd.Series, spec: dict) -> pd.DataFrame:
        s = col.astype("object")
        if spec["type"] == "binary":
            mp = spec["map"]
            out = s.map(mp)
            return pd.DataFrame({col.name: out})
        if spec["type"] == "ordinal":
            order = spec["order"]
            na = set(spec.get("na", set()))
            cat = pd.Categorical(s, categories=order, ordered=True)
            out = cat.codes.astype("float")
            out[out < 0] = np.nan
            # custom NA labels
            out[s.astype(str).isin(na)] = np.nan
            return pd.DataFrame({col.name: out})
        # nominal → one-hot
        d = pd.get_dummies(s, prefix=col.name, dummy_na=False)
        return d

    # ---- find formats for selected columns ----
    fmt_map = dict(zip(meta["FrageName"], meta["Format"]))
    X_blocks = []
    used_cols = []
    for c in want:
        fmt = fmt_map.get(c, None)
        if fmt in F:
            block = encode_series(df[c], F[fmt])
            X_blocks.append(block)
            used_cols.extend(block.columns.tolist())
        else:
            # fallback: treat as nominal one-hot
            block = pd.get_dummies(df[c].astype("object"), prefix=c, dummy_na=False)
            X_blocks.append(block)
            used_cols.extend(block.columns.tolist())

    # ---- assemble feature matrix ----
    if not X_blocks:
        X = np.empty((len(df), 0))
        feat_names = []
    else:
        X_df = pd.concat(X_blocks, axis=1)
        feat_names = X_df.columns.tolist()

        # Identify column types by dtype produced by our encoders:
        # - Ordinal scores came out as float with possible NaNs
        # - One-hot/binary dummies are integer/uint (0/1)
        float_cols = X_df.select_dtypes(include=["float", "float32", "float64"]).columns
        int_cols   = X_df.select_dtypes(include=["int", "int32", "int64", "uint8", "uint16"]).columns

        # 1) Ordinal floats: median impute (column-wise)
        if len(float_cols) > 0:
            med_imp = SimpleImputer(strategy="median")
            X_df.loc[:, float_cols] = med_imp.fit_transform(X_df[float_cols])

        # 2) Dummies/binary: fill NaN with 0
        if len(int_cols) > 0:
            X_df.loc[:, int_cols] = X_df[int_cols].fillna(0)

        # (Optional) If you want to drop rows still containing NaN (e.g., unforeseen types):
        # X_df = X_df.dropna(axis=0)

        # ---- standardize numeric score columns (keep dummies as 0/1) ----
        X_scaled = X_df.copy()
        if len(float_cols) > 0:
            mu = X_df[float_cols].mean(axis=0)
            sd = X_df[float_cols].std(axis=0).replace(0, 1.0)
            X_scaled.loc[:, float_cols] = (X_df[float_cols] - mu) / sd

        X = X_scaled.to_numpy(dtype=float)

    # ---- SVD → k dims ----
    k_dims = int(min(max(2, svd_dims.value), max(2, X.shape[1] if X.shape[1] else 2)))
    if X.shape[1] == 0:
        Z = np.zeros((len(df), k_dims))
        explained = None
    else:
        svd = TruncatedSVD(n_components=k_dims, random_state=42)
        Z = svd.fit_transform(X)
        explained = getattr(svd, "explained_variance_ratio_", None)

    # ---- 2D projection ----
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

    # ---- return df with components ----
    df_embed = df.copy()
    df_embed["comp1"] = XY[:, 0]
    df_embed["comp2"] = XY[:, 1]
    df_embed["_row_id"] = np.arange(len(df_embed))
    return c, df_embed, explained, projection_desc


@app.cell
def _(c, color_by, df_embed, explained, mo, pd, projection_desc, want):
    import altair as alt

    # Decide color encoding (numeric vs categorical)
    color_field = color_by.value[1]
    # Select only the requested columns plus comp1/comp2 and ensure color field is present
    base_cols = [
        "Methode", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "WV",
        "Wohndauer_Adresse", "Wohndauer_Quartier", "Wohndauer_Kanton",
        "Wohnflaeche", "Zimmeranzahl", "comp1", "comp2",
    ]

    # include everything from want (ensure it's iterable)
    want_cols = base_cols + list(want) if want is not None else []

    # ensure the color field is included so subsequent code can access it
    if color_field not in want_cols and color_field in df_embed.columns:
        want_cols.append(color_field)

    # preserve order and remove duplicates, then keep only existing columns
    seen = set()
    sel_cols = []
    for want_col in want_cols:
        if want_col not in seen:
            seen.add(c)
            if want_col in df_embed.columns:
                sel_cols.append(want_col)

    df_embed_copy = df_embed[sel_cols].copy()

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
        # keep all categories (no top-12 collapse); mark missing explicitly
        cats = df_embed_copy[color_field].astype("object")
        df_embed_copy["_color_cat"] = cats.where(~pd.isna(cats), other="Missing").astype(str)
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
    if sel is None or (hasattr(sel, "empty") and sel.empty):
        sel = df_embed.iloc[0:0]  # empty with schema
    mo.md("### Brush-Auswahl: vollständiger Datensatz (alle Spalten)")
    mo.as_html(sel)
    return


if __name__ == "__main__":
    app.run()
