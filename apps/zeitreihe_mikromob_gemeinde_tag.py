# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "altair==5.5.0",
#   "babel==2.17.0",
#   "imageio==2.37.2",
#   "marimo>=0.7.11",
#   "numpy==2.3.4",
#   "pandas==2.2.3",
#   "requests==2.32.5",
#   "shapely==2.1.2",
#   "vl-convert-python==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    import requests
    import os
    import json
    from pathlib import Path
    return alt, json, mo, os, pd, requests


@app.cell
def _(os, pd, requests):
    def get_dataset(dataset_id: str) -> pd.DataFrame:
        url = f"https://data.bs.ch/api/explore/v2.1/catalog/datasets/{dataset_id}/exports/csv"
        r = requests.get(
            url,
            params={
                "timezone": "Europe%2FZurich",
                "use_labels": "true",
            },
            timeout=60,
        )
        data_path = os.path.join(os.getcwd(), "..", "data")
        os.makedirs(data_path, exist_ok=True)
        csv_path = os.path.join(data_path, f"{dataset_id}.csv")
        with open(csv_path, "wb") as f:
            f.write(r.content)

        df = pd.read_csv(
            url,
            sep=";",
            on_bad_lines="warn",
            encoding_errors="ignore",
            low_memory=False,
        )
        if df.shape[1] <= 1:
            print(
                "Die Daten wurden nicht korrekt importiert – vermutlich falscher Separator. Bitte Dataset prüfen."
            )
        return df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    # Geteilte Mikromobilität nach Gemeinde und Tag (Kanton Basel-Stadt)
    Choropleth der kumulierten durchschnittlichen Verfügbarkeit von Mikromobilitätsfahrzeugen pro Gemeinde für den gewählten Tag.
    """)
    return


@app.cell
def _(os, pd):
    # Daten lesen
    df = get_dataset("100422")
    # df = pd.read_csv(os.path.join("data/100422.csv"), sep=";", on_bad_lines="warn", encoding_errors="ignore", low_memory=False)
    return (df,)


@app.cell
def _(df, pd):
    # Datumsfeld normalisieren
    if "date" in df.columns:
        df["date"] = (
            pd.to_datetime(df["date"], errors="coerce", utc=True)
              .dt.tz_convert("Europe/Zurich")
              .dt.normalize()
        )
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # Numerik
    for c in ["mean", "min", "max", "num_measures"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # IDs/Names
    if "objid" in df.columns:
        df["objid"] = pd.to_numeric(df["objid"], errors="coerce").astype("Int64")

    for c in ["name", "xs_provider_name", "date_str", "gemeinde_na"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    return


@app.cell
def _(df, json, pd):
    # Geometrien je Gemeinde (letzter Stand)
    gemeinden = (
        df.loc[df["geometry"].notna(), ["objid", "name", "geometry", "date"]]
          .sort_values(["objid", "date"], ascending=[True, False])
          .drop_duplicates(subset=["objid"], keep="first")
          .drop(columns=["date"])
          .reset_index(drop=True)
    )

    # Typen sichern
    if "objid" in gemeinden.columns:
        gemeinden["objid"] = pd.to_numeric(gemeinden["objid"], errors="coerce").astype("Int64")

    # Geometrie validieren/parsen
    def _parse_geom(g):
        if isinstance(g, str):
            try:
                return json.loads(g)
            except Exception:
                return None
        return g

    gemeinden["geometry"] = gemeinden["geometry"].map(_parse_geom)
    gemeinden = gemeinden[
        gemeinden["geometry"].apply(lambda g: isinstance(g, dict) and "type" in g and "coordinates" in g)
    ].reset_index(drop=True)

    gemeinden.rename(columns={"name": "gemeinde_name"}, inplace=True)
    return (gemeinden,)


@app.cell
def _(df, mo):
    if df.empty:
        mo.stop("Keine Daten geladen.")
    # verfügbare Tage
    if "date" not in df.columns:
        mo.stop("Spalte 'date' fehlt/ungültig.")
    dates = (
        df.loc[df["date"].notna(), "date"]
          .dt.normalize()
          .sort_values()
          .unique()
    )
    if len(dates) == 0:
        mo.stop("Kein gültiges Datum in den Daten.")
    return (dates,)


@app.cell
def _(dates, mo, pd):
    start_str = str(pd.to_datetime(dates[0]).date())
    stop_str  = str(pd.to_datetime(dates[-1]).date())
    value_str = stop_str

    selected_date = mo.ui.date(
        value=value_str, start=start_str, stop=stop_str, label="Tag"
    )
    selected_date
    return (selected_date,)


@app.cell
def _(df, mo, pd):
    alle = "Alle"

    def opt(series):
        uniq = pd.Series(sorted(v for v in pd.unique(series) if pd.notna(v))).tolist()
        return [alle] + uniq

    dd_anbieter = mo.ui.dropdown(
        opt(df["xs_provider_name"]), value=alle, label="Anbieter"
    )
    dd_anbieter
    return alle, dd_anbieter


@app.cell
def _(alle, dd_anbieter, df, gemeinden, pd, selected_date):
    chosen_date_str = selected_date.value.strftime("%Y-%m-%d")
    day_df = df[df["date_str"] == chosen_date_str]

    if dd_anbieter is not None and dd_anbieter.value != alle:
        day_df = day_df[day_df["xs_provider_name"] == dd_anbieter.value]

    # Aggregation pro Gemeinde (ohne Geometrie in groupby)
    if not {"objid", "name", "mean"}.issubset(day_df.columns):
        # Fallback: Spaltennamen prüfen
        raise RuntimeError("Benötigte Spalten 'objid', 'name', 'mean' fehlen im Dataset 100422.")

    values = (
        day_df.loc[:, ["objid", "name", "mean"]]
              .groupby(["objid", "name"], dropna=False, as_index=False)
              .agg(value=("mean", "sum"))
    )

    # Join auf Geometrien (alle Gemeinden behalten)
    agg = gemeinden.merge(
        values.rename(columns={"name": "gemeinde_name"}),
        on=["objid", "gemeinde_name"],
        how="left"
    )
    agg["value"] = pd.to_numeric(agg["value"], errors="coerce").fillna(0.0)
    return (agg,)


@app.cell
def _(agg, json, pd):
    features = []
    for _, row in agg.iterrows():
        geom = row["geometry"]
        if isinstance(geom, str):
            try:
                geom = json.loads(geom)
            except Exception:
                continue
        if not isinstance(geom, dict) or "type" not in geom or "coordinates" not in geom:
            continue

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "objid": int(row["objid"]) if pd.notna(row["objid"]) else None,
                "gemeinde_name": row["gemeinde_name"],
                "value": float(row["value"]) if pd.notna(row["value"]) else 0.0,
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    return (geojson,)


@app.cell
def _(alt, geojson):
    if geojson["features"]:
        data = alt.InlineData(values=geojson, format={"type": "json", "property": "features"})
        chart = (
            alt.Chart(data)
            .mark_geoshape(stroke="white", strokeWidth=0.5)
            .encode(
                color=alt.Color("properties.value:Q", title="∑ Verfügbarkeit"),
                tooltip=[
                    alt.Tooltip("properties.gemeinde_name:N", title="Gemeinde"),
                    alt.Tooltip("properties.value:Q", title="∑ Verfügbarkeit", format=".3f"),
                ],
            )
            .project(type="mercator")
            .properties(width=720, height=520)
        )
    chart
    return


@app.cell
def _(mo):
    btn_ts = mo.ui.run_button(
        label="Generiere Zeitreihe",
        kind="success",
        tooltip="Erzeuge GIF-Zeitreihe"
    )
    btn_ts
    return (btn_ts,)


@app.cell
def _(alle, alt, btn_ts, dd_anbieter, df, gemeinden, json, mo, os, pd):
    import io
    import re
    import numpy as np
    from vl_convert import vegalite_to_png
    import imageio.v3 as iio
    from babel.dates import format_date

    mo.stop(not btn_ts.value)

    # --- Auswahl
    provider = dd_anbieter.value if dd_anbieter is not None else None

    base_gif = df.copy()
    if dd_anbieter is not None and provider != alle:
        base_gif = base_gif[base_gif["xs_provider_name"] == provider]

    # Dates
    ts = (
        base_gif.loc[base_gif["date"].notna(), "date"]
          .dt.normalize()
          .sort_values()
          .unique()
    )
    if len(ts) == 0:
        mo.stop("Keine Daten für die aktuelle Auswahl.")

    # Globale Farbdomäne: 95. Perzentil (robust)
    base_clean = base_gif.copy()
    base_clean["mean"] = pd.to_numeric(base_clean["mean"], errors="coerce")
    base_clean = base_clean.dropna(subset=["date_str", "objid"])
    vals = (
        base_clean.groupby(["date_str", "objid", "name"], as_index=False)["mean"]
                 .sum()
                 .rename(columns={"mean": "value"})
    )
    p = 0.95
    if vals.empty:
        min_val, max_val = 0.0, 1.0
    else:
        min_val = 0.0
        max_val = float(vals["value"].quantile(p))
        if not np.isfinite(max_val) or max_val <= 0:
            max_val = float(vals["value"].max() or 1.0)

    def slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip()) if s else "_"

    title_bits = []
    if dd_anbieter is not None and provider != alle:
        title_bits.append(provider)
    title_suffix = " • ".join(title_bits) if title_bits else "Alle"

    frames = []
    dates_norm = [pd.to_datetime(d) for d in ts]
    total = len(dates_norm)

    with mo.status.progress_bar(total=total, title="Generiere GIF", subtitle="Initialisiere…") as bar:
        for d in dates_norm:
            dkey   = pd.to_datetime(d).strftime("%Y-%m-%d")
            dlabel = format_date(d, format="full", locale="de_CH")

            day_vals = (
                base_gif.loc[base_gif["date_str"] == dkey, ["objid", "name", "mean"]]
                        .groupby(["objid", "name"], dropna=False, as_index=False)
                        .agg(value=("mean", "sum"))
                        .rename(columns={"name": "gemeinde_name"})
            )

            day_full = gemeinden.merge(day_vals, on=["objid", "gemeinde_name"], how="left")
            day_full["value"] = pd.to_numeric(day_full["value"], errors="coerce").fillna(0.0)

            features_gif = []
            for _, r in day_full.iterrows():
                geom_gif = r["geometry"]
                if isinstance(geom_gif, str):
                    try:
                        geom_gif = json.loads(geom_gif)
                    except Exception:
                        continue
                if not isinstance(geom_gif, dict) or "type" not in geom_gif or "coordinates" not in geom_gif:
                    continue

                features_gif.append({
                    "type": "Feature",
                    "geometry": geom_gif,
                    "properties": {
                        "objid": int(r["objid"]) if pd.notna(r["objid"]) else None,
                        "gemeinde_name": r["gemeinde_name"],
                        "value": float(r["value"]) if pd.notna(r["value"]) else 0.0,
                    },
                })

            if not features_gif:
                bar.update(subtitle=f"{dlabel} (leer)")
                continue

            geojson_gif = {"type": "FeatureCollection", "features": features_gif}
            data_gif = alt.InlineData(values=geojson_gif, format={"type": "json", "property": "features"})
            title = f"∑ Verfügbarkeit {dlabel} — {title_suffix}"

            chart_gif = (
                alt.Chart(data_gif)
                  .mark_geoshape(stroke="white", strokeWidth=0.5)
                  .encode(
                      color=alt.Color(
                          "properties.value:Q",
                          title="∑ Verfügbarkeit",
                          scale=alt.Scale(domain=[min_val, max_val], clamp=True, nice=False),
                      ),
                      tooltip=[
                          alt.Tooltip("properties.gemeinde_name:N", title="Gemeinde"),
                          alt.Tooltip("properties.value:Q", title="∑ Verfügbarkeit", format=".3f"),
                      ],
                  )
                  .project(type="mercator")
                  .properties(width=720, height=520, title=title)
            )

            from vl_convert import vegalite_to_png
            png_bytes = vegalite_to_png(chart_gif.to_dict())
            import imageio.v3 as iio, io as _io
            frames.append(iio.imread(_io.BytesIO(png_bytes)))
            bar.update(subtitle=dlabel)

    if not frames:
        mo.stop("Keine gültigen Frames erzeugt.")

    out_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(out_dir, exist_ok=True)
    fname_bits = ["zeitreihe"]
    if dd_anbieter is not None and provider:
        fname_bits.append(slug(provider))
    fname = "_".join(fname_bits) + ".gif"
    out_path = os.path.join(out_dir, fname)

    import imageio.v3 as iio
    iio.imwrite(out_path, frames, duration=1000, loop=0)

    preview = mo.image(out_path, caption=f"Zeitreihe ({len(frames)} Frames)")
    dl = mo.download(
        data=lambda: open(out_path, "rb"),
        filename=fname,
        mimetype="image/gif",
        label="GIF herunterladen"
    )
    mo.vstack([preview, dl], gap="0.75rem")
    return


if __name__ == "__main__":
    app.run()
