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
    def get_dataset(dataset_id):
        url = f"https://data.bs.ch/api/explore/v2.1/catalog/datasets/{dataset_id}/exports/csv"
        r = requests.get(
            url, 
            params={
                "timezone": "Europe%2FZurich",
                "use_labels": "true"
            }
        )
        data_path = os.path.join(os.getcwd(), "..", "data")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        csv_path = os.path.join(data_path, f"{dataset_id}.csv")
        with open(csv_path, "wb") as f:
            f.write(r.content)
        data = pd.read_csv(
            url, sep=";", on_bad_lines="warn", encoding_errors="ignore", low_memory=False
        )
        # if dataframe only has one column or less the data is not ";" separated
        if data.shape[1] <= 1:
            print(
                "The data wasn't imported properly. Very likely the correct separator couldn't be found.\nPlease check the dataset manually and adjust the code."
            )
        return data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    # Mikromobilität im Kanton Basel-Stadt nach Bezirk
    Die Karte zeigt die kumulierte durchschnittliche Verfügbarkeit von Mikromobilitätsfahrzeugen (E-Scooter, E-Bikes etc.) pro Bezirk im Kanton Basel-Stadt für den ausgewählten Tag. Die Daten stammen von verschiedenen Anbietern und können nach Anbieter, Fahrzeugtyp, Bauweise, Antriebsart und Reichweite gefiltert werden.
    """)
    return


@app.cell
def _(os, pd):
    # Read the dataset
    df = get_dataset(dataset_id="100416")
    # df = pd.read_csv(os.path.join("data/100416.csv"), sep=";", on_bad_lines="warn", encoding_errors="ignore", low_memory=False)
    return (df,)


@app.cell
def _(df, pd):
    if "date" in df.columns:
        df["date"] = (
            pd.to_datetime(df["date"], errors="coerce", utc=True)
              .dt.tz_convert("Europe/Zurich")
              .dt.normalize()
        )
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")       # plain string

    # Numerik
    for c in ["xs_max_range_meters", "mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["wov_id", "bez_id","num_measures", "min", "max"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in ["gemeinde_na","wov_name","bez_name","xs_provider_name","xs_vehicle_type_name","xs_form_factor","xs_propulsion_type", "date_str"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    return


@app.cell
def _(df, json, pd):
    bezirke = (
        df.loc[df["geometry"].notna(), ["bez_id", "bez_name", "geometry", "date"]]
          .sort_values(["bez_id", "date"], ascending=[True, False])
          .drop_duplicates(subset=["bez_id"], keep="first")
          .drop(columns=["date"])
    )

    # Ensure types
    if "bez_id" in bezirke.columns:
        bezirke["bez_id"] = pd.to_numeric(bezirke["bez_id"], errors="coerce").astype("Int64")

    # Validate JSON-ish geometries (drop rows that cannot be parsed)
    def _parse_geom(g):
        if isinstance(g, str):
            try:
                return json.loads(g)
            except Exception:
                return None
        return g

    bezirke["geometry"] = bezirke["geometry"].map(_parse_geom)
    bezirke = bezirke[
        bezirke["geometry"].apply(lambda g: isinstance(g, dict) and "type" in g and "coordinates" in g)
    ].reset_index(drop=True)
    return (bezirke,)


@app.cell
def _(df, mo):
    if df.empty:
        mo.stop("Keine Daten geladen.")
    # verfügbare Tage
    dates = (
        df.loc[df["date"].notna(), "date"]
        .dt.normalize()
        .sort_values()
        .unique()
    )
    if len(dates) == 0:
        mo.stop("Spalte 'Datum' leer/ungültig.")
    return (dates,)


@app.cell
def _(dates, mo, pd):
    # Date picker bounded to available data
    start_str = str(pd.to_datetime(dates[0]).date())
    stop_str  = str(pd.to_datetime(dates[-1]).date())
    value_str = stop_str  # default to most recent date

    selected_date = mo.ui.date(
        value=value_str,
        start=start_str,
        stop=stop_str,
        label="Tag"
    )
    selected_date
    return (selected_date,)


@app.cell
def _(pd):
    alle = "Alle"

    threshold_options = 2

    def opt(series):
        uniq = pd.Series(sorted(v for v in pd.unique(series) if pd.notna(v))).tolist()
        return [alle] + uniq

    def opt_numeric(series):
        uniq = sorted(set(pd.to_numeric(series, errors="coerce").dropna().tolist()))
        # keep as strings so the dropdown displays nicely; convert back when filtering
        return [alle] + [str(int(x)) if float(x).is_integer() else str(x) for x in uniq]
    return alle, opt, threshold_options


@app.cell
def _(alle, df, mo, opt, selected_date):
    day_str = selected_date.value.strftime("%Y-%m-%d")
    providers = df.loc[df["date_str"] == day_str, "xs_provider_name"]

    dd_anbieter = mo.ui.dropdown(
        opt(providers),
        value=alle,
        label="Anbieter"
    )
    dd_anbieter
    return day_str, dd_anbieter


@app.cell
def _(alle, day_str, dd_anbieter, df, mo, opt, threshold_options):
    # compute filtered df for option-building
    base = df[df["date_str"] == day_str]
    if dd_anbieter.value != alle:
        base = base[base["xs_provider_name"] == dd_anbieter.value]

    # If Anbieter == Alle -> don't show dependent filters
    widgets = []

    # Fahrzeugtyp
    fahrzeug_opts = opt(base["xs_vehicle_type_name"])
    dd_fahrzeugtyp = mo.ui.dropdown(
        fahrzeug_opts, value=fahrzeug_opts[0], label="Fahrzeugtyp"
    )
    formfaktor_opts = opt(base["xs_form_factor"])
    dd_formfaktor = mo.ui.dropdown(
        formfaktor_opts, value=formfaktor_opts[0], label="Bauweise"
    )
    if dd_anbieter.value != alle:
        if len(fahrzeug_opts) > threshold_options:
            widgets.append(dd_fahrzeugtyp)
        if len(formfaktor_opts) > threshold_options:
            widgets.append(dd_formfaktor)

    # Render only what exists
    mo.hstack(widgets, justify="start", gap="1.0rem") if widgets else mo.md("")
    return dd_fahrzeugtyp, dd_formfaktor


@app.cell
def _(
    alle,
    bezirke,
    dd_anbieter,
    dd_fahrzeugtyp,
    dd_formfaktor,
    df,
    pd,
    selected_date,
):
    chosen_date_str = selected_date.value.strftime("%Y-%m-%d")
    day_df = df[df["date_str"] == chosen_date_str]

    if dd_anbieter.value != alle:
        day_df = day_df[day_df["xs_provider_name"] == dd_anbieter.value]
    if dd_fahrzeugtyp.value != alle:
        day_df = day_df[day_df["xs_vehicle_type_name"] == dd_fahrzeugtyp.value]
    if dd_formfaktor.value != alle:
        day_df = day_df[day_df["xs_form_factor"] == dd_formfaktor.value]

    # Aggregate by Bezirk (WITHOUT geometry in the groupby)
    values = (
        day_df.loc[:, ["bez_id", "bez_name", "mean"]]
              .groupby(["bez_id", "bez_name"], dropna=False, as_index=False)
              .agg(value=("mean", "sum"))
    )

    # Left-join against all Bezirke to keep polygons even when value is NaN/absent
    agg = bezirke.merge(values, on=["bez_id", "bez_name"], how="left")
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
        coords = geom.get("coordinates", [])
        if coords is None:
            continue

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "bez_id": int(row["bez_id"]) if pd.notna(row["bez_id"]) else None,
                "bez_name": row["bez_name"],
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
                    alt.Tooltip("properties.bez_name:N", title="Bezirk"),
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
    btn_ts = mo.ui.run_button(label="Generiere Zeitreihe", kind="success", tooltip="Erzeuge GIF-Zeitreihe")
    btn_ts
    return (btn_ts,)


@app.cell
def _(
    alle,
    alt,
    bezirke,
    btn_ts,
    dd_anbieter,
    dd_fahrzeugtyp,
    dd_formfaktor,
    df,
    json,
    mo,
    os,
    pd,
):
    import io
    import re
    import numpy as np
    from vl_convert import vegalite_to_png
    import imageio.v3 as iio
    from babel.dates import format_date

    mo.stop(not btn_ts.value)

    # --- selection ---
    provider = dd_anbieter.value
    veh      = dd_fahrzeugtyp.value
    form     = dd_formfaktor.value

    base_gif = df.copy()
    if provider != alle:
        base_gif = base_gif[base_gif["xs_provider_name"] == provider]
    if veh != alle:
        base_gif = base_gif[base_gif["xs_vehicle_type_name"] == veh]
    if form != alle:
        base_gif = base_gif[base_gif["xs_form_factor"] == form]

    # Ensure dates (normalized)
    ts = (
        base_gif.loc[base_gif["date"].notna(), "date"]
        .dt.normalize()
        .sort_values()
        .unique()
    )
    if len(ts) == 0:
        mo.stop("Keine Daten für die aktuelle Auswahl.")

    # --- robust global domain from per-frame 'value' with percentile ---
    base_clean = base_gif.copy()
    base_clean["mean"] = pd.to_numeric(base_clean["mean"], errors="coerce")
    base_clean = base_clean.dropna(subset=["date_str","bez_id"])
    vals = (
        base_clean.groupby(["date_str","bez_id","bez_name"], as_index=False)["mean"]
                  .sum()
                  .rename(columns={"mean":"value"})
    )
    p = 0.95
    if vals.empty:
        min_val, max_val = 0.0, 1.0
    else:
        min_val = 0.0
        max_val = float(vals["value"].quantile(p))
        if not np.isfinite(max_val) or max_val <= 0:
            max_val = float(vals["value"].max() or 1.0)

    # ---------- helpers ----------
    def slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip()) if s else "_"

    title_bits = []
    if provider != alle: title_bits.append(provider)
    if veh != alle:      title_bits.append(veh)
    if form != alle:     title_bits.append(form)
    title_suffix = " • ".join(title_bits) if title_bits else "Alle"

    frames = []
    dates_norm = [pd.to_datetime(d) for d in ts]
    total = len(dates_norm)

    # --- frame loop (note dkey vs dlabel) ---
    with mo.status.progress_bar(total=total, title="Generiere GIF", subtitle="Initialisiere…") as bar:
        for d in dates_norm:
            dkey   = pd.to_datetime(d).strftime("%Y-%m-%d")        # filter key
            dlabel = format_date(d, format="full", locale="de_CH") # pretty label

            day_vals = (
                base_gif.loc[base_gif["date_str"] == dkey, ["bez_id","bez_name","mean"]]
                        .groupby(["bez_id","bez_name"], dropna=False, as_index=False)
                        .agg(value=("mean","sum"))
            )

            day_full = bezirke.merge(day_vals, on=["bez_id","bez_name"], how="left")
            day_full["value"] = pd.to_numeric(day_full["value"], errors="coerce").fillna(0.0)

            features_gif = []
            for _, r in day_full.iterrows():
                geom_gif = r["geometry"]
                if isinstance(geom_gif, str):
                    try: geom_gif = json.loads(geom_gif)
                    except Exception: continue
                if not isinstance(geom_gif, dict) or "type" not in geom_gif or "coordinates" not in geom_gif:
                    continue

                features_gif.append({
                    "type":"Feature",
                    "geometry":geom_gif,
                    "properties":{
                        "bez_id": int(r["bez_id"]) if pd.notna(r["bez_id"]) else None,
                        "bez_name": r["bez_name"],
                        "value": float(r["value"]) if pd.notna(r["value"]) else 0.0,
                    },
                })

            if not features_gif:
                bar.update(subtitle=f"{dlabel} (leer)")
                continue

            geojson_gif = {"type":"FeatureCollection","features":features_gif}
            data_gif = alt.InlineData(values=geojson_gif, format={"type":"json","property":"features"})
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
                          alt.Tooltip("properties.bez_name:N", title="Bezirk"),
                          alt.Tooltip("properties.value:Q", title="∑ Verfügbarkeit", format=".3f"),
                      ],
                  )
                  .project(type="mercator")
                  .properties(width=720, height=520, title=title)
            )

            png_bytes = vegalite_to_png(chart_gif.to_dict())
            frames.append(iio.imread(io.BytesIO(png_bytes)))
            bar.update(subtitle=dlabel)


    if not frames:
        mo.stop("Keine gültigen Frames erzeugt.")

    # Output
    out_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"zeitreihe_{slug(provider)}_{slug(veh)}_{slug(form)}.gif"
    out_path = os.path.join(out_dir, fname)

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
