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

__generated_with = "0.18.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    import requests
    import os
    import json
    import io
    import re
    import numpy as np
    from vl_convert import vegalite_to_png
    import imageio.v3 as iio, io as _io
    from babel.dates import format_date
    return (
        alt,
        format_date,
        iio,
        io,
        json,
        mo,
        np,
        os,
        pd,
        re,
        requests,
        vegalite_to_png,
    )


@app.cell
def _(os):
    def ensure_data_dir() -> str:
        data_path = os.path.join(os.getcwd(), "..", "data")
        os.makedirs(data_path, exist_ok=True)
        return data_path
    return (ensure_data_dir,)


@app.cell
def _(io, pd, requests):
    def download_csv_chunk(
        session: requests.Session,
        dataset_id: str,
        extra_params: dict | None = None,
    ) -> pd.DataFrame:
        base_url = "https://data.bs.ch/api/explore/v2.1"
        url = f"{base_url}/catalog/datasets/{dataset_id}/exports/csv"

        params = {
            "timezone": "Europe/Zurich",
            "use_labels": "false",
        }
        if extra_params:
            params.update(extra_params)

        r = session.get(url, params=params)
        r.raise_for_status()

        buf = io.BytesIO(r.content)

        df = pd.read_csv(
            buf,
            sep=";",
            on_bad_lines="warn",
            encoding_errors="ignore",
            low_memory=False,
        )

        return df
    return (download_csv_chunk,)


@app.function
def pick_best_facet(facets_json: dict, max_rows_per_chunk: int) -> tuple[str | None, list[str]]:
    """
    Choose a facet column where each bucket has <= max_rows_per_chunk rows.
    Among all such columns, pick the one with the smallest worst-case bucket.
    Returns (facet_name, list_of_values) or (None, []) if nothing suitable.
    """
    best_name = None
    best_values: list[str] = []
    best_max_bucket = None

    for facet in facets_json.get("facets", []):
        facet_name = facet.get("name")
        value_list = facet.get("facets", [])
        if not facet_name or not value_list:
            continue

        counts = [v.get("count", 0) for v in value_list]
        if not counts:
            continue

        max_bucket = max(counts)
        if max_bucket > max_rows_per_chunk:
            # this facet would still exceed the chunk limit for some values
            continue

        if best_max_bucket is None or max_bucket < best_max_bucket:
            best_max_bucket = max_bucket
            best_name = facet_name
            best_values = [v.get("value") for v in value_list if v.get("value") is not None]

    return best_name, best_values


@app.cell
def _(download_csv_chunk, ensure_data_dir, mo, os, pd, requests):
    def get_dataset(dataset_id: str, max_rows_per_chunk: int = 50_000) -> pd.DataFrame:
        """
        Download a dataset from data.bs.ch.

        - For small datasets (<= max_rows_per_chunk): single CSV export.
        - For large datasets:
            * Inspect /facets to find a good splitting column.
            * Download one CSV per facet value via refine.<col>=<value>.
            * Additionally download rows where that column is NULL via q=#null(<col>).

        The combined result is written to ../data/{dataset_id}.csv and returned as a DataFrame.
        """
        base_url = "https://data.bs.ch/api/explore/v2.1"
        records_url = f"{base_url}/catalog/datasets/{dataset_id}/records"
        facets_url = f"{base_url}/catalog/datasets/{dataset_id}/facets"

        session = requests.Session()
        common_params = {
            "timezone": "Europe/Zurich",
            "use_labels": "false",
        }

        # 1) Get total_count via /records
        r = session.get(records_url, params={**common_params, "limit": 1})
        r.raise_for_status()
        meta = r.json()
        total_count = meta.get("total_count", 0)

        if not isinstance(total_count, int):
            # fall back to simple export if we can't read total_count
            total_count = 0

        # 2) Small dataset: single CSV export
        if total_count == 0 or total_count <= max_rows_per_chunk:
            df = download_csv_chunk(session, dataset_id, extra_params={})
            data_path = ensure_data_dir()
            csv_path = os.path.join(data_path, f"{dataset_id}.csv")
            df.to_csv(csv_path, index=False)
            return df

        # 3) Large dataset: inspect /facets to decide how to split
        r = session.get(facets_url, params=common_params)
        r.raise_for_status()
        facets_json = r.json()

        facet_name, facet_values = pick_best_facet(facets_json, max_rows_per_chunk=max_rows_per_chunk)

        if facet_name is None or not facet_values:
            raise RuntimeError(
                f"Could not find a facet column to split dataset {dataset_id} into "
                f"chunks of <= {max_rows_per_chunk} rows. Please handle this dataset manually."
            )

        dfs: list[pd.DataFrame] = []

        # 4) Download each facet value (refine.<facet_name>=value)
        n_parts = len(facet_values) + 1  # +1 for NULLs

        with mo.status.progress_bar(
            total=n_parts,
            title=f"Lade Datensatz {dataset_id}",
            subtitle=f"Split nach '{facet_name}'…",
        ) as bar:
            # NULLs
            bar.update(subtitle=f"{facet_name} = NULL")
            null_query = {"qv1": f"#null({facet_name})"}
            try:
                df_null = download_csv_chunk(session, dataset_id, extra_params=null_query)
                if not df_null.empty:
                    dfs.append(df_null)
            except requests.HTTPError as e:
                print(f"Warning: NULL download for {facet_name} failed: {e}")
            # non-NULL values
            for v in facet_values:
                bar.update(subtitle=f"{facet_name} = {v!r}")
                params = {"refine": f'{facet_name}:"{v}"'}
                df_chunk = download_csv_chunk(session, dataset_id, extra_params=params)
                dfs.append(df_chunk)

        # 5) Combine all parts, save to a single CSV, return DataFrame
        if dfs:
            full_df = pd.concat(dfs, ignore_index=True)
        else:
            full_df = pd.DataFrame()

        data_path = ensure_data_dir()
        csv_path = os.path.join(data_path, f"{dataset_id}.csv")
        full_df.to_csv(csv_path, index=False)

        return full_df
    return (get_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    # Geteilte Mikromobilität nach Gemeinde und Tag
    """)
    return


@app.cell
def _(get_dataset):
    # Daten lesen
    df = get_dataset("100422")
    # df = pd.read_csv(os.path.join("data/100422.csv"), sep=";", on_bad_lines="warn", encoding_errors="ignore", low_memory=False)
    df
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
                color=alt.Color("properties.value:Q", title="∑ Verfügbarkeit",
                               scale=alt.Scale(scheme="blues")),
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
def _(
    alle,
    alt,
    btn_ts,
    dd_anbieter,
    df,
    format_date,
    gemeinden,
    iio,
    json,
    mo,
    np,
    os,
    pd,
    re,
    vegalite_to_png,
):
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
                          scale=alt.Scale(domain=[min_val, max_val], clamp=True, nice=False, scheme="blues"),
                      ),
                      tooltip=[
                          alt.Tooltip("properties.gemeinde_name:N", title="Gemeinde"),
                          alt.Tooltip("properties.value:Q", title="∑ Verfügbarkeit", format=".3f"),
                      ],
                  )
                  .project(type="mercator")
                  .properties(width=720, height=520, title=title)
            )

            png_bytes = vegalite_to_png(chart_gif.to_dict())
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
