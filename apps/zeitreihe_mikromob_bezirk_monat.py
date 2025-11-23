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
    return alt, iio, io, json, mo, np, os, pd, re, requests, vegalite_to_png


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
    # Geteilte Mikromobilität nach Bezirk, Wochentagesabschnitt und Monat
    """)
    return


@app.cell
def _(get_dataset):
    # Daten laden (lokal oder via get_dataset)
    df = get_dataset(dataset_id="100428")
    # df = pd.read_csv(os.path.join("data/100428.csv"), sep=";")
    return (df,)


@app.cell
def _(df, pd):
    # --- Feldaufbereitung (ohne Spaltenprüfungen) ---
    # Monat als String (YYYY-MM) beibehalten
    df["date"] = df["date"].astype("string").str.strip()
    df["month_str"] = df["date"]  # Alias für Klarheit

    # Anbieter
    df["xs_provider_name"] = df["xs_provider_name"].astype("string").str.strip()

    # Bezirk
    df["bez_id"] = pd.to_numeric(df["bez_id"], errors="coerce").astype("Int64")
    df["bez_name"] = df["bez_name"].astype("string").str.strip()

    # Geometrie (als JSON-Struktur)
    df["geometry"] = df["geometry"].astype("string")

    # Mittelwert (Aggregationsfeld)
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")

    # Wochentag-Zuordnung (1=Montag … 7=Sonntag)
    weekday_map = {
        1: "Montag",
        2: "Dienstag",
        3: "Mittwoch",
        4: "Donnerstag",
        5: "Freitag",
        6: "Samstag",
        7: "Sonntag",
    }
    df["weekday"] = pd.to_numeric(df["weekday"], errors="coerce").astype("Int64")
    df["weekday_name"] = df["weekday"].map(weekday_map).astype("string")

    # 3-Stunden-Zeiträume als String "HH:MM-HH:MM"
    df["timerange_start"] = df["timerange_start"].astype("string").str.slice(0, 5)
    df["timerange_end"]   = df["timerange_end"].astype("string").str.slice(0, 5)
    df["timerange"] = df["timerange_start"] + "-" + df["timerange_end"]
    return


@app.cell
def _(df, json):
    # Neueste Geometrie pro Bezirk sichern (eine Zeile pro Bezirk)
    bezirke = (
        df.loc[df["geometry"].notna(), ["bez_id", "bez_name", "geometry"]]
          .drop_duplicates(subset=["bez_id"], keep="first")
          .reset_index(drop=True)
    )

    # JSON parse
    def _parse_geom(g):
        try:
            return json.loads(g) if isinstance(g, str) else g
        except Exception:
            return None

    bezirke["geometry"] = bezirke["geometry"].map(_parse_geom)
    bezirke = bezirke[
        bezirke["geometry"].apply(lambda g: isinstance(g, dict) and "type" in g and "coordinates" in g)
    ].reset_index(drop=True)
    return (bezirke,)


@app.cell
def _(df, mo, pd):
    alle = "Alle"

    # Dropdown-Optionen
    months = pd.Series(sorted(df["month_str"].dropna().unique().tolist())).tolist()
    weekdays = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    ranges = [
        "00:00-03:00", "03:00-06:00", "06:00-09:00", "09:00-12:00",
        "12:00-15:00", "15:00-18:00", "18:00-21:00", "21:00-24:00",
    ]
    providers = [alle] + pd.Series(sorted(df["xs_provider_name"].dropna().unique().tolist())).tolist()

    dd_month = mo.ui.dropdown(months, value=months[-1] if months else None, label="Monat (YYYY-MM)")
    dd_weekday = mo.ui.dropdown(weekdays, value="Montag", label="Wochentag")
    dd_range = mo.ui.dropdown(ranges, value="18:00-21:00", label="Zeitfenster (3h)")
    dd_provider = mo.ui.dropdown(providers, value=alle, label="Anbieter")

    mo.hstack([dd_month, dd_weekday, dd_range, dd_provider], gap="1rem")
    return alle, dd_month, dd_provider, dd_range, dd_weekday


@app.cell
def _(alle, bezirke, dd_month, dd_provider, dd_range, dd_weekday, df, pd):
    # Auswahl übernehmen
    month_sel = dd_month.value
    weekday_sel = dd_weekday.value
    range_sel = dd_range.value
    provider_sel = dd_provider.value

    # Filter
    day_df = df[
        (df["month_str"] == month_sel) &
        (df["weekday_name"] == weekday_sel) &
        (df["timerange"] == range_sel)
    ]
    if provider_sel != alle:
        day_df = day_df[day_df["xs_provider_name"] == provider_sel]

    # Aggregation (Summe der "mean" pro Bezirk)
    values = (
        day_df.loc[:, ["bez_id", "bez_name", "mean"]]
              .groupby(["bez_id", "bez_name"], as_index=False, dropna=False)
              .agg(value=("mean", "sum"))
    )

    # Join mit Geometrien; fehlende Werte = 0
    agg = bezirke.merge(values, on=["bez_id", "bez_name"], how="left")
    agg["value"] = pd.to_numeric(agg["value"], errors="coerce").fillna(0.0)

    # GeoJSON erzeugen
    features = []
    for _, row in agg.iterrows():
        geom = row["geometry"]
        if not (isinstance(geom, dict) and "type" in geom and "coordinates" in geom):
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

    # Label für Titel
    subtitle_bits = [month_sel, weekday_sel, range_sel]
    if provider_sel != alle:
        subtitle_bits.append(provider_sel)
    subtitle = " • ".join(subtitle_bits)
    return geojson, subtitle


@app.cell
def _(alt, geojson, subtitle):
    if geojson["features"]:
        data = alt.InlineData(values=geojson, format={"type": "json", "property": "features"})
        chart = (
            alt.Chart(data)
            .mark_geoshape(stroke="white", strokeWidth=0.5)
            .encode(
                color=alt.Color("properties.value:Q", title="∑ Verfügbarkeit",
                                scale=alt.Scale(scheme="blues")),
                tooltip=[
                    alt.Tooltip("properties.bez_name:N", title="Bezirk"),
                    alt.Tooltip("properties.value:Q", title="∑ Verfügbarkeit", format=".3f"),
                ],
            )
            .project(type="mercator")
            .properties(width=720, height=520, title=f"∑ Verfügbarkeit — {subtitle}")
        )
    chart
    return


@app.cell
def _(mo):
    btn_gif = mo.ui.run_button(
        label="Generiere GIF (Monat × Wochentag × Zeitfenster)",
        kind="success",
        tooltip="Erzeuge GIF über alle Monate, Wochentage und Zeitfenster"
    )
    btn_gif
    return (btn_gif,)


@app.cell
def _(
    alle,
    alt,
    bezirke,
    btn_gif,
    dd_provider,
    df,
    iio,
    io,
    json,
    mo,
    np,
    os,
    pd,
    re,
    vegalite_to_png,
):
    # only run when button is pressed
    mo.stop(not btn_gif.value)

    # --- selection: provider ---
    provider_sel_local = dd_provider.value

    # base data (optionally filtered by provider)
    base_local = df.copy()
    if provider_sel_local != alle:
        base_local = base_local[base_local["xs_provider_name"] == provider_sel_local]

    if base_local.empty:
        mo.stop("Keine Daten für die aktuelle Anbieterauswahl.")

    # canonical ordering for loops
    months_local = sorted(base_local["month_str"].dropna().unique().tolist())
    weekdays_order = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    weekdays_local = [
        w for w in weekdays_order
        if w in set(base_local["weekday_name"].dropna().unique())
    ]
    ranges_local = sorted(base_local["timerange"].dropna().unique().tolist())

    if not months_local or not weekdays_local or not ranges_local:
        mo.stop("Unvollständige Daten (Monat / Wochentag / Zeitfenster fehlen).")

    # --- global color domain across all frames (95%-Quantil) ---
    base_vals = base_local.copy()
    base_vals["mean"] = pd.to_numeric(base_vals["mean"], errors="coerce")

    vals_local = (
        base_vals
        .dropna(subset=["month_str", "weekday_name", "timerange", "bez_id"])
        .groupby(
            ["month_str", "weekday_name", "timerange", "bez_id", "bez_name"],
            as_index=False,
            dropna=False,
        )["mean"]
        .sum()
        .rename(columns={"mean": "value"})
    )

    if vals_local.empty:
        min_val_local, max_val_local = 0.0, 1.0
    else:
        min_val_local = 0.0
        p = 0.95
        max_val_local = float(vals_local["value"].quantile(p))
        if not np.isfinite(max_val_local) or max_val_local <= 0:
            max_val_local = float(vals_local["value"].max() or 1.0)

    # --- helper for filenames ---
    def slug_local(s: str | None) -> str:
        if not s:
            return "_"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())

    # label suffix for titles / filename
    title_bits_local = [provider_sel_local] if provider_sel_local != alle else ["Alle Anbieter"]
    title_suffix_local = " • ".join(title_bits_local)

    frames_local: list = []

    total_frames = len(months_local) * len(weekdays_local) * len(ranges_local)

    with mo.status.progress_bar(
        total=total_frames,
        title="Generiere GIF",
        subtitle="Initialisiere…",
    ) as bar:
        for m in months_local:
            for wd in weekdays_local:
                for tr in ranges_local:
                    bar.update(subtitle=f"{m} • {wd} • {tr}")

                    frame_df = base_local[
                        (base_local["month_str"] == m)
                        & (base_local["weekday_name"] == wd)
                        & (base_local["timerange"] == tr)
                    ]

                    if frame_df.empty:
                        continue

                    # Aggregate per Bezirk
                    values_local = (
                        frame_df.loc[:, ["bez_id", "bez_name", "mean"]]
                                .groupby(["bez_id", "bez_name"], as_index=False, dropna=False)
                                .agg(value=("mean", "sum"))
                    )

                    # join with geometries
                    agg_local = bezirke.merge(values_local, on=["bez_id", "bez_name"], how="left")
                    agg_local["value"] = pd.to_numeric(agg_local["value"], errors="coerce").fillna(0.0)

                    # build GeoJSON
                    features_local = []
                    for _, row_local in agg_local.iterrows():
                        geom_local = row_local["geometry"]
                        if isinstance(geom_local, str):
                            try:
                                geom_local = json.loads(geom_local)
                            except Exception:
                                continue
                        if not (
                            isinstance(geom_local, dict)
                            and "type" in geom_local
                            and "coordinates" in geom_local
                        ):
                            continue

                        features_local.append({
                            "type": "Feature",
                            "geometry": geom_local,
                            "properties": {
                                "bez_id": int(row_local["bez_id"]) if pd.notna(row_local["bez_id"]) else None,
                                "bez_name": row_local["bez_name"],
                                "value": float(row_local["value"]) if pd.notna(row_local["value"]) else 0.0,
                            },
                        })

                    if not features_local:
                        continue

                    geojson_frame = {"type": "FeatureCollection", "features": features_local}
                    data_gif = alt.InlineData(
                        values=geojson_frame,
                        format={"type": "json", "property": "features"},
                    )

                    title_local = f"∑ Verfügbarkeit — {m} • {wd} • {tr} — {title_suffix_local}"

                    chart_gif = (
                        alt.Chart(data_gif)
                        .mark_geoshape(stroke="white", strokeWidth=0.5)
                        .encode(
                            color=alt.Color(
                                "properties.value:Q",
                                title="∑ Verfügbarkeit",
                                scale=alt.Scale(
                                    domain=[min_val_local, max_val_local],
                                    clamp=True,
                                    nice=False,
                                    scheme="blues",
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("properties.bez_name:N", title="Bezirk"),
                                alt.Tooltip(
                                    "properties.value:Q",
                                    title="∑ Verfügbarkeit",
                                    format=".3f",
                                ),
                            ],
                        )
                        .project(type="mercator")
                        .properties(width=720, height=520, title=title_local)
                    )

                    png_bytes = vegalite_to_png(chart_gif.to_dict())
                    frames_local.append(iio.imread(io.BytesIO(png_bytes)))

    if not frames_local:
        mo.stop("Keine gültigen Frames erzeugt.")

    # --- output GIF ---
    out_dir_local = os.path.join(os.getcwd(), "data")
    os.makedirs(out_dir_local, exist_ok=True)

    fname_bits_local = ["zeitreihe_monat_wochentag_zeitfenster"]
    if provider_sel_local != alle:
        fname_bits_local.append(slug_local(provider_sel_local))
    fname_local = "_".join(fname_bits_local) + ".gif"
    out_path_local = os.path.join(out_dir_local, fname_local)

    iio.imwrite(out_path_local, frames_local, duration=1000, loop=0)

    preview = mo.image(out_path_local, caption=f"Zeitreihe ({len(frames_local)} Frames)")
    dl = mo.download(
        data=lambda: open(out_path_local, "rb"),
        filename=fname_local,
        mimetype="image/gif",
        label="GIF herunterladen",
    )
    mo.vstack([preview, dl], gap="0.75rem")
    return


if __name__ == "__main__":
    app.run()
