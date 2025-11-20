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
        data = pd.read_csv(url, sep=";", on_bad_lines="warn", encoding_errors="ignore", low_memory=False)
        return data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    # Geteilte Mikromobilität nach Bezirk, Wochentagesabschnitt und Monat
    """)
    return


@app.cell
def _(os, pd):
    # Daten laden (lokal oder via get_dataset)
    # df = get_dataset(dataset_id="100428")
    df = pd.read_csv(os.path.join("data/100428.csv"), sep=";")
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
                color=alt.Color("properties.value:Q", title="∑ Verfügbarkeit"),
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


if __name__ == "__main__":
    app.run()
