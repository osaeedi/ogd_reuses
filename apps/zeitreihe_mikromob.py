# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "altair==5.5.0",
#   "marimo>=0.7.11",
#   "pandas==2.2.3",
#   "requests==2.32.5",
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
    return (get_dataset,)


@app.cell
def _(get_dataset):
    # Read the dataset
    raw = get_dataset(dataset_id="100416")
    raw
    return (raw,)


@app.cell
def _(pd, raw):
    # Standardisiere Spaltennamen → maschinenfreundlich
    rename_map = {
        "Datum": "datum",
        "Gemeinde": "gemeinde",
        "Wohnviertel-ID": "wohnviertel_id",
        "Wohnviertel": "wohnviertel",
        "Bezirks-ID": "bezirks_id",
        "Bezirk": "bezirk",
        "geometry": "geometry",
        "Anbieter Name": "anbieter",
        "Fahrzeugtyp": "fahrzeugtyp",
        "Formfaktor": "bauweise",  # "Bauweise" (Formfaktor)
        "Antriebstyp": "antriebsart",
        "Reichweite bei vollem Akku": "reichweite",
        "Anzahl Messungen": "anzahl_messungen",
        "Durchschnittliche Verfügbarkeit": "durchschn_verf",
        "Minimale Verfügbarkeit": "min_verf",
        "Maximale Verfügbarkeit": "max_verf",
        "geo_point_2d": "geo_point_2d",
    }
    df = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns}).copy()

    # Typen
    if not df.empty:
        # Datum
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
        # Numerik
        for c in ["reichweite", "anzahl_messungen", "durchschn_verf", "min_verf", "max_verf"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # IDs as int if possible
        for c in ["wohnviertel_id", "bezirks_id"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        # Trim strings
        for c in ["gemeinde","wohnviertel","bezirk","anbieter","fahrzeugtyp","bauweise","antriebsart"]:
            if c in df.columns:
                df[c] = df[c].astype("string").str.strip()
    df
    return (df,)


@app.cell
def _(df, mo):
    if df.empty:
        mo.stop("Keine Daten geladen.")
    # verfügbare Tage
    dates = (
        df.loc[df["datum"].notna(), "datum"]
        .dt.normalize()
        .sort_values()
        .unique()
    )
    if len(dates) == 0:
        mo.stop("Spalte 'Datum' leer/ungültig.")
    # Dropdown-Defaults (werden dynamisch weiter unten gebunden)
    alle = "Alle"
    return alle, dates


@app.cell
def _(dates, mo, pd):
    # Tages-Slider über Index (robust für nicht-kontinuierliche Daten)
    date_index = mo.ui.slider(
        start=0, stop=len(dates)-1, step=1, value=len(dates)-1,
        show_value=True, label="Tag wählen (Index)"
    )
    mo.md(f"**Ausgewähltes Datum:** {pd.to_datetime(dates[date_index.value]).date()}")
    date_index
    return (date_index,)


@app.cell
def _(alle, df, mo, pd):
    # Helper zum Bauen von Dropdowns mit "Alle"
    def opt(values):
        uniq = pd.Series(sorted(v for v in pd.unique(values) if pd.notna(v))).tolist()
        return [alle] + uniq

    # Platzhalter-Werte (werden dynamisch im nächsten Cell gesetzt)
    dd_fahrzeugtyp = mo.ui.dropdown(opt(df["fahrzeugtyp"]) if "fahrzeugtyp" in df else [alle], label="Fahrzeugtyp")
    dd_bauweise    = mo.ui.dropdown(opt(df["bauweise"]) if "bauweise" in df else [alle], label="Bauweise")
    dd_antrieb     = mo.ui.dropdown(opt(df["antriebsart"]) if "antriebsart" in df else [alle], label="Antriebsart")
    dd_reichweite  = mo.ui.dropdown(opt(df["reichweite"]) if "reichweite" in df else [alle], label="Reichweite (voller Akku)")
    mo.hstack([dd_fahrzeugtyp, dd_bauweise, dd_antrieb, dd_reichweite], justify="start", gap="1.0rem")
    return dd_antrieb, dd_bauweise, dd_fahrzeugtyp, dd_reichweite, opt


@app.cell
def _(
    alle,
    dd_antrieb,
    dd_bauweise,
    dd_fahrzeugtyp,
    dd_reichweite,
    df,
    mo,
    opt,
    pd,
):
    # Abhängige Optionen: basierend auf aktueller Wahl der anderen Dropdowns
    def _apply_filter(_df):
        if dd_fahrzeugtyp.value != alle:
            _df = _df[_df["fahrzeugtyp"] == dd_fahrzeugtyp.value]
        if dd_bauweise.value != alle:
            _df = _df[_df["bauweise"] == dd_bauweise.value]
        if dd_antrieb.value != alle:
            _df = _df[_df["antriebsart"] == dd_antrieb.value]
        if dd_reichweite.value != alle:
            # reichweite ist numerisch → exakte Übereinstimmung
            _df = _df[_df["reichweite"] == pd.to_numeric(dd_reichweite.value, errors="coerce")]
        return _df

    filtered_for_options = _apply_filter(df)

    # Neue Optionslisten abhängig vom Filter der jeweils anderen
    dd_fahrzeugtyp.options = opt(filtered_for_options["fahrzeugtyp"])
    dd_bauweise.options    = opt(filtered_for_options["bauweise"])
    dd_antrieb.options     = opt(filtered_for_options["antriebsart"])
    dd_reichweite.options  = opt(filtered_for_options["reichweite"])

    mo.callout("""Dropdown-Optionen werden dynamisch gefiltert. 
    **Hinweis:** Auswahl *Alle* summiert die Verfügbarkeiten.""", kind="info")
    return


@app.cell
def _(
    alle,
    date_index,
    dd_antrieb,
    dd_bauweise,
    dd_fahrzeugtyp,
    dd_reichweite,
    df,
    mo,
    pd,
):
    # Daten für den gewählten Tag + Filter vorbereiten
    chosen_date = pd.to_datetime(df["datum"].dt.normalize().unique()[date_index.value])
    day_df = df[df["datum"].dt.normalize() == chosen_date].copy()

    if dd_fahrzeugtyp.value != alle:
        day_df = day_df[day_df["fahrzeugtyp"] == dd_fahrzeugtyp.value]
    if dd_bauweise.value != alle:
        day_df = day_df[day_df["bauweise"] == dd_bauweise.value]
    if dd_antrieb.value != alle:
        day_df = day_df[day_df["antriebsart"] == dd_antrieb.value]
    if dd_reichweite.value != alle:
        day_df = day_df[day_df["reichweite"] == pd.to_numeric(dd_reichweite.value, errors="coerce")]

    # Aggregation pro Wohnviertel (Geometrie + Summe Verfügbarkeit)
    # Falls Geometrie mehrfach vorkommt, wird ∑ der 'durchschn_verf' gebildet.
    group_cols = ["wohnviertel_id", "wohnviertel", "geometry"]
    agg = (
        day_df[group_cols + ["durchschn_verf"]]
        .dropna(subset=["geometry"])
        .groupby(group_cols, dropna=False, as_index=False)
        .agg(value=("durchschn_verf", "sum"))
    )

    if agg.empty:
        mo.warning("Keine Daten für die aktuelle Auswahl.")
    agg
    return agg, chosen_date


@app.cell
def _(agg, json, pd):
    # Erzeuge GeoJSON FeatureCollection
    features = []
    for _, row in agg.iterrows():
        try:
            geom = row["geometry"]
            if isinstance(geom, str):
                geom = json.loads(geom)
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "wohnviertel_id": int(row["wohnviertel_id"]) if pd.notna(row["wohnviertel_id"]) else None,
                    "wohnviertel": row["wohnviertel"],
                    "value": float(row["value"]) if pd.notna(row["value"]) else None,
                },
            })
        except Exception as e:
            # Invalid geometry; skip
            continue
    geojson = {"type": "FeatureCollection", "features": features}
    geojson
    return (geojson,)


@app.cell
def _(agg, alt, geojson, mo):
    if not agg.empty and len(geojson["features"]) > 0:
        chart = (
            alt.Chart(alt.Data(values=geojson))
            .mark_geoshape(stroke="white", strokeWidth=0.5)
            .encode(
                color=alt.Color("properties.value:Q", title="∑ Verfügbarkeit"),
                tooltip=[
                    alt.Tooltip("properties.wohnviertel:N", title="Wohnviertel"),
                    alt.Tooltip("properties.value:Q", title="∑ Verfügbarkeit", format=".3f"),
                ],
            )
            .project(type="mercator")
            .properties(width=720, height=520)
        )
        mo.ui.altair_chart(chart)
    else:
        mo.warning("Keine darstellbaren Geometrien.")
    return


@app.cell
def _(
    chosen_date,
    dd_antrieb,
    dd_bauweise,
    dd_fahrzeugtyp,
    dd_reichweite,
    mo,
    pd,
):
    mo.md(f"""
    **Aktuelle Filter** — Datum: `{pd.to_datetime(chosen_date).date()}` · "
        f"Fahrzeugtyp: `{dd_fahrzeugtyp.value}` · Bauweise: `{dd_bauweise.value}` · "
        f"Antriebsart: `{dd_antrieb.value}` · Reichweite: `{dd_reichweite.value}`
    """)
    return


if __name__ == "__main__":
    app.run()
