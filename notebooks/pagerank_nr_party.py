# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.7",
#     "networkx==3.5",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "requests==2.32.5",
# ]
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Panaschier-Flüsse: Blöcke, Netzwerke & PageRank

    Dieses Notebook:
    1) lädt den "Resultate der Nationalratswahlen"-Datensatz,  
    2) mappt Listen → politische **Blöcke**,  
    3) baut eine **Flussmatrix** \(F\) (Quelle → Ziel),  
    4) analysiert mit **Markow-Ketten / PageRank**, und  
    5) visualisiert **Sankey**, **Netzwerk**, **Heatmap** und exportiert **GEXF**.
    """
    )
    return


@app.cell
def _():
    import requests
    import logging
    import os
    import pandas as pd
    import numpy as np
    from numpy.linalg import norm
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    return Path, PathPatch, logging, norm, np, nx, os, pd, plt, requests


@app.cell
def _(logging, np):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Executing {__file__}...")
    np.set_printoptions(suppress=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Daten laden
    Wir laden den Datensatz und achten auf den korrekten CSV-Separator. Falls alles in **eine** Spalte fällt, bitte die Quelle prüfen und den Separator anpassen.
    """
    )
    return


@app.cell
def _(os, pd, requests):
    # helper function for reading datasets with proper separator
    def get_dataset(url):
        r = requests.get(url, params={"format": "csv", "timezone": "Europe%2FZurich"})
        data_path = os.path.join(os.getcwd(), "..", "data")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        csv_path = os.path.join(data_path, "100281.csv")
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
    df = get_dataset('https://data.bs.ch/explore/dataset/100281/download')
    df = df[df['wahlkreis_code'] == 1]
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Block-Definitionen & Quellspalten
    Wir fassen Listenvarianten zu **Blöcken** (FDP, LDP, SP, …) zusammen.  
    Die Quellspalten in jeder Zeile enthalten die **eingehenden Panachage-Stimmen** aus anderen Listen.
    """
    )
    return


@app.cell
def _():
    # Party bloc colors
    bloc_color = {
        "FDP":  "#1b87aa",
        "LDP":  "#0b5a73",
        "SP":   "#ce4631",
        "Mitte":"#df7c18",
        "BGB":  "#93a345",
        "GLP":  "#00988c",
        "SVP":  "#316641",
        "EVP":  "#e3b13e",
        "EDU":  "#ae49a2",
        "PdA":  "#ac004f",
        "VA":   "#7e7e8f",
        "MV":   "#610e59",
        "Ohne": "#BABABA"
    }

    # Columns that encode source-list panachage
    source_cols = [
        "01_fdp","03_ldp","04_evp","05_sp","09_edu","10_glp","12_svp","14_va",
        "18_jsvp","21_jgb","23_jlb","25_jglp","00_ohne",
        "06_jfdp","07_mitte","08_bgb","11_pda","17_basta_rm","20_juso","22_jmitte",
        "24_sp60","26_fdp","27_basta_ja","28_mv","30_svp_gew","31_svp60",
        "32_glp_k_u","33_glp_b","34_ldp_gew","35_glp_r_e","36_mitte60","37_glp_kmu","38_glp_i",
    ]

    # Map each source column to a bloc
    col2bloc = {
        # FDP
        "01_fdp":"FDP","06_jfdp":"FDP","26_fdp":"FDP",
        # LDP
        "03_ldp":"LDP", "23_jlb":"LDP", "34_ldp_gew":"LDP",
        # SP
        "05_sp":"SP","20_juso":"SP","24_sp60":"SP",
        # Mitte
        "07_mitte":"Mitte","36_mitte60":"Mitte","22_jmitte":"Mitte",
        # BGB 
        "08_bgb":"BGB","17_basta_rm":"BGB","27_basta_ja":"BGB","21_jgb":"BGB",
        # GLP
        "10_glp":"GLP","25_jglp":"GLP","32_glp_k_u":"GLP","33_glp_b":"GLP",
        "35_glp_r_e":"GLP","37_glp_kmu":"GLP","38_glp_i":"GLP",
        # SVP
        "12_svp":"SVP","18_jsvp":"SVP","30_svp_gew":"SVP","31_svp60":"SVP",
        # Solos
        "04_evp":"EVP", "09_edu":"EDU","11_pda":"PdA","14_va":"VA","28_mv":"MV",
        # without party
        "00_ohne": "Ohne",
    }
    return bloc_color, col2bloc, source_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ziel-Listen → Blöcke mappen
    Wir ordnen `parteikurzbezeichnung` je Zeile einem Block zu. Unbekannte Labels werden ausgelassen.
    """
    )
    return


@app.cell
def _(logging):
    # Map each row (destination list) to a bloc using parteikurzbezeichnung (robust to variants)
    def map_dest_bloc(parteikurz):
        s = str(parteikurz).strip().lower()
        if "fdp" in s: return "FDP"
        if s.startswith("ldp") or s.startswith("jlb"): return "LDP"
        if s.startswith("sp") or s.startswith("juso"): return "SP"
        if "mitte" in s: return "Mitte"
        if s == "bgb" or s.startswith("basta") or s == "jgb": return "BGB"
        if "glp" in s: return "GLP"
        if "svp" in s: return "SVP"
        if s == "evp": return "EVP"
        if s == "edu": return "EDU"
        if s == "pda": return "PdA"
        if s == "va":  return "VA"
        if s == "mv":  return "MV"
        logging.info(f"No entry found for Parteikutzbezeichnung: {parteikurz}")
        return None
    return (map_dest_bloc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Eingabespalten bereinigen & typisieren
    Fehlende `source_cols` werden als 0 ergänzt, numerisch gecastet; zusätzlich erzeugen wir `dest_bloc` pro Zeile.
    """
    )
    return


@app.cell
def _(df, map_dest_bloc, pd, source_cols):
    # Cast source vote columns to numeric
    for c in source_cols + ["kandidatenstimmen_unveranderte_wahlzettel"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Destination bloc per row
    df["dest_bloc"] = df["parteikurzbezeichnung"].apply(map_dest_bloc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Block-zu-Block-Flussmatrix

    Sei die geordnete Blockliste \(B\) mit \(|B|=n\).  
    Wir konstruieren \(F\in\mathbb{R}^{n\times n}\) mit

    \[
    F_{ij} \;=\; \text{Stimmenfluss vom Quellblock } B_i \text{ zum Zielblock } B_j .
    \]

    Jede Datensatzzeile entspricht einer Zielliste (→ \(B_j\)); deren Spalten (z. B. `01_fdp`, `10_glp`, …) sind **Quellen** und werden via `col2bloc` zu \(B_i\) aggregiert.  
    Die Diagonale \(F_{jj}\) enthält die unveränderten Wahlzettel der Zielliste.
    """
    )
    return


@app.cell
def _(col2bloc, df, logging, np, source_cols):
    blocs = ["Ohne","PdA","BGB","SP","EVP","GLP","Mitte","LDP","FDP","EDU","SVP","MV","VA"]
    bloc_index = {b:i for i,b in enumerate(blocs)}
    flow = np.zeros((len(blocs), len(blocs)), dtype=float)  # rows: source, cols: dest

    for _, r in df.iterrows():
        dest = r["dest_bloc"]
        j = bloc_index[dest]
        flow[j, j] += r["kandidatenstimmen_unveranderte_wahlzettel"]
        for col in source_cols:
            src_bloc = col2bloc.get(col)
            if src_bloc is None:
                logging.info(col)
                logging.info(src_bloc)
                continue
            i = bloc_index[src_bloc]
            flow[i, j] += r[col]

    flow
    return blocs, flow


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Sankey: Quellen → Ziele
    Wir zeigen die wichtigsten Flüsse; ein Mindestanteil filtert Kleinstflüsse, um die Lesbarkeit zu erhöhen.
    """
    )
    return


@app.cell
def _(Path, PathPatch, bloc_color, blocs, flow, np, plt):
    def plot_bipartite_sankey(flow, blocs, bloc_color, min_share=0.03, gap=0.02, figsize=(10,8)):
        """
        Draw a simple bipartite Sankey:
          left column: sources (sum of rows)
          right column: destinations (sum of cols)
          ribbons: flows with share >= min_share (share relative to source total)
        """
        F = flow.astype(float)
        src_tot = F.sum(axis=1)
        dst_tot = F.sum(axis=0)
        total = src_tot.sum()
        if total <= 0:
            raise ValueError("No flows to draw.")

        # vertical positions (stacked bars 0..1 with gaps)
        def stack_positions(weights):
            w = weights / weights.sum() if weights.sum() > 0 else np.zeros_like(weights)
            y0 = 0.02
            spans = []
            for frac in w:
                h = max(0.0, frac - gap)
                spans.append((y0, y0 + h))
                y0 += frac
            return spans

        src_spans = stack_positions(src_tot)
        dst_spans = stack_positions(dst_tot)

        # helpers to draw bezier ribbon from (x0, y0a..y0b) to (x1, y1a..y1b)
        def ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.35):
            verts = [
                (x0, y0a), (x0+0.2, y0a), (x1-0.2, y1a), (x1, y1a),
                (x1, y1b), (x1-0.2, y1b), (x0+0.2, y0b), (x0, y0b),
                (x0, y0a)
            ]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            patch = PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)
            ax.add_patch(patch)

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

        # draw source and destination bars
        for i, b in enumerate(blocs):
            y0, y1 = src_spans[i]
            ax.add_patch(plt.Rectangle((0.02, y0), 0.08, y1-y0, color=bloc_color.get(b, "#888")))
            ax.text(0.0, (y0+y1)/2, b, va="center", ha="right", fontsize=10, fontweight="bold")
        for j, b in enumerate(blocs):
            y0, y1 = dst_spans[j]
            ax.add_patch(plt.Rectangle((0.90, y0), 0.08, y1-y0, color=bloc_color.get(b, "#888")))
            ax.text(1.00, (y0+y1)/2, b, va="center", ha="left", fontsize=10, fontweight="bold")

        # Implement accumulators to avoid overlap
        src_acc = np.zeros(len(blocs))
        dst_acc = np.zeros(len(blocs))
        for i in range(len(blocs)):
            if src_tot[i] == 0: 
                continue
            for j in range(len(blocs)):
                if F[i, j] <= 0: 
                    continue
                share = F[i, j] / src_tot[i]
                if share < min_share: 
                    continue

                # source segment
                s_y0, s_y1 = src_spans[i]
                src_height_total = (s_y1 - s_y0)
                h = share * src_height_total
                y0a = s_y0 + src_acc[i]
                y0b = y0a + h
                src_acc[i] += h

                # destination segment (pack within destination bar)
                d_y0, d_y1 = dst_spans[j]
                dst_height_total = (d_y1 - d_y0)
                frac_dst = F[i, j] / dst_tot[j] if dst_tot[j] > 0 else 0
                hd = frac_dst * dst_height_total
                y1a = d_y0 + dst_acc[j]
                y1b = y1a + hd
                dst_acc[j] += hd

                ribbon(ax, 0.10, y0a, y0b, 0.90, y1a, y1b, color=bloc_color.get(blocs[i], "#888"))

        ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Panachage flows (bipartite Sankey): sources → destinations")
        plt.tight_layout()
        return fig

    # Usage
    fig_sankey = plot_bipartite_sankey(flow, blocs, bloc_color, min_share=0.03)
    fig_sankey
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Markow-Kette & PageRank auf Flüssen

    Zeilen-Normalisierung von \(F\) zur Übergangsmatrix \(P\):

    \[
    P_{ij} = \frac{F_{ij}}{\sum_k F_{ik}} \quad \text{(falls Zeilensumme \(\gt 0\))}
    \]

    Mit Dämpfung \(\alpha\) und Teleport-Vektor \(v\) iterieren wir:

    \[
    \pi^{(t+1)} = \alpha\, v + (1-\alpha)\,\pi^{(t)} P,
    \]

    bis zur Konvergenz. \(\pi\) bewertet Blöcke danach, wie stark sie von starken Quellen **angezeigt** werden (rekursives Prestige).
    """
    )
    return


@app.cell
def _(blocs, flow, norm, np, pd):
    # Row-normalize (source distribution) to Markov P
    row_sums = flow.sum(axis=1)
    P = np.zeros_like(flow, dtype=float)
    nonzero = row_sums > 0
    P[nonzero] = (flow[nonzero].T / row_sums[nonzero]).T

    alpha = 0.15
    n = len(blocs)
    v = np.ones(n) / n
    pi = np.ones(n) / n

    for _ in range(200):
        pi_new = alpha * v + (1 - alpha) * (pi @ P)
        if norm(pi_new - pi, 1) < 1e-12:
            break
        pi = pi_new

    pagerank = pd.DataFrame({"bloc": blocs, "pagerank": pi}).sort_values("pagerank", ascending=False)
    return (pagerank,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Netzwerkansicht (gefiltert)
    Pro Quelle behalten wir die **Top-k** ausgehenden Anteile oberhalb eines Mindestschwellwerts.  
    Knotengröße \(\propto\) PageRank; Kantenbreite \(\propto \sqrt{\text{Anteil}}\).
    """
    )
    return


@app.cell
def _(bloc_color, blocs, flow, np, nx, pagerank, plt):
    def plot_party_network(flow, blocs, pagerank_df, bloc_color,
                           top_k_per_src=3, min_share=0.04,
                           seed=42, figsize=(9,7),
                           layout="circle", rotate_deg=0):
        """
        Draw a cleaner directed network:
          - remove self loops
          - keep only top_k_per_src outgoing edges per source, and only if share >= min_share
          - edge width ~ sqrt(weight share), arrows visible
          - node size ~ PageRank
          - layouts: 'circle' (ordered ring), 'shell', 'kk' (kamada-kawai), 'spring'
        """
        # ---- shares per source (row-normalize) ----
        row_sums = flow.sum(axis=1, keepdims=True)
        shares = np.divide(flow, row_sums, out=np.zeros_like(flow, dtype=float), where=row_sums>0)

        # ---- build graph with pr attribute ----
        pr_map = dict(zip(pagerank_df["bloc"], pagerank_df["pagerank"]))
        G = nx.DiGraph()
        for b in blocs:
            G.add_node(b, pr=float(pr_map.get(b, 0.0)))

        # ---- add pruned edges ----
        for i, src in enumerate(blocs):
            # candidate outgoing edges (exclude self)
            candidates = [(j, shares[i, j]) for j in range(len(blocs)) if j != i and shares[i, j] > 0]
            candidates.sort(key=lambda x: x[1], reverse=True)
            for j, sh in candidates[:top_k_per_src]:
                if sh >= min_share:
                    G.add_edge(src, blocs[j], weight=float(flow[i, j]), share=float(sh))

        # ---- layouts ----
        def _circular_pos(nodes_in_order, rotation_deg=0):
            n = len(nodes_in_order)
            if n == 0:
                return {}
            # angles go CCW from +x; rotate by rotation_deg
            theta0 = np.deg2rad(rotation_deg)
            angles = theta0 + np.linspace(0, 2*np.pi, n, endpoint=False)
            pos = {node: np.array([np.cos(a), np.sin(a)]) for node, a in zip(nodes_in_order, angles)}
            return pos

        if layout == "circle":
            pos = _circular_pos(blocs, rotate_deg)
        elif layout == "shell":
            # split by PageRank into core (top 25%) and outer ring
            pr_vals = nx.get_node_attributes(G, "pr")
            if pr_vals:
                thresh = np.percentile(list(pr_vals.values()), 75)
                core = [n for n, p in pr_vals.items() if p >= thresh]
                rim  = [n for n in blocs if n not in core]
            else:
                core, rim = blocs[:max(1, len(blocs)//4)], blocs[max(1, len(blocs)//4):]
            pos = nx.shell_layout(G, nlist=[core, rim])
        elif layout == "kk":
            pos = nx.kamada_kawai_layout(G, weight="weight")
        else:  # "spring"
            pos = nx.spring_layout(G, seed=seed, weight="weight")

        # ---- draw ----
        fig = plt.figure(figsize=figsize)

        # edges (curved to reduce overlap on circular/shell)
        ew = [max(1.0, 8.0 * (G[u][v]["share"] ** 0.5)) for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos, width=ew, alpha=0.35,
            arrows=True, arrowstyle="-|>", arrowsize=14,
            connectionstyle="arc3,rad=0.25"
        )

        # nodes
        ns = [3500 * max(0.001, G.nodes[n]["pr"]) for n in G.nodes()]  # tiny floor avoids invisibility
        nc = [bloc_color.get(n, "#888888") for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=ns, node_color=nc,
                               linewidths=0.8, edgecolors="#333333")

        # labels: push outward slightly (works best for circle/shell; harmless otherwise)
        lbl_pos = {}
        for n, (x, y) in pos.items():
            r = np.hypot(x, y)
            if r == 0:
                lbl_pos[n] = (x, y)
            else:
                lbl_pos[n] = (x * 1.10, y * 1.10)
        nx.draw_networkx_labels(G, lbl_pos, font_size=10, font_weight="bold")

        plt.title("Panachage flows between blocs (node size = PageRank; top edges only)")
        plt.axis("off")
        plt.tight_layout()
        return fig, G

    fig_net, G = plot_party_network(flow, blocs, pagerank, bloc_color,
                                    top_k_per_src=1, min_share=0,
                                    layout="circle", rotate_deg=90)
    fig_net
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## PageRank-Balkendiagramm
    Schneller Überblick über die Rangfolge (höchste Blöcke oben).
    """
    )
    return


@app.cell
def _(bloc_color, pagerank):
    def plot_pagerank_bar(pagerank_df, bloc_color, top_n=12, figsize=(7,5)):
        import matplotlib.pyplot as plt
        top = pagerank_df.sort_values("pagerank", ascending=True).tail(top_n)
        colors = [bloc_color.get(b, "#888") for b in top["bloc"]]
        fig = plt.figure(figsize=figsize)
        plt.barh(top["bloc"], top["pagerank"], color=colors)
        plt.xlabel("PageRank"); plt.title("Top blocs by PageRank")
        plt.tight_layout()
        return fig

    fig_pr = plot_pagerank_bar(pagerank, bloc_color)
    fig_pr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Flow-Heatmap
    Je nach Normalisierung zeigt die Heatmap **Anteile pro Quelle** (Zeilen), **Anteile pro Ziel** (Spalten) oder **Rohzahlen**.
    """
    )
    return


@app.cell
def _(blocs, flow):
    def plot_flow_heatmap(flow, blocs, normalize="row", annotate=True, figsize=(8,6)):
        """
        normalize: "row" (shares per source), "col" (per destination), or None (raw counts)
        """
        import numpy as np, matplotlib.pyplot as plt
        M = flow.astype(float).copy()
        if normalize == "row":
            s = M.sum(axis=1, keepdims=True); M = np.divide(M, s, out=np.zeros_like(M), where=s>0)
        elif normalize == "col":
            s = M.sum(axis=0, keepdims=True); M = np.divide(M, s, out=np.zeros_like(M), where=s>0)

        fig = plt.figure(figsize=figsize)
        im = plt.imshow(M, aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(blocs)), blocs, rotation=45, ha="right")
        plt.yticks(range(len(blocs)), blocs)
        title = "Flow heatmap"
        if normalize == "row":  title += " (share per source)"
        if normalize == "col":  title += " (share per destination)"
        plt.title(title)
        if annotate and len(blocs) <= 16:
            for i in range(len(blocs)):
                for j in range(len(blocs)):
                    val = M[i, j]
                    if val > 0:
                        txt = f"{val:.2f}" if normalize else f"{int(flow[i,j])}"
                        plt.text(j, i, txt, ha="center", va="center", fontsize=7)
        plt.tight_layout()
        return fig

    fig_hm = plot_flow_heatmap(flow, blocs, normalize="row")
    fig_hm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Export nach Gephi (GEXF)
    Wir exportieren ein gerichtetes Netzwerk incl. Blockfarbe und PageRank-Attributen für externe Analysen.
    """
    )
    return


@app.cell
def _(bloc_color, blocs, flow, pagerank):
    def save_gexf_from_flow(flow, blocs, pagerank_df, bloc_color,
                            path="party_flows.gexf",
                            include_self=False, min_weight=0):
        """Export directed graph with bloc attributes for Gephi."""
        import networkx as nx
        G = nx.DiGraph()
        pr_map = dict(zip(pagerank_df["bloc"], pagerank_df["pagerank"]))
        for b in blocs:
            G.add_node(
                b,
                pagerank=float(pr_map.get(b, 0.0)),
                bloc=b,
                color=bloc_color.get(b, "#888888")
            )
        for i, src in enumerate(blocs):
            for j, dst in enumerate(blocs):
                w = float(flow[i, j])
                if (i == j and not include_self) or w <= min_weight:
                    continue
                if w > 0:
                    G.add_edge(src, dst, weight=w)
        nx.write_gexf(G, path)
        return path

    gexf_path = save_gexf_from_flow(flow, blocs, pagerank, bloc_color,
                                    path="data/party_flows.gexf")
    print("Saved:", gexf_path)
    return


if __name__ == "__main__":
    app.run()
