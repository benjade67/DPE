import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.loaders import load_df_corr, load_dep_geojson

# ==========================
# Paramètres colonnes (adapte si nécessaire)
# ==========================
COL_REAL = "conso_m2_reelle"
COL_DPE  = "conso_5_usages_par_m2_ef_moy"   # ta conso DPE conventionnelle agrégée
COL_DEP  = "code_departement"

# ==========================
# Load
# ==========================
df = load_df_corr()

missing = [c for c in [COL_REAL, COL_DPE] if c not in df.columns]
if missing:
    st.error(f"Colonnes manquantes dans df_corr.parquet : {missing}")
    st.stop()

# Base exploitable
dff = df.dropna(subset=[COL_REAL, COL_DPE]).copy()
dff["ecart_dpe_reel"] = dff[COL_REAL] - dff[COL_DPE]

# ==========================
# Reconstruction étiquette DPE depuis la conso conventionnelle
# ==========================
def dpe_label_from_conso(x):
    if pd.isna(x):
        return np.nan
    if x <= 70:
        return "A"
    elif x <= 110:
        return "B"
    elif x <= 180:
        return "C"
    elif x <= 250:
        return "D"
    elif x <= 330:
        return "E"
    elif x <= 420:
        return "F"
    else:
        return "G"

dff["etiquette_dpe_calc"] = dff[COL_DPE].apply(dpe_label_from_conso)

st.title("1️⃣ Écart consommation par étiquette énergétique : ENEDIS réelle − DPE conventionnelle")

# ==========================
# Sidebar filtres
# ==========================
st.sidebar.header("Filtres")

if "chauffage_elec" in dff.columns:
    chauffage = st.sidebar.selectbox("Chauffage", ["Tous", "Électrique", "Non électrique"], 1)
    if chauffage == "Électrique":
        dff = dff[dff["chauffage_elec"] == 1]
    elif chauffage == "Non électrique":
        dff = dff[dff["chauffage_elec"] == 0]

if "ecs_elec" in dff.columns:
    ecs = st.sidebar.selectbox("ECS", ["Tous", "Électrique", "Non électrique"], 1)
    if ecs == "Électrique":
        dff = dff[dff["ecs_elec"] == 1]
    elif ecs == "Non électrique":
        dff = dff[dff["ecs_elec"] == 0]

if "zone_climatique_mode" in dff.columns:
    zones = ["Toutes"] + sorted(dff["zone_climatique_mode"].dropna().unique().tolist())
    zone = st.sidebar.selectbox("Zone climatique", zones, 0)
    if zone != "Toutes":
        dff = dff[dff["zone_climatique_mode"] == zone]

if "annee_construction" in dff.columns:
    ymin = int(np.nanpercentile(dff["annee_construction"], 1))
    ymax = int(np.nanpercentile(dff["annee_construction"], 99))
    yr = st.sidebar.slider("Année de construction", ymin, ymax, (ymin, ymax))
    dff = dff[dff["annee_construction"].between(yr[0], yr[1])]

if "n_dpe" in dff.columns:
    ndpe_min = st.sidebar.slider("n_dpe minimum", 1, int(dff["n_dpe"].max()), 3)
    dff = dff[dff["n_dpe"] >= ndpe_min]

if len(dff) == 0:
    st.warning("Aucune donnée après filtrage.")
    st.stop()


# ==========================
# Écart par étiquette DPE (reconstruite)
# ==========================
COL_ETIQ = "etiquette_dpe_calc"

tmp = dff.dropna(subset=[COL_ETIQ, "ecart_dpe_reel"]).copy()

if tmp.empty:
    st.warning("Aucune donnée exploitable pour l’analyse par étiquette.")
else:
    # Bornage pour lisibilité (1%–99%)
    lo2, hi2 = np.percentile(tmp["ecart_dpe_reel"], [1, 99])
    tmp["ecart_clip"] = tmp["ecart_dpe_reel"].clip(lo2, hi2)

    # Compter les volumes et les afficher dans l’axe X
    counts = tmp[COL_ETIQ].value_counts().to_dict()

    tmp["etiquette_label"] = tmp[COL_ETIQ].apply(
        lambda x: f"{x} (n={counts.get(x, 0):,})".replace(",", " ")
    )

    # Ordre A→G (uniquement celles présentes)
    order_labels = [
        f"{k} (n={counts[k]:,})".replace(",", " ")
        for k in list("ABCDEFG")
        if k in counts
    ]

    # Boxplot épuré
    fig_box = px.box(
        tmp,
        x="etiquette_label",
        y="ecart_clip",
        category_orders={"etiquette_label": order_labels},
        points=False,
        labels={
            "etiquette_label": "Étiquette DPE (volume après filtres)",
            "ecart_clip": "Enedis − DPE (kWh/m²/an)",
        },
        title="Distribution de l’écart par étiquette DPE",
    )
    fig_box.update_layout(height=420, margin={"r": 0, "t": 50, "l": 0, "b": 0})
    st.plotly_chart(fig_box, use_container_width=True)

    st.info(
        "Plus l’étiquette DPE est mauvaise, plus la consommation conventionnelle du DPE "
        "tend à surestimer la consommation électrique réellement observée."
    )
