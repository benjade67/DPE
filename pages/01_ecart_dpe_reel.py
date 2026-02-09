import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.loaders import load_df_corr, load_dep_geojson

# ==========================
# Paramètres colonnes (adapte si nécessaire)
# ==========================
COL_REAL = "conso_m2_reelle"
COL_DPE  = "conso_5_usages_par_m2_ef_moy"
COL_DEP  = "code_departement"

st.title("1️⃣ Écart consommation par étiquette énergétique : ENEDIS réelle − DPE conventionnelle")

# ==========================
# Helpers
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

@st.cache_data(show_spinner=False)
def get_df_corr():
    return load_df_corr()

def compute_result(df: pd.DataFrame, filters: dict):
    missing = [c for c in [COL_REAL, COL_DPE] if c not in df.columns]
    if missing:
        return {"error": f"Colonnes manquantes dans df_corr.parquet : {missing}"}

    dff = df.dropna(subset=[COL_REAL, COL_DPE]).copy()
    dff["ecart_dpe_reel"] = dff[COL_REAL] - dff[COL_DPE]
    dff["etiquette_dpe_calc"] = dff[COL_DPE].apply(dpe_label_from_conso)

    # Filtres
    if "chauffage_elec" in dff.columns and filters["chauffage"] != "Tous":
        dff = dff[dff["chauffage_elec"] == (1 if filters["chauffage"] == "Électrique" else 0)]

    if "ecs_elec" in dff.columns and filters["ecs"] != "Tous":
        dff = dff[dff["ecs_elec"] == (1 if filters["ecs"] == "Électrique" else 0)]

    if "zone_climatique_mode" in dff.columns and filters["zone"] != "Toutes":
        dff = dff[dff["zone_climatique_mode"] == filters["zone"]]

    if "annee_construction" in dff.columns:
        y0, y1 = filters["yr"]
        dff = dff[dff["annee_construction"].between(y0, y1)]

    if "n_dpe" in dff.columns:
        dff = dff[dff["n_dpe"] >= filters["ndpe_min"]]

    if dff.empty:
        return {"warning": "Aucune donnée après filtrage."}

    # Analyse
    COL_ETIQ = "etiquette_dpe_calc"
    tmp = dff.dropna(subset=[COL_ETIQ, "ecart_dpe_reel"]).copy()
    if tmp.empty:
        return {"warning": "Aucune donnée exploitable pour l’analyse par étiquette."}

    lo2, hi2 = np.percentile(tmp["ecart_dpe_reel"], [1, 99])
    tmp["ecart_clip"] = tmp["ecart_dpe_reel"].clip(lo2, hi2)

    counts = tmp[COL_ETIQ].value_counts().to_dict()
    tmp["etiquette_label"] = tmp[COL_ETIQ].apply(
        lambda x: f"{x} (n={counts.get(x, 0):,})".replace(",", " ")
    )

    order_labels = [
        f"{k} (n={counts[k]:,})".replace(",", " ")
        for k in list("ABCDEFG")
        if k in counts
    ]

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

    return {
        "fig": fig_box,
        "info": (
            "Plus l’étiquette DPE est mauvaise, plus la consommation conventionnelle du DPE "
            "tend à surestimer la consommation électrique réellement observée."
        ),
        "n": len(tmp),
    }

# ==========================
# 1) Charger df au chargement (cache) + préparer options une fois
# ==========================
df = get_df_corr()

# ==========================
# 2) Définir filtres par défaut (au tout premier run)
# ==========================
if "page1_filters_applied" not in st.session_state:
    # valeurs par défaut comme dans ton code original (index=1 => "Électrique")
    default_filters = {
        "chauffage": "Électrique",
        "ecs": "Électrique",
        "zone": "Toutes",
        "yr": None,          # rempli ci-dessous selon les données
        "ndpe_min": 1,
    }

    # bornes année depuis df (si dispo)
    if "annee_construction" in df.columns and df["annee_construction"].notna().any():
        ymin = int(np.nanpercentile(df["annee_construction"], 1))
        ymax = int(np.nanpercentile(df["annee_construction"], 99))
        default_filters["yr"] = (ymin, ymax)
    else:
        default_filters["yr"] = (1900, 2025)

    st.session_state["page1_filters_applied"] = default_filters

    # calcul initial (valeurs par défaut)
    with st.spinner("Chargement des données (valeurs par défaut)..."):
        st.session_state["page1_result"] = compute_result(df, default_filters)

# ==========================
# 3) Sidebar : filtres modifiables mais appliqués seulement via bouton
# ==========================
st.sidebar.header("Filtres")

# options zone depuis df (maintenant qu'on l'a)
if "zone_climatique_mode" in df.columns:
    zones = ["Toutes"] + sorted(df["zone_climatique_mode"].dropna().unique().tolist())
else:
    zones = ["Toutes"]

# bornes année depuis df (pour slider)
if "annee_construction" in df.columns and df["annee_construction"].notna().any():
    ymin = int(np.nanpercentile(df["annee_construction"], 1))
    ymax = int(np.nanpercentile(df["annee_construction"], 99))
else:
    ymin, ymax = 1800, 2025

# bornes ndpe (sécurisé)
ndpe_max = int(df["n_dpe"].max()) if "n_dpe" in df.columns and pd.notna(df["n_dpe"].max()) else 100

applied = st.session_state["page1_filters_applied"]

with st.sidebar.form("page1_filters_form"):
    chauffage = st.selectbox("Chauffage", ["Tous", "Électrique", "Non électrique"],
                             index=["Tous", "Électrique", "Non électrique"].index(applied["chauffage"]))
    ecs = st.selectbox("ECS", ["Tous", "Électrique", "Non électrique"],
                       index=["Tous", "Électrique", "Non électrique"].index(applied["ecs"]))

    zone = st.selectbox("Zone climatique", zones, index=zones.index(applied["zone"]) if applied["zone"] in zones else 0)

    yr = st.slider("Année de construction", ymin, ymax, applied["yr"])

    ndpe_min = st.slider("n_dpe minimum", 1, ndpe_max, int(applied["ndpe_min"]))

    submitted = st.form_submit_button("Appliquer")

if submitted:
    new_filters = {
        "chauffage": chauffage,
        "ecs": ecs,
        "zone": zone,
        "yr": yr,
        "ndpe_min": int(ndpe_min),
    }
    st.session_state["page1_filters_applied"] = new_filters
    with st.spinner("Application des filtres..."):
        st.session_state["page1_result"] = compute_result(df, new_filters)

# ==========================
# 4) Affichage : toujours le dernier résultat appliqué
# ==========================
result = st.session_state.get("page1_result")

if result is None:
    st.info("Aucun résultat en mémoire.")
else:
    if "error" in result:
        st.error(result["error"])
    elif "warning" in result:
        st.warning(result["warning"])
    else:
        st.plotly_chart(result["fig"], use_container_width=True)
        st.info(result["info"])
