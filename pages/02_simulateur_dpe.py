import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import shap
import matplotlib.pyplot as plt

from utils.loaders import load_model, load_defaults
from utils.shap_utils import build_shap_tools, transform_for_shap


st.title("2️⃣ Simulateur — Prédiction consommation électrique via les données du DPE")

defaults = load_defaults()

st.write("✅ Page chargée. Prêt à charger le modèle…")

# =========================================================
# Chargement modèle & defaults
# =========================================================
with st.spinner("Chargement du modèle…"):
    try:
        model = load_model()   # <-- IMPORTANT: ici seulement
        st.success("✅ Modèle chargé")
    except Exception as e:
        st.error("❌ Erreur au chargement du modèle")
        st.exception(e)
        st.stop()


# SHAP tools (cache dans utils/shap_utils.py)
preprocess, explainer, feature_names = build_shap_tools(model)


# =========================================================
# UI
# =========================================================
st.caption(
    "Le simulateur estime une consommation électrique attendue (kWh/m²/an) à partir des caractéristiques décrites par le DPE, "
    "corrigée à partir des consommations réelles observées. "
    "La section SHAP explique ensuite *pourquoi* le modèle arrive à cette valeur."
)

mode = st.radio("Mode de saisie", ["Simple", "Avancé"], horizontal=True)
st.divider()

# =========================================================
# Entrées utilisateur
# =========================================================
st.subheader("Caractéristiques principales")

c1, c2, c3, c4 = st.columns(4)

with c1:
    chauffage_elec = st.selectbox(
        "Chauffage électrique",
        [0, 1],
        index=0,
        format_func=lambda x: "Oui" if x == 1 else "Non",
    )
with c2:
    ecs_elec = st.selectbox(
        "ECS électrique",
        [0, 1],
        index=0,
        format_func=lambda x: "Oui" if x == 1 else "Non",
    )
with c3:
    zone = st.selectbox(
        "Zone climatique",
        ["H1a", "H1b", "H1c", "H2a", "H2b", "H2c", "H2d", "H3"],
        index=1,
    )
with c4:
    annee = st.slider(
        "Année de construction",
        1850,
        2025,
        1974,
    )

c5, c6, c7 = st.columns(3)

with c5:
    n_logements = st.number_input(
        "Nombre de logements",
        min_value=1,
        max_value=5000,
        value=14,
        step=1,
    )
with c6:
    surface_totale = st.number_input(
        "Surface totale (m²)",
        min_value=20,
        max_value=1_500_000,
        value=928,
        step=10,
    )
with c7:
    n_dpe = st.number_input(
        "Nombre de DPE agrégés (n_dpe)",
        min_value=1,
        max_value=20000,
        value=4,
        step=1,
    )

type_ecs = st.selectbox("Type d'installation ECS", ["individuel", "collectif"], index=1)
type_chauffage = st.selectbox(
    "Type d'installation chauffage",
    ["individuel", "collectif", "mixte (collectif-individuel)"],
    index=1,
)
type_bat = st.selectbox("Type de bâtiment", ["habitation", "appartement", "maison"], index=1)

st.divider()

# =========================================================
# Paramètres thermiques (avancé)
# =========================================================
if mode == "Avancé":
    st.subheader("Paramètres thermiques (avancé)")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        ubat = st.slider(
            "UBAT (W/m².K)",
            0.0,
            12.0,
            0.55,
            step=0.05,
        )
    with a2:
        dep_air = st.slider(
            "Déperditions renouvellement d'air",
            0.0,
            200.0,
            48.7,
            step=1.0,
        )
    with a3:
        dep_baies = st.slider(
            "Déperditions baies vitrées",
            0.0,
            200.0,
            31.5,
            step=1.0,
        )
    with a4:
        dep_murs = st.slider(
            "Déperditions murs",
            0.0,
            200.0,
            14.7,
            step=1.0,
        )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        dep_pt = st.slider(
            "Déperditions ponts thermiques",
            0.0,
            200.0,
            30.9,
            step=1.0,
        )
    with b2:
        dep_portes = st.slider(
            "Déperditions portes",
            0.0,
            200.0,
            6.7,
            step=1.0,
        )
    with b3:
        dep_pb = st.slider(
            "Déperditions planchers bas",
            0.0,
            200.0,
            0.0,
            step=1.0,
        )
    with b4:
        dep_ph = st.slider(
            "Déperditions planchers hauts",
            0.0,
            200.0,
            51.2,
            step=1.0,
        )
else:
    ubat = float(defaults.get("ubat_w_par_m2_k_moy", 1.2))
    dep_air = float(defaults.get("deperditions_renouvellement_air_moy", 10.0))
    dep_baies = float(defaults.get("deperditions_baies_vitrees_moy", 15.0))
    dep_murs = float(defaults.get("deperditions_murs_moy", 10.0))
    dep_pt = float(defaults.get("deperditions_ponts_thermiques_moy", 5.0))
    dep_portes = float(defaults.get("deperditions_portes_moy", 2.0))
    dep_pb = float(defaults.get("deperditions_planchers_bas_moy", 8.0))
    dep_ph = float(defaults.get("deperditions_planchers_hauts_moy", 8.0))


# =========================================================
# Construction X_user (doit matcher les features du modèle)
# =========================================================
X_user = pd.DataFrame([{
    "nombre_de_logements": float(n_logements),
    "n_dpe": float(n_dpe),
    "surface_totale": float(surface_totale),
    "annee_construction": float(annee),
    "ubat_w_par_m2_k_moy": float(ubat),
    "deperditions_renouvellement_air_moy": float(dep_air),
    "deperditions_baies_vitrees_moy": float(dep_baies),
    "deperditions_murs_moy": float(dep_murs),
    "deperditions_ponts_thermiques_moy": float(dep_pt),
    "deperditions_portes_moy": float(dep_portes),
    "deperditions_planchers_bas_moy": float(dep_pb),
    "deperditions_planchers_hauts_moy": float(dep_ph),
    "chauffage_elec": int(chauffage_elec),
    "ecs_elec": int(ecs_elec),

    "zone_climatique_mode": zone,
    "type_batiment_mode": type_bat,
    "type_installation_chauffage_mode": type_chauffage,
    "type_installation_ecs_mode": type_ecs,
}])


# =========================================================
# Prédiction
# =========================================================
y_pred = float(model.predict(X_user)[0])

ref = defaults.get("conso_ref", None)
ref_txt = f"{float(ref):.1f} kWh/m²/an" if ref is not None else "—"

st.subheader("Résultat — consommation électrique corrigée")
k1, k2, k3 = st.columns(3)
k1.metric("Conso corrigée", f"{y_pred:.1f} kWh/m²/an")
k2.metric("Référence parc (médiane)", ref_txt)
k3.metric("Écart vs médiane", f"{(y_pred - float(ref)):.1f}" if ref is not None else "—")

st.caption(
    "Note : cet indicateur est une consommation électrique attendue *à caractéristiques DPE données* (corrigée des biais observés). "
    "Ce n’est pas une prévision de facture individuelle."
)

with st.expander("Voir les variables envoyées au modèle"):
    st.dataframe(X_user, use_container_width=True)


# =========================================================
# SHAP local
# =========================================================
st.divider()
st.subheader("Explication SHAP (locale) — pourquoi cette prédiction ?")

X_user_trans_df = transform_for_shap(preprocess, X_user, feature_names)

shap_vals = explainer.shap_values(X_user_trans_df)

base_value = float(explainer.expected_value)
pred_from_shap = base_value + float(shap_vals[0].sum())



st.caption(
    "Le graphique décompose la prédiction en contributions additives des variables. "
    "Contribution positive = augmente la conso ; négative = diminue."
)

exp = shap.Explanation(
    values=shap_vals[0],
    base_values=base_value,
    data=X_user_trans_df.iloc[0].values,
    feature_names=X_user_trans_df.columns
)

fig = plt.figure(figsize=(10, 6))
shap.plots.waterfall(exp, max_display=15, show=False)
st.pyplot(fig, clear_figure=True)
