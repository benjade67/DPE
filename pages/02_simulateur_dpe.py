import streamlit as st
import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt

from utils.loaders import load_model, load_defaults
# from utils.shap_utils import build_shap_tools, transform_for_shap


st.title("2️⃣ Simulateur — Prédiction consommation électrique via les données du DPE")

# =========================================================
# Cache: modèle & defaults
# =========================================================
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()

@st.cache_data(show_spinner=False)
def get_defaults():
    return load_defaults()

defaults = get_defaults()

# st.write("✅ Page chargée. Prêt à charger le modèle…")

with st.spinner("Chargement du modèle…"):
    try:
        model = get_model()
        st.success("✅ Modèle chargé")
    except Exception as e:
        st.error("❌ Erreur au chargement du modèle")
        st.exception(e)
        st.stop()

st.caption(
    "Le simulateur estime une consommation électrique attendue (kWh/m²/an) à partir des caractéristiques décrites par le DPE, "
    "corrigée à partir des consommations réelles observées."
)

# =========================================================
# Helpers
# =========================================================
def build_X_user(params: dict, defaults: dict) -> pd.DataFrame:
    mode = params["mode"]

    if mode == "Avancé":
        ubat = float(params["ubat"])
        dep_air = float(params["dep_air"])
        dep_baies = float(params["dep_baies"])
        dep_murs = float(params["dep_murs"])
        dep_pt = float(params["dep_pt"])
        dep_portes = float(params["dep_portes"])
        dep_pb = float(params["dep_pb"])
        dep_ph = float(params["dep_ph"])
    else:
        ubat = float(defaults.get("ubat_w_par_m2_k_moy", 1.2))
        dep_air = float(defaults.get("deperditions_renouvellement_air_moy", 10.0))
        dep_baies = float(defaults.get("deperditions_baies_vitrees_moy", 15.0))
        dep_murs = float(defaults.get("deperditions_murs_moy", 10.0))
        dep_pt = float(defaults.get("deperditions_ponts_thermiques_moy", 5.0))
        dep_portes = float(defaults.get("deperditions_portes_moy", 2.0))
        dep_pb = float(defaults.get("deperditions_planchers_bas_moy", 8.0))
        dep_ph = float(defaults.get("deperditions_planchers_hauts_moy", 8.0))

    X_user = pd.DataFrame([{
        "nombre_de_logements": float(params["n_logements"]),
        "n_dpe": float(params["n_dpe"]),
        "surface_totale": float(params["surface_totale"]),
        "annee_construction": float(params["annee"]),
        "ubat_w_par_m2_k_moy": float(ubat),
        "deperditions_renouvellement_air_moy": float(dep_air),
        "deperditions_baies_vitrees_moy": float(dep_baies),
        "deperditions_murs_moy": float(dep_murs),
        "deperditions_ponts_thermiques_moy": float(dep_pt),
        "deperditions_portes_moy": float(dep_portes),
        "deperditions_planchers_bas_moy": float(dep_pb),
        "deperditions_planchers_hauts_moy": float(dep_ph),
        "chauffage_elec": int(params["chauffage_elec"]),
        "ecs_elec": int(params["ecs_elec"]),
        "zone_climatique_mode": params["zone"],
        "type_batiment_mode": params["type_bat"],
        "type_installation_chauffage_mode": params["type_chauffage"],
        "type_installation_ecs_mode": params["type_ecs"],
    }])

    return X_user

def compute_prediction(model, defaults, params: dict):
    X_user = build_X_user(params, defaults)
    y_pred = float(model.predict(X_user)[0])
    ref = defaults.get("conso_ref", None)
    return {"X_user": X_user, "y_pred": y_pred, "ref": ref}

# =========================================================
# 1) Valeurs par défaut + calcul initial (au chargement)
# =========================================================
if "sim_params_applied" not in st.session_state:
    st.session_state["sim_params_applied"] = {
        "mode": "Simple",
        "chauffage_elec": 0,
        "ecs_elec": 0,
        "zone": "H1b",
        "annee": 1974,
        "n_logements": 14,
        "surface_totale": 928,
        "n_dpe": 4,
        "type_ecs": "collectif",
        "type_chauffage": "collectif",
        "type_bat": "appartement",
        # valeurs avancées par défaut (si mode Avancé)
        "ubat": 0.55,
        "dep_air": 48.7,
        "dep_baies": 31.5,
        "dep_murs": 14.7,
        "dep_pt": 30.9,
        "dep_portes": 6.7,
        "dep_pb": 0.0,
        "dep_ph": 51.2,
    }
    with st.spinner("Calcul initial (valeurs par défaut)..."):
        st.session_state["sim_result"] = compute_prediction(
            model, defaults, st.session_state["sim_params_applied"]
        )

applied = st.session_state["sim_params_applied"]

# =========================================================
# 2) Mode hors form => affichage immédiat des champs avancés
# =========================================================
mode = st.radio(
    "Mode de saisie",
    ["Simple", "Avancé"],
    horizontal=True,
    index=0 if applied["mode"] == "Simple" else 1,
)

# On mémorise le mode immédiatement (sans recalcul)
if mode != applied["mode"]:
    st.session_state["sim_params_applied"] = {**applied, "mode": mode}
    applied = st.session_state["sim_params_applied"]

st.divider()

# =========================================================
# 3) Formulaire: inputs + bouton Appliquer
# =========================================================
with st.form("sim_form"):
    st.subheader("Caractéristiques principales")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        chauffage_elec = st.selectbox(
            "Chauffage électrique",
            [0, 1],
            index=0 if applied["chauffage_elec"] == 0 else 1,
            format_func=lambda x: "Oui" if x == 1 else "Non",
        )
    with c2:
        ecs_elec = st.selectbox(
            "ECS électrique",
            [0, 1],
            index=0 if applied["ecs_elec"] == 0 else 1,
            format_func=lambda x: "Oui" if x == 1 else "Non",
        )
    with c3:
        zones = ["H1a", "H1b", "H1c", "H2a", "H2b", "H2c", "H2d", "H3"]
        zone = st.selectbox(
            "Zone climatique",
            zones,
            index=zones.index(applied["zone"]) if applied["zone"] in zones else 1,
        )
    with c4:
        annee = st.slider("Année de construction", 1850, 2025, int(applied["annee"]))

    c5, c6, c7 = st.columns(3)
    with c5:
        n_logements = st.number_input(
            "Nombre de logements",
            min_value=1,
            max_value=5000,
            value=int(applied["n_logements"]),
            step=1,
        )
    with c6:
        surface_totale = st.number_input(
            "Surface totale (m²)",
            min_value=20,
            max_value=1_500_000,
            value=int(applied["surface_totale"]),
            step=10,
        )
    with c7:
        n_dpe = st.number_input(
            "Nombre de DPE agrégés (n_dpe)",
            min_value=1,
            max_value=20000,
            value=int(applied["n_dpe"]),
            step=1,
        )

    type_ecs = st.selectbox(
        "Type d'installation ECS",
        ["individuel", "collectif"],
        index=0 if applied["type_ecs"] == "individuel" else 1,
    )
    type_chauffage_opts = ["individuel", "collectif", "mixte (collectif-individuel)"]
    type_chauffage = st.selectbox(
        "Type d'installation chauffage",
        type_chauffage_opts,
        index=type_chauffage_opts.index(applied["type_chauffage"])
        if applied["type_chauffage"] in type_chauffage_opts else 1,
    )
    type_bat_opts = ["habitation", "appartement", "maison"]
    type_bat = st.selectbox(
        "Type de bâtiment",
        type_bat_opts,
        index=type_bat_opts.index(applied["type_bat"])
        if applied["type_bat"] in type_bat_opts else 1,
    )

    st.divider()

    # Paramètres thermiques visibles immédiatement (mode hors form)
    if mode == "Avancé":
        st.subheader("Paramètres thermiques (avancé)")

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            ubat = st.slider("UBAT (W/m².K)", 0.0, 12.0, float(applied["ubat"]), step=0.05)
        with a2:
            dep_air = st.slider("Déperditions renouvellement d'air", 0.0, 200.0, float(applied["dep_air"]), step=1.0)
        with a3:
            dep_baies = st.slider("Déperditions baies vitrées", 0.0, 200.0, float(applied["dep_baies"]), step=1.0)
        with a4:
            dep_murs = st.slider("Déperditions murs", 0.0, 200.0, float(applied["dep_murs"]), step=1.0)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            dep_pt = st.slider("Déperditions ponts thermiques", 0.0, 200.0, float(applied["dep_pt"]), step=1.0)
        with b2:
            dep_portes = st.slider("Déperditions portes", 0.0, 200.0, float(applied["dep_portes"]), step=1.0)
        with b3:
            dep_pb = st.slider("Déperditions planchers bas", 0.0, 200.0, float(applied["dep_pb"]), step=1.0)
        with b4:
            dep_ph = st.slider("Déperditions planchers hauts", 0.0, 200.0, float(applied["dep_ph"]), step=1.0)
    else:
        # placeholders (non utilisés en Simple)
        ubat = applied["ubat"]
        dep_air = applied["dep_air"]
        dep_baies = applied["dep_baies"]
        dep_murs = applied["dep_murs"]
        dep_pt = applied["dep_pt"]
        dep_portes = applied["dep_portes"]
        dep_pb = applied["dep_pb"]
        dep_ph = applied["dep_ph"]

    submitted = st.form_submit_button("Appliquer")

# =========================================================
# 4) Appliquer => recalcul uniquement ici
# =========================================================
if submitted:
    new_params = {
        "mode": mode,
        "chauffage_elec": chauffage_elec,
        "ecs_elec": ecs_elec,
        "zone": zone,
        "annee": annee,
        "n_logements": n_logements,
        "surface_totale": surface_totale,
        "n_dpe": n_dpe,
        "type_ecs": type_ecs,
        "type_chauffage": type_chauffage,
        "type_bat": type_bat,
        "ubat": ubat,
        "dep_air": dep_air,
        "dep_baies": dep_baies,
        "dep_murs": dep_murs,
        "dep_pt": dep_pt,
        "dep_portes": dep_portes,
        "dep_pb": dep_pb,
        "dep_ph": dep_ph,
    }
    st.session_state["sim_params_applied"] = new_params
    with st.spinner("Calcul en cours..."):
        st.session_state["sim_result"] = compute_prediction(model, defaults, new_params)

# =========================================================
# 5) Affichage du dernier résultat appliqué
# =========================================================
res = st.session_state.get("sim_result")
if res is None:
    st.info("Aucun résultat en mémoire.")
    st.stop()

y_pred = res["y_pred"]
X_user = res["X_user"]
ref = res["ref"]

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
# SHAP local (optionnel)
# =========================================================
# st.divider()
# st.subheader("Explication SHAP (locale) — pourquoi cette prédiction ?")
# ... (inchangé, à réactiver si besoin)
