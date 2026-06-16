import streamlit as st

st.set_page_config(
    page_title="DPE × Enedis — démonstrateur",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("DPE × Enedis — démonstrateur")

st.divider()

st.subheader("🎯 Objectif")
st.markdown(
    """
- **Visualiser** l’écart entre la consommation conventionnelle du DPE et les consommations électriques observées.
- **Proposer** une estimation de consommation électrique attendue à partir des caractéristiques DPE via un simulateur.
"""
)

st.divider()

st.subheader("🧭 Pages disponibles")
c1, c2 = st.columns(2)

with c1:
    st.markdown("### 1️⃣ Écart DPE / réel par classe calculée")
    st.write(
        "Visualise la distribution de l'écart **Enedis − DPE** par classe calculée depuis la consommation DPE (A–G). "
        "Utile pour constater les écarts systématiques, notamment en tout-électrique."
    )
    st.page_link("pages/01_ecart_dpe_reel.py", label="➡️ Ouvrir l’analyse des écarts", icon="📊")

with c2:
    st.markdown("### 2️⃣ Simulateur (prédiction via DPE)")
    st.write(
        "Saisis des caractéristiques issues du DPE et obtiens une **prédiction de consommation électrique** (kWh/m²/an) "
        "apprise sur les consommations observées."
    )
    st.page_link("pages/02_simulateur_dpe.py", label="➡️ Ouvrir le simulateur", icon="🧮")

st.divider()

st.subheader("📝 Définitions rapides")
st.markdown(
    """
- **Consommation DPE** : consommation *conventionnelle* (scénarios standardisés), utilisée pour l’étiquette.
- **Consommation Enedis** : consommation électrique *observée* (agrégée, ≥10 logements).
- **Écart (Enedis − DPE)** : négatif ⇒ le DPE surestime la consommation électrique observée ; positif ⇒ le DPE sousestime la consommation électrique observée.
"""
)

st.divider()

st.subheader("⚠️ Limites")
st.markdown(
    """
- Le DPE n’est **pas une facture** : il décrit une performance conventionnelle du bâti.
- La consommation Enedis est **agrégée** (≥10 logements) : Il n'y a pas d’analyse à l’unité logement.
- Le simulateur estime une consommation **attendue** à caractéristiques données : ce n’est pas une prédiction individuelle.
"""
)

st.divider()

st.subheader("🚀 Pistes d’amélioration")

st.markdown(
    """
##### 1️⃣ Intégrer des données socio-économiques 

- Revenus médians
- Catégorie socio professionnelle

##### 2️⃣ Intégrer des données d'usage 

- Taux d'occupation
- Taille des ménages

##### 3️⃣ Intégrer d'autres sources d'énergie
- Gaz avec GRDF
"""
)


