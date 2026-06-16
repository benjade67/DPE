# DPE x Enedis - Streamlit

Application Streamlit de demonstration autour des ecarts entre consommations DPE conventionnelles et consommations electriques observees.

## Pages

- Ecart consommation : visualisation de l'ecart Enedis - DPE conventionnelle par classe calculee.
- Simulateur : prediction de consommation electrique corrigee a partir de variables DPE.

## Modele du simulateur

Le simulateur utilise par defaut un modele local leger :

```text
models/simulateur_light.joblib
```

Ce modele est embarque dans le depot pour eviter un telechargement long ou fragile au demarrage de l'application en production.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```
