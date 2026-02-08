# utils/loaders.py
import os
import json
import urllib.request
import numpy as np
import pandas as pd
import joblib
import streamlit as st

DATA_PATH = "data/df_corr.parquet"
MODEL_PATH = "models/rf_dpe_corrige.joblib"
DEFAULTS_PATH = "models/defaults_simulateur.joblib"
DEP_GEOJSON_PATH = "data/departements.geojson"
DEP_GEOJSON_URL = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"

MODEL_URL = st.secrets.get(
    "MODEL_URL",
    "https://huggingface.co/DataBenFr/rf_dpe_corrige/resolve/main/rf_dpe_corrige.joblib?download=true",
)

@st.cache_data
def load_df_corr(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

@st.cache_resource
def load_defaults(path: str = DEFAULTS_PATH) -> dict:
    return joblib.load(path)

@st.cache_data
def load_dep_geojson(path: str = DEP_GEOJSON_PATH) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(DEP_GEOJSON_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _download_if_missing(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


@st.cache_resource
def load_model(path: str = MODEL_PATH):
    _download_if_missing(MODEL_URL, path)
    return joblib.load(path)








