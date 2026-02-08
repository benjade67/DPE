# utils/shap_utils.py
import pandas as pd
import shap
import streamlit as st


@st.cache_resource
def build_shap_tools(_pipeline):
    """
    Retourne preprocess, explainer, feature_names
    _pipeline => underscore pour éviter unhashable Streamlit.
    """
    preprocess = _pipeline.named_steps["preprocess"]
    rf_model = _pipeline.named_steps["model"]

    features_num = preprocess.transformers_[0][2]
    features_cat = preprocess.transformers_[1][2]

    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(features_cat)

    feature_names = list(features_num) + list(cat_feature_names)

    explainer = shap.TreeExplainer(rf_model)
    return preprocess, explainer, feature_names


def transform_for_shap(preprocess, X_user: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    X_trans = preprocess.transform(X_user)
    return pd.DataFrame(X_trans, columns=feature_names)
