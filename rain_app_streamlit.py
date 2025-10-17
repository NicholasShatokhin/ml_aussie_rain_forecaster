import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
from pathlib import Path

st.set_page_config(page_title="Tomorow Rain Forecast — Random Forest", page_icon="🌧️", layout="wide")
st.title("🌧️ Will it rain tomorrow? — Weather Forecast")
st.caption("Load `aussie_rain.joblib` via Upload (Opt.1), Google Drive (Opt.2), or GitHub (Opt.3), then predict.")

# Utilities
def load_bundle_from_bytes(file_bytes: bytes):
    file_obj = io.BytesIO(file_bytes)
    bundle = joblib.load(file_obj)
    return bundle

def parse_gdrive_id(text: str):
    text = (text or "").strip()
    if not text:
        return None
    if "drive.google.com" in text:
        import re
        m = re.search(r"/d/([A-Za-z0-9_-]+)", text)
        if m:
            return m.group(1)
        m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", text)
        if m:
            return m.group(1)
        return None
    return text

@st.cache_data(show_spinner=False)
def download_from_gdrive(file_id: str) -> bytes:
    import requests
    session = requests.Session()
    URL = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    response = session.get(URL, params=params, stream=True)
    def _get_confirm_token(resp):
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                return v
        return None
    token = _get_confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(URL, params=params, stream=True)
    response.raise_for_status()
    return response.content

@st.cache_data(show_spinner=False)
def download_from_github(raw_url: str, token: str | None = None) -> bytes:
    import requests
    headers = {}
    if token:
        headers["Authorization"] = f"token {token.strip()}"
    url = raw_url.strip()
    if "github.com" in url and "raw.githubusercontent.com" not in url:
        parts = url.split("github.com/")[-1].split("/")
        if "blob" in parts:
            i = parts.index("blob")
            user, repo = parts[0], parts[1]
            branch = parts[i+1]
            path = "/".join(parts[i+2:])
            url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    return r.content

with st.sidebar:
    st.header("Load Model Bundle")
    st.sidebar.markdown("If no file is provided, the app will try to load the default model from `./model/aussie_rain.joblib`.")
    tab_up, tab_gd, tab_gh = st.tabs(["Upload", "Google Drive", "GitHub (raw)"])
    bundle_bytes = None
    with tab_up:
        up = st.file_uploader("Upload aussie_rain.joblib", type=["joblib","pkl"])
        if up is not None:
            bundle_bytes = up.read()
            st.success("File uploaded.")
    with tab_gd:
        gdrive_in = st.text_input("Paste Google Drive link or File ID")
        if st.button("Download from Drive"):
            file_id = parse_gdrive_id(gdrive_in)
            if not file_id:
                st.error("Could not parse Google Drive file ID.")
            else:
                try:
                    bundle_bytes = download_from_gdrive(file_id)
                    st.success("Downloaded from Google Drive.")
                except Exception as e:
                    st.error(f"Download failed: {e}")
    with tab_gh:
        st.caption("Paste a raw GitHub URL (e.g., https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/to/aussie_rain.joblib).")
        gh_url = st.text_input("GitHub raw URL or 'blob' URL (we'll convert)")
        gh_token = st.text_input("Optional: GitHub token (for private repos)", type="password")
        if st.button("Download from GitHub"):
            if not gh_url.strip():
                st.error("Please paste a GitHub URL.")
            else:
                try:
                    bundle_bytes = download_from_github(gh_url, token=gh_token or None)
                    st.success("Downloaded from GitHub.")
                except Exception as e:
                    st.error(f"GitHub download failed: {e}")

    if bundle_bytes is None:
        default_path = Path("./model/aussie_rain.joblib")
        if default_path.exists():
            with open(default_path, "rb") as f:
                bundle_bytes = f.read()
            st.success("Loaded default model from ./model/aussie_rain.joblib")
        else:
            st.warning("No model uploaded or found in ./model/aussie_rain.joblib")
            st.stop()

try:
    bundle = load_bundle_from_bytes(bundle_bytes)
except Exception as e:
    st.error(f"Failed to load bundle: {e}")
    st.stop()

model = bundle["model"]
num_cols = bundle["numeric_cols"]
cat_cols = bundle["categorical_cols"]
input_cols = bundle["input_cols"]
classes = getattr(model, "classes_", ["No","Yes"])

st.sidebar.write("**Target:**", bundle.get("target_col", "RainTomorrow"))
st.sidebar.write("**#Numeric:**", len(num_cols))
st.sidebar.write("**#Categorical:**", len(cat_cols))

def preprocess_single(row_dict: dict, bundle: dict):
    num_cols = bundle["numeric_cols"]
    cat_cols = bundle["categorical_cols"]
    enc_cols = bundle["encoded_cols"]
    df = pd.DataFrame([row_dict])
    for c in bundle["input_cols"]:
        if c not in df.columns:
            df[c] = np.nan
    df[num_cols] = bundle["imputer"].transform(df[num_cols])
    df[num_cols] = bundle["scaler"].transform(df[num_cols])
    df[enc_cols] = bundle["encoder"].transform(df[cat_cols])
    X = pd.concat([df[num_cols], df[enc_cols]], axis=1)
    return X

def preprocess_batch(df: pd.DataFrame, bundle: dict):
    num_cols = bundle["numeric_cols"]
    cat_cols = bundle["categorical_cols"]
    enc_cols = bundle["encoded_cols"]
    for c in bundle["input_cols"]:
        if c not in df.columns:
            df[c] = np.nan
    df[num_cols] = bundle["imputer"].transform(df[num_cols])
    df[num_cols] = bundle["scaler"].transform(df[num_cols])
    df[enc_cols] = bundle["encoder"].transform(df[cat_cols])
    X = pd.concat([df[num_cols], df[enc_cols]], axis=1)
    return X

st.markdown("---")
tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

with tab_single:
    st.subheader("Enter today's weather (AUS format)")
    colL, colR = st.columns(2)
    single = {}
    with colL:
        for c in cat_cols:
            single[c] = st.text_input(c, value="")
    with colR:
        for c in num_cols:
            single[c] = st.number_input(c, value=0.0, step=0.1)
    if st.button("Predict (Single)", type="primary"):
        try:
            X = preprocess_single(single, bundle)
            proba = model.predict_proba(X)[0]
            import numpy as _np
            pred_idx = int(_np.argmax(proba))
            pred = classes[pred_idx]
            conf = float(proba[pred_idx])
            st.success(f"Prediction: **{pred}**  (probability={conf:.3f})")
            st.caption("Classes order: " + ", ".join(map(str, classes)))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab_batch:
    st.subheader("Upload CSV (weatherAUS-compatible)")
    up_csv = st.file_uploader("CSV file", type=["csv"], key="batch")
    if up_csv is not None:
        try:
            df_csv = pd.read_csv(up_csv)
            X = preprocess_batch(df_csv.copy(), bundle)
            probs = model.predict_proba(X)
            import numpy as _np
            preds = _np.take(model.classes_, _np.argmax(probs, axis=1))
            out = df_csv.copy()
            out["prediction"] = preds
            yes_idx = list(model.classes_).index("Yes") if "Yes" in model.classes_ else 1
            out["prob_yes"] = probs[:, yes_idx]
            st.success(f"Inferred {len(out)} rows.")
            st.dataframe(out.head(50))
            st.download_button("Download predictions", data=out.to_csv(index=False).encode("utf-8"),
                               file_name="aussie_rain_predictions.csv", mime="text/csv")
            if "RainTomorrow" in df_csv.columns:
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                acc = accuracy_score(df_csv["RainTomorrow"], preds)
                prec, rec, f1, _ = precision_recall_fscore_support(df_csv["RainTomorrow"], preds, average="binary", pos_label="Yes", zero_division=0)
                st.info(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

st.caption("Note: For 100MB+ models, GitHub/Drive may require special handling. For private repos, provide a personal access token.")
