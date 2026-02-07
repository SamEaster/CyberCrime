import os
import numpy as np
import pandas as pd
import joblib

# ─── Load the trained model once at import time ─────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'risk_model.joblib')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'n_cyber_crime.csv')

# Load artifact
_artifact = joblib.load(MODEL_PATH)
_model = _artifact['model']  # This is now the RandomForestRegressor object
_mn_year = _artifact.get('mn_year', 2020)
_mx_city = _artifact.get('mx_city', 1)
_mx_fin_score = _artifact.get('mx_fin_score', 100)
_past_incidents = _artifact.get('past_incidents', {})

# Feature columns the Random Forest model expects
FEATURE_COLS = [str(f) for f in _model.feature_names_in_]

THREAT_COLS = [
    'Ransomware', 'Data Breach', 'Hacking', 'Malware',
    'Identity Theft', 'Phishing', 'Online Fraud',
    'Cyber Bullying', 'Others',
]

CATEGORY_ENCODE = {
    'Social Media': 0, 'Government': 1, 'Corporate': 2, 'E-commerce': 3,
    'Educational': 4, 'Financial': 5, 'Personal': 6, 'Health': 7,
}

def _encode_city(city_str: str) -> int:
    """Encode a city name to a numeric value using historical frequency."""
    if os.path.exists(CSV_PATH):
        hist = pd.read_csv(CSV_PATH)
        freq = hist['City'].value_counts().to_dict()
        return freq.get(city_str, 1)
    return 1

def _predict_scores(df: pd.DataFrame) -> np.ndarray:
    """Run prediction through the trained Random Forest model."""
    # Ensure we only pass the exact columns the model expects, in order
    X_input = df[FEATURE_COLS].copy()
    
    # Random Forest (sklearn) supports predicting directly on DataFrames
    preds = _model.predict(X_input)
    
    return np.clip(np.round(preds, 2), 0, 10)

def calculate_importance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate risk scores using the trained Random Forest regressor.
    Input df can have string City/Category — they'll be encoded automatically.
    Adds a 'Risk_Score' column (0–10) and returns the df sorted descending.
    """
    df = df.copy()

    # ── Ensure numeric types ────────────────────────────────────────────────
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(2025).astype(int)
    df['Amount_Lost_INR'] = pd.to_numeric(df['Amount_Lost_INR'], errors='coerce').fillna(0).astype(float)
    
    # Feature Engineering: Random Forest was likely trained on Sqrt(Amount) if you used previous logic
    # Uncomment the line below if your training script used np.sqrt(Amount)
    # df['Amount_Lost_INR'] = np.sqrt(df['Amount_Lost_INR'])

    # ── Encode Category to numeric if needed ────────────────────────────────
    if not pd.api.types.is_numeric_dtype(df['Category']):
        df['Category'] = df['Category'].map(CATEGORY_ENCODE).fillna(6).astype(int)

    # ── Encode City to numeric (frequency) if needed ────────────────────────
    if not pd.api.types.is_numeric_dtype(df['City']):
        if os.path.exists(CSV_PATH):
            hist = pd.read_csv(CSV_PATH)
            city_freq = hist['City'].value_counts().to_dict()
            df['City'] = df['City'].map(lambda c: city_freq.get(c, 1)).astype(int)
        else:
            df['City'] = 1

    # ── Fill prev_ columns with stored defaults if missing ──────────────────
    for t in THREAT_COLS:
        prev_col = f'prev_{t}'
        if prev_col not in df.columns:
            default_val = int(_past_incidents.get(prev_col, 0))
            df[prev_col] = default_val
        else:
            df[prev_col] = pd.to_numeric(df[prev_col], errors='coerce').fillna(0).astype(int)

    # ── Ensure one-hot threat columns exist ─────────────────────────────────
    for t in THREAT_COLS:
        if t not in df.columns:
            df[t] = 0

    # ── Predict ─────────────────────────────────────────────────────────────
    try:
        df['Risk_Score'] = _predict_scores(df)
    except Exception as e:
        print(f"Prediction Error: {e}")
        # Fallback if columns don't match exactly
        df['Risk_Score'] = 0.0

    df = df.sort_values(by='Risk_Score', ascending=False)
    return df

if __name__ == "__main__":
    # Test path
    path = "/Users/shubham/Desktop/Projects/CyberCrime Hackathon/data/n_cyber_crime.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(calculate_importance_score(df[:5].copy()))
    else:
        print(f"File not found at {path}")