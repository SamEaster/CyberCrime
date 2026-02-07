import os
import numpy as np
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'n_cyber_crime.csv')

THREAT_COLS = [
    'Ransomware', 'Data Breach', 'Hacking', 'Malware',
    'Identity Theft', 'Phishing', 'Online Fraud',
    'Cyber Bullying', 'Others',
]


def calculate_importance_score(df):
    """
    Calculate a risk score guaranteed to be in [0, 10].
    Each sub-score is normalized to [0, 10], weights sum to 1.0,
    and a final clamp ensures the result never exceeds 10.
    """
    df = df.copy()

    W_FINANCE = 0.30
    W_CONTEXT = 0.35
    W_HISTORY = 0.15
    W_CITY    = 0.05
    W_RECENCY = 0.15

    risk_map = {
        'Government':  {'Ransomware': 10, 'Hacking': 10, 'Data Breach': 9, 'Malware': 7, 'default': 6},
        'Health':      {'Ransomware': 10, 'Data Breach': 9, 'Malware': 8, 'default': 5},
        'Financial':   {'Online Fraud': 10, 'Identity Theft': 9, 'Phishing': 9, 'Hacking': 8, 'default': 5},
        'Corporate':   {'Ransomware': 9, 'Data Breach': 9, 'Hacking': 8, 'default': 5},
        'Educational': {'Cyber Bullying': 9, 'Data Breach': 7, 'Phishing': 6, 'default': 4},
        'E-commerce':  {'Online Fraud': 9, 'Identity Theft': 9, 'Hacking': 7, 'default': 5},
        'Social Media':{'Cyber Bullying': 9, 'Identity Theft': 8, 'Phishing': 7, 'default': 4},
        'Personal':    {'Identity Theft': 8, 'Online Fraud': 8, 'Cyber Bullying': 7, 'default': 3},
    }

    # --- Ensure numeric types early ---
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(2025).astype(int)
    df['Amount_Lost_INR'] = pd.to_numeric(df['Amount_Lost_INR'], errors='coerce').fillna(0)

    # --- 1. Financial score (0-10) via log-normalisation ---
    fin_raw = np.log1p(df['Amount_Lost_INR'].astype(float))
    fin_max = fin_raw.max()
    fin_score = (fin_raw / fin_max * 10) if fin_max > 0 else pd.Series(0.0, index=df.index)

    # --- 2. Context severity score (0-10) from category × threat map ---
    def get_context_severity(row):
        category = row.get('Category', 'Personal')
        sector_risks = risk_map.get(category, risk_map['Personal'])
        max_severity = 0
        for threat in THREAT_COLS:
            if row.get(threat, 0) == 1:
                severity = sector_risks.get(threat, sector_risks.get('default', 3))
                max_severity = max(max_severity, severity)
        return max_severity if max_severity > 0 else sector_risks.get('default', 3)

    context_score = df.apply(get_context_severity, axis=1).clip(0, 10)

    # --- 3. History score (0-10) from previous incident counts ---
    history_scores_raw = np.zeros(len(df))
    for threat in THREAT_COLS:
        prev_col = f"prev_{threat}"
        if prev_col in df.columns:
            df[prev_col] = pd.to_numeric(df[prev_col], errors='coerce').fillna(0)
            mask = df[threat] == 1
            history_scores_raw[mask] += np.log1p(df.loc[mask, prev_col].values)

    max_hist = history_scores_raw.max()
    history_score = (history_scores_raw / max_hist * 10) if max_hist > 0 else history_scores_raw
    history_score = np.clip(history_score, 0, 10)

    # --- 4. City hotspot score (0-10) from frequency ---
    if not pd.api.types.is_numeric_dtype(df['City']):
        if os.path.exists(CSV_PATH):
            hist = pd.read_csv(CSV_PATH)
            city_freq = hist['City'].value_counts().to_dict()
            city_numeric = df['City'].map(lambda c: city_freq.get(c, 1)).astype(float)
        else:
            city_numeric = pd.Series(1.0, index=df.index)
    else:
        city_numeric = df['City'].astype(float)

    city_max = city_numeric.max()
    city_min = city_numeric.min()
    if city_max > city_min:
        city_score = ((city_numeric - city_min) / (city_max - city_min) * 10)
    else:
        city_score = pd.Series(5.0, index=df.index)
    city_score = city_score.clip(0, 10)

    # --- 5. Recency score (0-10) — more recent years score higher ---
    if 'Year' in df.columns and df['Year'].nunique() > 1:
        year_min = df['Year'].min()
        year_diff = (df['Year'] - year_min).astype(float)
        recency_raw = np.exp(year_diff)
        recency_score = (recency_raw / recency_raw.max()) * 10
    else:
        recency_score = pd.Series(5.0, index=df.index)
    recency_score = np.clip(recency_score, 0, 10)

    # --- Weighted combination & final clamp to [0, 10] ---
    df['Risk_Score'] = (
        W_FINANCE * fin_score +
        W_CONTEXT * context_score +
        W_HISTORY * history_score +
        W_CITY    * city_score +
        W_RECENCY * recency_score
    ).round(2).clip(0, 10)

    df = df.sort_values(by='Risk_Score', ascending=False)

    return df