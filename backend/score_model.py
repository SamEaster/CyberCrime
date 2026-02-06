import numpy as np
import pandas as pd
import joblib

CITY_TIER_MAP = {
        
        "bangalore": 1,
        "bengaluru": 1,
        "delhi": 1,
        "chennai": 1,
        "hyderabad": 1,
        "mumbai": 1,
        "pune": 1,
        "kolkata": 1,
        "ahmedabad": 1,

        "amritsar": 2,
        "bhopal": 2,
        "bhubaneswar": 2,
        "chandigarh": 2,
        "faridabad": 2,
        "ghaziabad": 2,
        "jamshedpur": 2,
        "jaipur": 2,
        "kochi": 2,
        "lucknow": 2,
        "nagpur": 2,
        "patna": 2,
        "raipur": 2,
        "surat": 2,
        "visakhapatnam": 2,
        "agra": 2,
        "ajmer": 2,
        "kanpur": 2,
        "mysuru": 2,
        "mysore": 2,
        "srinagar": 2,

        "etawah": 3,
        "roorkee": 3,
        "rajahmundry": 3,
        "rajamundry": 3,
        "bhatinda": 3,
        "bathinda": 3,
        "hajipur": 3,
        "rohtak": 3,
        "hosur": 3,
        "junagadh": 3,
        "udaipur": 3,
        "salem": 3,
        "jhansi": 3,
        "madurai": 3,
        "vijayawada": 3,
        "meerut": 3,
        "mathura": 3,
        "bikaner": 3,
        "cuttack": 3,
        "nashik": 3,

        "banswara": 4,
        "bhadreswar": 4,
        "chilakaluripet": 4,
        "datia": 4,
        "gangtok": 4,
        "kalyani": 4,
        "kapurthala": 4,
        "kasganj": 4,
        "nagda": 4,
        "sujangarh": 4,
}

victim_weights = {
    'Government': 10, 'Health': 9, 'Financial': 8, 'Corporate': 7,
    'Educational': 6, 'E-commerce': 5, 'Social Media': 4, 'Personal': 2
}

def process_cities(series):
        return series.astype('str').str.lower().map(CITY_TIER_MAP).fillna(0)

frozen_data = joblib.load("data/risk_model.joblib")

def previous_count(data):
    categories = ['Cyber Bullying', 'Data Breach',
        'Hacking', 'Identity Theft', 'Malware', 'Online Fraud', 'Others',
        'Phishing', 'Ransomware']

    for cat in categories:
        data[f'prev_{cat}'] = 0

    if data['Year']!=frozen_data['past_incidents']['Year']:
      for cat in categories:
        data[f"prev_{cat}"] = data[cat]

    else:
      for cat in categories:
        data[f"prev_{cat}"] = data[cat] + frozen_data['past_incidents'][f'prev_{cat}']

    return data

def scoring_model(x):

    if x['Category'].dtype=='O':
        x['Category'] = x['Category'].map(victim_weights)

    if x['City'].dtype=='O':
        x['City'] = x['City'].map(process_cities)
    
    x["Amount_Lost_INR"] = np.sqrt(x["Amount_Lost_INR"])

    x = previous_count(x)
    x = pd.DataFrame([x])
    x = x[frozen_data['cols']]

    return frozen_data['model'].predict()


def calculate_importance_score(df):

    W_FINANCE = 0.3    
    W_CONTEXT = 0.4
    W_HISTORY = 0.15   
    W_RECENCY = 0.15   

    risk_map = {
        'Government': {'Ransomware': 10, 'Hacking': 10, 'Data Breach': 9, 'Malware': 7, 'default': 6},
        'Health':     {'Ransomware': 10, 'Data Breach': 9, 'Malware': 8, 'default': 5},
        'Financial':  {'Online Fraud': 10, 'Identity Theft': 9, 'Phishing': 9, 'Hacking': 8, 'default': 5},
        'Corporate':  {'Ransomware': 9, 'Data Breach': 9, 'Hacking': 8, 'default': 5},
        'Educational':{'Cyber Bullying': 9, 'Data Breach': 7, 'Phishing': 6, 'default': 4},
        'E-commerce': {'Online Fraud': 9, 'Identity Theft': 9, 'Hacking': 7, 'default': 5},
        'Social Media':{'Cyber Bullying': 9, 'Identity Theft': 8, 'Phishing': 7, 'default': 4},
        'Personal':   {'Identity Theft': 8, 'Online Fraud': 8, 'Cyber Bullying': 7, 'default': 3}
    }

    threat_cols = [
        'Ransomware', 'Data Breach', 'Hacking', 'Malware',
        'Identity Theft', 'Phishing', 'Online Fraud',
        'Cyber Bullying', 'Others'
    ]

    fin_score = np.log1p(df['Amount_Lost_INR'])
    if fin_score.max() > 0:
        fin_score = (fin_score / fin_score.max()) * 10
    else:
        fin_score = 0

    
    def get_context_severity(row):
        category = row.get('Category', 'Personal')
        sector_risks = risk_map.get(category, risk_map['Personal'])
        
        max_severity = 0
        for threat in threat_cols:
            if row.get(threat, 0) == 1:
                severity = sector_risks.get(threat, sector_risks.get('default', 3))
                max_severity = max(max_severity, severity)
        
        return max_severity if max_severity > 0 else sector_risks.get('default', 3)

    context_score = df.apply(get_context_severity, axis=1)


    history_scores_raw = np.zeros(len(df))
    
    for threat in threat_cols:
        prev_col = f"prev_{threat}"
        if prev_col in df.columns:
            mask = df[threat] == 1
            history_scores_raw[mask] += np.log1p(df.loc[mask, prev_col])

    max_hist = history_scores_raw.max()
    history_score = (history_scores_raw / max_hist * 10) if max_hist > 0 else history_scores_raw
    history_score += max(df['City']) - df['City'] + 2


    if 'Year' in df.columns:
        year_diff = df['Year'] - df['Year'].min()
        recency_score = np.exp(year_diff) 
        recency_score = (recency_score / recency_score.max()) * 10

    else:
        recency_score = 5 

    df['Risk_Score'] = (
        (W_FINANCE * fin_score) +
        (W_CONTEXT * context_score) +
        (W_HISTORY * history_score) +
        (W_RECENCY * recency_score)
    )

    # df['Risk_Score'] = df['Risk_Score'].round(2)
    
    # df = df.sort_values(by='Risk_Score', ascending=False)
    return df


if __name__ == "__main__":

    path = "data/n_cyber_crime.csv"
    # df = pd.read_csv(path)

    # n_df = calculate_importance_score(df)
    # x = df
    # y = calculate_importance_score(df)

    print(frozen_data.keys())


    