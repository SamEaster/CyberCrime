import numpy as np
import pandas as pd


import pandas as pd
import numpy as np

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
    
    df = df.sort_values(by='Risk_Score', ascending=False)

    return df

