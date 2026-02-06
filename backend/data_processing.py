import pandas as pd
import numpy as np
import os

def previous_counts(df):
    categories = ['Cyber Bullying', 'Data Breach',
        'Hacking', 'Identity Theft', 'Malware', 'Online Fraud', 'Others',
        'Phishing', 'Ransomware']

    if 'prev_Cyber Bullying' not in df.columns:
        for cat in categories:
            df[f'prev_{cat}'] = 0

        prev = -1
        for i in reversed(range(len(df))):
            if prev==-1 or prev!=df.Year[i]:
                prev = df.Year[i]
                for cat in categories:
                    df.loc[i, f'prev_{cat}'] = df.loc[i, cat]
            else:
                for cat in categories:
                    df.loc[i, f'prev_{cat}'] = df.loc[i, cat] + df.loc[i+1, f'prev_{cat}']

    return df


if __name__ == "__main__":
    path = "/Users/shubham/Desktop/Projects/CyberCrime Hackathon/data/cybersecurity_cases_india_combined.csv"
    df = pd.read_csv(path)

    df.drop('Day', axis=1, inplace=True)
    n_data = pd.get_dummies(data=df, columns=['Incident_Type'], dtype = 'int', prefix='', prefix_sep='')
    n_data['Malware'] = n_data['Malware']+n_data['Malware_Attacks']
    n_data.drop('Malware_Attacks', axis=1, inplace=True)

    n_data.sort_values(by='Year', ascending=False, inplace=True, ignore_index=True)


    n_df = previous_counts(n_data.copy())

    dir = '/Users/shubham/Desktop/Projects/CyberCrime Hackathon/data/'
    filename = 'cyber_crime_1.csv'
    full_path = os.path.join(dir, filename)
    n_df.to_csv(full_path, index=False)



