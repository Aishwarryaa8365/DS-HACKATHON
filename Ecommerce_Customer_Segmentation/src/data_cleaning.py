import pandas as pd

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['MembershipType'] = df['MembershipType'].map({'Bronze': 0, 'Silver': 1, 'Gold': 2})
    return df
