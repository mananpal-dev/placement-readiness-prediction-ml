import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Drop unnecessary column
    if 'StudentID' in df.columns:
        df.drop('StudentID', axis=1, inplace=True)

    # Encode categorical columns
    le = LabelEncoder()

    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df