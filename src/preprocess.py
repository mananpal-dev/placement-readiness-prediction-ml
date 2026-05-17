"""
Advanced preprocessing pipeline with feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')


FEATURE_COLS = [
    'CGPA', 'Internships', 'Projects', 'Workshops_Certifications',
    'AptitudeTestScore', 'SoftSkillsRating', 'ExtracurricularActivities',
    'PlacementTraining', 'SSC_Marks', 'HSC_Marks', 'Backlogs',
    'CommunicationScore', 'TechnicalScore', 'MockInterviews',
    'GitHub_Repos', 'CompetitiveCoding'
]

TARGET_COL = 'PlacementStatus'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create powerful engineered features from raw data."""
    df = df.copy()

    # Academic composite
    df['AcademicScore'] = (df['CGPA'] * 10 + df['SSC_Marks'] * 0.3 + df['HSC_Marks'] * 0.3) / 4
    df['AcademicScore'] = df['AcademicScore'].round(3)

    # Technical composite
    df['TechnicalComposite'] = (
        df['TechnicalScore'] * 0.4 +
        df['AptitudeTestScore'] * 0.3 +
        df['GitHub_Repos'] * 3 +
        df['CompetitiveCoding'] * 5 +
        df['Projects'] * 4
    ).round(3)

    # Soft power index
    df['SoftPowerIndex'] = (
        df['SoftSkillsRating'] * 10 +
        df['CommunicationScore'] * 10 +
        df['ExtracurricularActivities'] * 5 +
        df['MockInterviews'] * 4
    ).round(3)

    # Experience index
    df['ExperienceIndex'] = (
        df['Internships'] * 15 +
        df['Workshops_Certifications'] * 5 +
        df['PlacementTraining'] * 10
    ).round(3)

    # Penalty for backlogs
    df['BacklogPenalty'] = df['Backlogs'] * 10

    # Overall readiness score (composite)
    df['ReadinessScore'] = (
        df['AcademicScore'] * 0.25 +
        df['TechnicalComposite'] * 0.3 +
        df['SoftPowerIndex'] * 0.25 +
        df['ExperienceIndex'] * 0.2 -
        df['BacklogPenalty'] * 0.1
    ).round(3)

    # CGPA bands
    df['CGPA_Band'] = pd.cut(
        df['CGPA'],
        bins=[0, 5, 6, 7, 8, 9, 10],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(float)

    # Interaction features
    df['CGPA_x_Aptitude'] = (df['CGPA'] * df['AptitudeTestScore'] / 100).round(3)
    df['Tech_x_Internship'] = (df['TechnicalScore'] * (df['Internships'] + 1)).round(3)
    df['Soft_x_Communication'] = (df['SoftSkillsRating'] * df['CommunicationScore']).round(3)

    return df


ENGINEERED_COLS = FEATURE_COLS + [
    'AcademicScore', 'TechnicalComposite', 'SoftPowerIndex',
    'ExperienceIndex', 'ReadinessScore', 'CGPA_Band',
    'CGPA_x_Aptitude', 'Tech_x_Internship', 'Soft_x_Communication'
]


def preprocess_data(file_path: str):
    """Full preprocessing pipeline."""
    df = pd.read_csv(file_path)

    if 'StudentID' in df.columns:
        df = df.drop('StudentID', axis=1)

    # Encode object columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = preprocess_data("data/placement_data.csv")
    print("Preprocessed shape:", df.shape)
    print("Features available:", list(df.columns))
