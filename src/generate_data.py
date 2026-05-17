"""
Synthetic dataset generator for placement prediction.
Generates realistic student data with nuanced correlations.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
np.random.seed(42)

n = 2000

cgpa = np.round(np.clip(np.random.normal(7.2, 1.1, n), 4.0, 10.0), 2)
internships = np.random.choice([0,1,2,3], n, p=[0.3,0.4,0.2,0.1])
projects = np.random.choice([0,1,2,3,4,5], n, p=[0.1,0.2,0.3,0.2,0.15,0.05])
workshops = np.random.choice([0,1,2,3,4], n, p=[0.2,0.3,0.25,0.15,0.1])
aptitude = np.round(np.clip(np.random.normal(68, 15, n), 20, 100), 1)
soft_skills = np.round(np.clip(np.random.normal(3.5, 0.9, n), 1.0, 5.0), 1)
extracurricular = np.random.choice([0,1], n, p=[0.4,0.6])
placement_training = np.random.choice([0,1], n, p=[0.35,0.65])
ssc = np.round(np.clip(np.random.normal(72, 12, n), 40, 100), 1)
hsc = np.round(np.clip(np.random.normal(70, 13, n), 40, 100), 1)
backlogs = np.random.choice([0,1,2,3], n, p=[0.6,0.2,0.12,0.08])
communication = np.round(np.clip(np.random.normal(3.4, 0.95, n), 1.0, 5.0), 1)
technical_score = np.round(np.clip(np.random.normal(65, 18, n), 10, 100), 1)
mock_interviews = np.random.choice([0,1,2,3,4,5], n, p=[0.15,0.2,0.25,0.2,0.12,0.08])
github_repos = np.random.choice([0,1,2,3,4,5,6,7,8,9,10], n,
    p=[0.05,0.1,0.15,0.2,0.2,0.12,0.08,0.04,0.03,0.02,0.01])
competitive_coding = np.random.choice([0,1,2,3], n, p=[0.4,0.3,0.2,0.1])

# Placement score formula with noise
score = (
    cgpa * 4.5 +
    internships * 6 +
    projects * 3 +
    workshops * 2 +
    aptitude * 0.4 +
    soft_skills * 5 +
    extracurricular * 4 +
    placement_training * 8 +
    ssc * 0.15 +
    hsc * 0.15 +
    communication * 4 +
    technical_score * 0.3 +
    mock_interviews * 3 +
    github_repos * 2 +
    competitive_coding * 4 -
    backlogs * 7 +
    np.random.normal(0, 8, n)
)

threshold = np.percentile(score, 42)
placement_status = (score > threshold).astype(int)

# Salary (only for placed students)
base_salary = 3.0
salary = np.where(
    placement_status == 1,
    np.round(np.clip(
        base_salary + cgpa * 0.6 + aptitude * 0.04 + technical_score * 0.03 +
        internships * 0.5 + np.random.normal(0, 0.8, n), 2.5, 18.0
    ), 2),
    0.0
)

df = pd.DataFrame({
    'StudentID': [f'STU{str(i+1).zfill(4)}' for i in range(n)],
    'CGPA': cgpa,
    'Internships': internships,
    'Projects': projects,
    'Workshops_Certifications': workshops,
    'AptitudeTestScore': aptitude,
    'SoftSkillsRating': soft_skills,
    'ExtracurricularActivities': extracurricular,
    'PlacementTraining': placement_training,
    'SSC_Marks': ssc,
    'HSC_Marks': hsc,
    'Backlogs': backlogs,
    'CommunicationScore': communication,
    'TechnicalScore': technical_score,
    'MockInterviews': mock_interviews,
    'GitHub_Repos': github_repos,
    'CompetitiveCoding': competitive_coding,
    'PlacementStatus': placement_status,
    'PackageLPA': salary
})

df.to_csv('/home/claude/placement_system/data/placement_data.csv', index=False)
print(f"Dataset created: {n} students, {placement_status.sum()} placed ({placement_status.mean()*100:.1f}%)")
print(df.head())
