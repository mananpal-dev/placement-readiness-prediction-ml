import pandas as pd
import joblib

# Load trained model
model = joblib.load("../models/placement_model.pkl")

columns = [
    'CGPA',
    'Internships',
    'Projects',
    'Workshops/Certifications',
    'AptitudeTestScore',
    'SoftSkillsRating',
    'ExtracurricularActivities',
    'PlacementTraining',
    'SSC_Marks',
    'HSC_Marks'
]

# Example student
new_student = [[7.5, 1, 2, 2, 85, 4.5, 1, 1, 78, 82]]

df = pd.DataFrame(new_student, columns=columns)

prediction = model.predict(df)

if prediction[0] == 1:
    print("Student is PLACEMENT READY")
else:
    print("Student is NOT READY")