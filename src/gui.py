import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

# Load models
rf_model = joblib.load("../models/placement_model.pkl")
lr_model = joblib.load("../models/lr_placement_model.pkl")
svm_model = joblib.load("../models/svm_placement_model.pkl")
knn_model = joblib.load("../models/knn_placement_model.pkl")

root = tk.Tk()
root.title("AI Placement Intelligence System")
root.geometry("900x650")
root.config(bg="#1e1e2f")

title = tk.Label(
    root,
    text="AI Based Placement Intelligence System",
    font=("Segoe UI", 18, "bold"),
    bg="#1e1e2f",
    fg="#00ffcc"
)
title.pack(pady=15)

frame = tk.Frame(root, bg="#2b2b3c", padx=20, pady=20)
frame.pack()

labels = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "Aptitude Test Score",
    "Soft Skills Rating",
    "Extracurricular Activities (0/1)",
    "Placement Training (0/1)",
    "SSC Marks",
    "HSC Marks"
]

entries = []

for i in range(len(labels)):
    label = tk.Label(frame, text=labels[i], bg="#2b2b3c", fg="white")
    label.grid(row=i, column=0, pady=5)

    entry = tk.Entry(frame)
    entry.grid(row=i, column=1, pady=5)

    entries.append(entry)

model_choice = ttk.Combobox(
    frame,
    values=["Random Forest", "Logistic Regression", "SVM", "KNN"],
    state="readonly"
)
model_choice.grid(row=10, column=1)
model_choice.current(0)

result_label = tk.Label(
    root,
    text="Prediction Result",
    font=("Segoe UI", 14),
    bg="#1e1e2f",
    fg="yellow"
)
result_label.pack(pady=20)


def predict():

    try:

        values = []

        for entry in entries:
            values.append(float(entry.get()))

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

        df = pd.DataFrame([values], columns=columns)

        model_name = model_choice.get()

        if model_name == "Random Forest":
            model = rf_model
        elif model_name == "Logistic Regression":
            model = lr_model
        elif model_name == "SVM":
            model = svm_model
        else:
            model = knn_model

        prediction = model.predict(df)
        probability = model.predict_proba(df)

        prob = probability[0][1] * 100

        if prediction[0] == 1:
            result_label.config(
                text=f"STUDENT IS PLACEMENT READY\nProbability: {prob:.2f}%"
            )
        else:
            result_label.config(
                text=f"STUDENT IS NOT PLACEMENT READY\nProbability: {prob:.2f}%"
            )

    except:
        messagebox.showerror("Error", "Enter valid numbers")


predict_btn = ttk.Button(root, text="Predict", command=predict)
predict_btn.pack(pady=10)

root.mainloop()