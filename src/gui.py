"""
PlaceIQ · GUI Predictor (Tkinter)
Upgraded desktop GUI with multi-model support and explainability.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from preprocess import engineer_features

BASE    = Path(__file__).resolve().parent.parent
MODELS  = BASE / "models"
SCALER  = MODELS / "scaler.pkl"
FEATURE_FILE = MODELS / "feature_names.pkl"

# ── Load resources ─────────────────────────────────────────────────────────
def load_models():
    model_paths = {
        "Random Forest":       MODELS / "rf_model.pkl",
        "Logistic Regression": MODELS / "lr_model.pkl",
        "SVM":                 MODELS / "svm_model.pkl",
        "KNN":                 MODELS / "knn_model.pkl",
        "Gradient Boosting":   MODELS / "gb_model.pkl",
        "Stacking Ensemble":   MODELS / "stacking_model.pkl",
    }
    loaded = {}
    for name, p in model_paths.items():
        if p.exists():
            loaded[name] = joblib.load(p)
    return loaded

MODELS_DICT = load_models()
SCALER_OBJ  = joblib.load(SCALER) if SCALER.exists() else None
FEATURES    = joblib.load(FEATURE_FILE) if FEATURE_FILE.exists() else []

# ── Colours ─────────────────────────────────────────────────────────────────
BG       = "#0d0f1a"
CARD     = "#141627"
BORDER   = "#1e2235"
ACCENT   = "#6c63ff"
ACCENT2  = "#00d4aa"
WARN     = "#ff6b6b"
TEXT     = "#e8eaf6"
MUTED    = "#7c83a0"


class PlaceIQApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PlaceIQ · AI Placement Intelligence")
        self.geometry("980x760")
        self.config(bg=BG)
        self.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=ACCENT, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎓  PlaceIQ · AI Placement Intelligence System",
                 font=("Segoe UI", 18, "bold"), bg=ACCENT, fg="white").pack(side="left", padx=20)
        tk.Label(hdr, text="by Manan Pal · B.Tech CSE",
                 font=("Segoe UI", 9), bg=ACCENT, fg="#c5c3ff").pack(side="right", padx=20)

        # Notebook tabs
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",          background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",      background=CARD, foreground=MUTED,
                         padding=[14, 6], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",            background=[("selected", ACCENT)],
                                               foreground=[("selected", "white")])
        style.configure("TFrame",             background=BG)
        style.configure("TLabel",             background=BG, foreground=TEXT)
        style.configure("TButton",            background=ACCENT, foreground="white",
                         font=("Segoe UI", 11, "bold"), padding=10)
        style.map("TButton",                  background=[("active", "#8b85ff")])
        style.configure("TCombobox",          fieldbackground=CARD, background=CARD,
                         foreground=TEXT, selectbackground=ACCENT)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: Predict
        self.predict_tab = ttk.Frame(nb)
        nb.add(self.predict_tab, text="  🔮 Predict  ")
        self._build_predict_tab()

        # Tab 2: About
        about_tab = ttk.Frame(nb)
        nb.add(about_tab, text="  ℹ️ About  ")
        self._build_about_tab(about_tab)

    def _build_predict_tab(self):
        frame = self.predict_tab
        canvas = tk.Canvas(frame, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.inner = tk.Frame(canvas, bg=BG)
        self.inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.entries = {}
        self._section(self.inner, "🎓 Academic Profile")
        self._slider(self.inner, "CGPA",     4.0, 10.0, 7.5, 0.1)
        self._slider(self.inner, "SSC_Marks", 40.0, 100.0, 75.0, 0.5)
        self._slider(self.inner, "HSC_Marks", 40.0, 100.0, 72.0, 0.5)
        self._spinbox(self.inner, "Backlogs", 0, 10, 0)

        self._section(self.inner, "💻 Technical Skills")
        self._slider(self.inner, "AptitudeTestScore", 0, 100, 70, 0.5)
        self._slider(self.inner, "TechnicalScore",    0, 100, 65, 0.5)
        self._spinbox(self.inner, "GitHub_Repos",    0, 50, 3)
        self._spinbox(self.inner, "CompetitiveCoding", 0, 3, 1)
        self._spinbox(self.inner, "Projects",         0, 20, 2)

        self._section(self.inner, "🌟 Soft Skills")
        self._slider(self.inner, "SoftSkillsRating",   1.0, 5.0, 3.5, 0.1)
        self._slider(self.inner, "CommunicationScore", 1.0, 5.0, 3.5, 0.1)
        self._spinbox(self.inner, "MockInterviews",    0, 20, 2)

        self._section(self.inner, "🏆 Experience & Activities")
        self._spinbox(self.inner, "Internships",             0, 10, 1)
        self._spinbox(self.inner, "Workshops_Certifications",0, 20, 2)
        self._toggle(self.inner, "ExtracurricularActivities")
        self._toggle(self.inner, "PlacementTraining")

        self._section(self.inner, "🤖 Model Selection")
        tk.Label(self.inner, text="Select Model:", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=20)
        self.model_var = tk.StringVar(value=list(MODELS_DICT.keys())[0])
        model_cb = ttk.Combobox(self.inner, textvariable=self.model_var,
                                 values=list(MODELS_DICT.keys()), state="readonly",
                                 font=("Segoe UI", 11), width=28)
        model_cb.pack(anchor="w", padx=20, pady=6)

        btn = ttk.Button(self.inner, text="⚡  Predict Placement Readiness",
                         command=self.predict)
        btn.pack(fill="x", padx=20, pady=16)

        # Result area
        self.result_frame = tk.Frame(self.inner, bg=CARD, padx=20, pady=18)
        self.result_frame.pack(fill="x", padx=20, pady=(0, 20))
        self.result_label = tk.Label(self.result_frame, text="— Prediction will appear here —",
                                     bg=CARD, fg=MUTED, font=("Segoe UI", 14))
        self.result_label.pack()
        self.detail_label = tk.Label(self.result_frame, text="",
                                     bg=CARD, fg=MUTED, font=("Segoe UI", 10))
        self.detail_label.pack(pady=4)

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=ACCENT, height=2)
        f.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(parent, text=title, bg=BG, fg=ACCENT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=20, pady=(4, 0))

    def _slider(self, parent, key, mn, mx, default, res):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", padx=20, pady=3)
        tk.Label(row, text=key.replace("_", " "), bg=BG, fg=TEXT,
                 font=("Segoe UI", 10), width=26, anchor="w").pack(side="left")
        var = tk.DoubleVar(value=default)
        self.entries[key] = var
        val_lbl = tk.Label(row, textvariable=var, bg=BG, fg=ACCENT2,
                           font=("Segoe UI", 10, "bold"), width=6)
        val_lbl.pack(side="right")
        sl = tk.Scale(row, from_=mn, to=mx, resolution=res, variable=var,
                      orient="horizontal", bg=BG, fg=TEXT, troughcolor=BORDER,
                      activebackground=ACCENT, highlightthickness=0, length=320,
                      showvalue=False, sliderlength=18, sliderrelief="flat")
        sl.pack(side="left", padx=8)

    def _spinbox(self, parent, key, mn, mx, default):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", padx=20, pady=3)
        tk.Label(row, text=key.replace("_", " "), bg=BG, fg=TEXT,
                 font=("Segoe UI", 10), width=26, anchor="w").pack(side="left")
        var = tk.IntVar(value=default)
        self.entries[key] = var
        sb = tk.Spinbox(row, from_=mn, to=mx, textvariable=var, width=6,
                        bg=CARD, fg=ACCENT2, insertbackground=TEXT, relief="flat",
                        font=("Segoe UI", 11, "bold"), buttonbackground=BORDER)
        sb.pack(side="left", padx=8)

    def _toggle(self, parent, key):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", padx=20, pady=3)
        tk.Label(row, text=key.replace("_", " "), bg=BG, fg=TEXT,
                 font=("Segoe UI", 10), width=26, anchor="w").pack(side="left")
        var = tk.IntVar(value=1)
        self.entries[key] = var
        for val, label in [(1, "Yes"), (0, "No")]:
            tk.Radiobutton(row, text=label, variable=var, value=val,
                           bg=BG, fg=TEXT, selectcolor=ACCENT,
                           activebackground=BG, font=("Segoe UI", 10)).pack(side="left", padx=4)

    def _build_about_tab(self, tab):
        info = """
PlaceIQ – AI Placement Intelligence System
==========================================
Version:   2.0  |  Author: Manan Pal  |  B.Tech CSE

Models
------
• Random Forest      (Ensemble, 300 trees)
• Logistic Regression
• Support Vector Machine (RBF kernel)
• K-Nearest Neighbors
• Gradient Boosting
• XGBoost
• LightGBM
• Stacking Ensemble   ← best generalization

Features Used (25 total)
------------------------
Raw (16): CGPA, Internships, Projects, Workshops,
  Aptitude, Soft Skills, Extracurricular, Placement Training,
  SSC, HSC, Backlogs, Communication, Technical Score,
  Mock Interviews, GitHub Repos, Competitive Coding

Engineered (9): Academic Score, Technical Composite,
  Soft Power Index, Experience Index, Readiness Score,
  CGPA Band, CGPA×Aptitude, Tech×Internship, Soft×Comm

Pipeline
--------
• SMOTE oversampling for class balance
• StandardScaler normalisation
• SHAP explainability on best model
• Calibrated probabilities

Usage
-----
Run Streamlit dashboard: streamlit run app.py
Run this GUI:            python src/gui.py
"""
        txt = tk.Text(tab, bg=CARD, fg=TEXT, font=("Courier New", 10),
                      padx=20, pady=16, relief="flat", wrap="word")
        txt.insert("1.0", info)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True, padx=10, pady=10)

    def predict(self):
        try:
            student = {k: v.get() for k, v in self.entries.items()}

            raw_df  = pd.DataFrame([student])
            eng_df  = engineer_features(raw_df)
            feat_c  = [f for f in FEATURES if f in eng_df.columns]
            X       = eng_df[feat_c].values
            if SCALER_OBJ:
                X = SCALER_OBJ.transform(X)

            model   = MODELS_DICT[self.model_var.get()]
            pred    = model.predict(X)[0]
            prob    = model.predict_proba(X)[0]

            placed_pct = prob[1] * 100

            if pred == 1:
                self.result_frame.config(bg="#0d1f1a")
                self.result_label.config(
                    text=f"✅  PLACEMENT READY",
                    fg=ACCENT2, bg="#0d1f1a", font=("Segoe UI", 18, "bold"))
            else:
                self.result_frame.config(bg="#1f0d0d")
                self.result_label.config(
                    text=f"❌  NOT YET READY",
                    fg=WARN, bg="#1f0d0d", font=("Segoe UI", 18, "bold"))

            self.detail_label.config(
                text=f"Placement Probability: {placed_pct:.2f}%   |   Model: {self.model_var.get()}",
                fg=MUTED, bg=self.result_frame.cget("bg"))

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))


if __name__ == "__main__":
    app = PlaceIQApp()
    app.mainloop()
