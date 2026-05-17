"""
🎓 AI-Powered Student Placement Intelligence System
A production-grade ML dashboard for placement readiness prediction.

Author: Manan Pal | B.Tech CSE
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA   = BASE / "data"
SRC    = BASE / "src"
sys.path.insert(0, str(SRC))

from preprocess import engineer_features, FEATURE_COLS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlaceIQ · AI Placement Intelligence",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette & custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

:root {
  --bg:      #0d0f1a;
  --card:    #141627;
  --border:  #1e2235;
  --accent:  #6c63ff;
  --accent2: #00d4aa;
  --warn:    #ff6b6b;
  --text:    #e8eaf6;
  --muted:   #7c83a0;
}

.stApp { background: var(--bg) !important; color: var(--text) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #0a0b15 !important;
  border-right: 1px solid var(--border);
}

/* Main header */
.main-header {
  background: linear-gradient(135deg, #141627 0%, #1a1d35 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 28px 32px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}
.main-header::before {
  content: '';
  position: absolute;
  top: -60%;
  right: -10%;
  width: 400px;
  height: 400px;
  background: radial-gradient(circle, rgba(108,99,255,0.12) 0%, transparent 70%);
  pointer-events: none;
}
.main-header h1 {
  font-size: 2.1rem;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(90deg, #ffffff, #6c63ff, #00d4aa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.main-header p { color: var(--muted); margin: 6px 0 0; font-size: 0.95rem; }

/* Metric cards */
.metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
}
.metric-card .val {
  font-size: 2rem;
  font-weight: 700;
  color: var(--accent2);
}
.metric-card .label {
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

/* Result banner */
.result-ready {
  background: linear-gradient(135deg, rgba(0,212,170,0.12), rgba(0,212,170,0.04));
  border: 1px solid rgba(0,212,170,0.4);
  border-radius: 14px;
  padding: 24px;
  text-align: center;
  margin: 16px 0;
}
.result-not-ready {
  background: linear-gradient(135deg, rgba(255,107,107,0.12), rgba(255,107,107,0.04));
  border: 1px solid rgba(255,107,107,0.4);
  border-radius: 14px;
  padding: 24px;
  text-align: center;
  margin: 16px 0;
}
.result-title { font-size: 1.5rem; font-weight: 700; margin: 0; }
.result-prob  { font-size: 0.9rem; color: var(--muted); margin-top: 6px; }

/* Section titles */
.section-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--accent);
  border-left: 3px solid var(--accent);
  padding-left: 10px;
  margin: 20px 0 14px;
}

/* Tabs */
.stTabs [role="tablist"] { gap: 8px; }
.stTabs [role="tab"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--muted) !important;
  padding: 6px 18px !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: white !important;
  border-color: var(--accent) !important;
}

/* Sliders & inputs */
.stSlider > div > div > div > div { background: var(--accent) !important; }
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select {
  background: var(--card) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #8b85ff) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 12px 32px !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  width: 100% !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(108,99,255,0.4) !important;
}

/* Skill bar container */
.skill-bar-wrap { margin-bottom: 8px; }
.skill-bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: var(--muted);
  margin-bottom: 4px;
}
.skill-bar-outer {
  background: var(--border);
  border-radius: 6px;
  height: 8px;
  overflow: hidden;
}
.skill-bar-inner {
  height: 100%;
  border-radius: 6px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  transition: width 0.6s ease;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* Alert boxes */
.tip-box {
  background: rgba(108,99,255,0.08);
  border: 1px solid rgba(108,99,255,0.25);
  border-radius: 10px;
  padding: 14px 18px;
  margin: 10px 0;
  font-size: 0.88rem;
  color: #c5c3ff;
}
</style>
""", unsafe_allow_html=True)


# ── Load resources ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    paths = {
        "Random Forest":       MODELS / "rf_model.pkl",
        "Logistic Regression": MODELS / "lr_model.pkl",
        "SVM":                 MODELS / "svm_model.pkl",
        "KNN":                 MODELS / "knn_model.pkl",
        "Gradient Boosting":   MODELS / "gb_model.pkl",
        "XGBoost":             MODELS / "xgb_model.pkl",
        "LightGBM":            MODELS / "lgb_model.pkl",
        "Stacking Ensemble":   MODELS / "stacking_model.pkl",
    }
    for name, p in paths.items():
        if p.exists():
            models[name] = joblib.load(p)
    return models

@st.cache_resource
def load_scaler():
    p = MODELS / "scaler.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_data
def load_metadata():
    p = MODELS / "model_metadata.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_dataset():
    p = DATA / "placement_data.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_resource
def load_feature_names():
    p = MODELS / "feature_names.pkl"
    return joblib.load(p) if p.exists() else FEATURE_COLS

models   = load_models()
scaler   = load_scaler()
metadata = load_metadata()
df_raw   = load_dataset()
features = load_feature_names()


# ── Helper: predict ────────────────────────────────────────────────────────────
def predict_student(student_dict: dict, model_name: str):
    raw_df = pd.DataFrame([student_dict])
    eng_df = engineer_features(raw_df)

    feats  = [f for f in features if f in eng_df.columns]
    X      = eng_df[feats].values
    if scaler:
        X = scaler.transform(X)

    model  = models[model_name]
    pred   = model.predict(X)[0]
    prob   = model.predict_proba(X)[0]
    return int(pred), float(prob[1]), float(prob[0])


def readiness_score_from_dict(d: dict) -> float:
    """Quick composite score (0-100) from input for gauge."""
    s = (
        d.get("CGPA", 0) * 4.5 +
        d.get("Internships", 0) * 6 +
        d.get("Projects", 0) * 3 +
        d.get("Workshops_Certifications", 0) * 2 +
        d.get("AptitudeTestScore", 0) * 0.35 +
        d.get("SoftSkillsRating", 0) * 5 +
        d.get("ExtracurricularActivities", 0) * 4 +
        d.get("PlacementTraining", 0) * 8 +
        d.get("SSC_Marks", 0) * 0.1 +
        d.get("HSC_Marks", 0) * 0.1 +
        d.get("CommunicationScore", 0) * 4 +
        d.get("TechnicalScore", 0) * 0.25 +
        d.get("MockInterviews", 0) * 3 +
        d.get("GitHub_Repos", 0) * 2 +
        d.get("CompetitiveCoding", 0) * 4 -
        d.get("Backlogs", 0) * 7
    )
    return round(min(max(s / 1.5, 0), 100), 1)


# ── Plotly theme ────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#e8eaf6"),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 24px;">
      <div style="font-size:2.4rem;">🎓</div>
      <div style="font-size:1.1rem;font-weight:700;color:#6c63ff;">PlaceIQ</div>
      <div style="font-size:0.75rem;color:#7c83a0;">AI Placement Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predict", "📊 Analytics", "⚖️ Compare Models", "📋 Batch Predict"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem;color:#7c83a0;padding:10px 0;">
    <b style="color:#e8eaf6;">Models Available</b><br><br>
    """ + "".join([
        f'<div style="display:flex;justify-content:space-between;margin:3px 0;">'
        f'<span>{n}</span>'
        f'<span style="color:#00d4aa;">✓</span></div>'
        for n in models.keys()
    ]) + """
    </div>
    """, unsafe_allow_html=True)

    if metadata:
        best = metadata.get("best_model", "—")
        auc  = metadata.get("metrics", {}).get(best, {}).get("roc_auc", "—")
        st.markdown(f"""
        <div style="background:#141627;border:1px solid #1e2235;border-radius:10px;padding:14px;margin-top:12px;">
          <div style="font-size:0.75rem;color:#7c83a0;text-transform:uppercase;letter-spacing:.08em;">Best Model</div>
          <div style="font-weight:600;color:#6c63ff;margin-top:4px;">{best}</div>
          <div style="font-size:0.8rem;color:#00d4aa;">ROC-AUC: {auc}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built by **Manan Pal** · B.Tech CSE")


# ══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
      <h1>🎓 PlaceIQ · AI Placement Intelligence</h1>
      <p>Multi-model machine learning system for student placement readiness prediction with explainability & insights</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    info  = metadata.get("dataset_info", {})
    mets  = metadata.get("metrics", {})
    best  = metadata.get("best_model", "Random Forest")
    b_met = mets.get(best, {})

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, info.get("total_samples", 2000), "Students"),
        (c2, info.get("num_features", 25), "Features"),
        (c3, len(models), "ML Models"),
        (c4, f"{b_met.get('roc_auc', 94.7)}%", "Best AUC"),
        (c5, f"{info.get('placement_rate', 58)}%", "Placement Rate"),
    ]
    for col, val, label in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="val">{val}</div>
              <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature overview + model leaderboard
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="section-title">📌 Feature Categories</div>', unsafe_allow_html=True)
        cats = {
            "🎓 Academic": ["CGPA", "SSC Marks", "HSC Marks", "Backlogs"],
            "💻 Technical": ["Aptitude Score", "Technical Score", "GitHub Repos", "Competitive Coding"],
            "🌟 Soft Skills": ["Soft Skills Rating", "Communication Score", "Mock Interviews"],
            "🏆 Experience": ["Internships", "Projects", "Workshops/Certs", "Placement Training"],
            "🎯 Engineered": ["Readiness Score", "Academic Composite", "Tech Composite", "+6 more"],
        }
        for cat, feats in cats.items():
            with st.expander(cat):
                for f in feats:
                    st.markdown(f"• {f}")

    with right:
        st.markdown('<div class="section-title">🏆 Model Leaderboard</div>', unsafe_allow_html=True)
        if mets:
            rows = []
            for m, v in mets.items():
                rows.append({"Model": m, "Accuracy": v['accuracy'], "F1": v['f1'], "ROC-AUC": v['roc_auc']})
            lb_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
            lb_df.index += 1

            fig = go.Figure(data=[
                go.Bar(
                    y=lb_df["Model"],
                    x=lb_df["ROC-AUC"],
                    orientation='h',
                    marker=dict(
                        color=lb_df["ROC-AUC"],
                        colorscale=[[0,"#2d3561"],[0.5,"#6c63ff"],[1,"#00d4aa"]],
                        showscale=False
                    ),
                    text=[f"{v}%" for v in lb_df["ROC-AUC"]],
                    textposition="inside",
                    textfont=dict(color="white", size=11)
                )
            ])
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=300,
                xaxis=dict(range=[75, 100], gridcolor="#1e2235"),
                yaxis=dict(gridcolor="#1e2235"),
                xaxis_title="ROC-AUC (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📖 How It Works</div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    steps = [
        ("1️⃣", "Input", "Enter student academic, technical & soft-skill data"),
        ("2️⃣", "Engineer", "25 features including 9 engineered composites"),
        ("3️⃣", "Predict", "8 ML models produce calibrated probability scores"),
        ("4️⃣", "Explain", "SHAP-powered feature importance & gap analysis"),
    ]
    for col, (ico, title, desc) in zip([h1,h2,h3,h4], steps):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;">
              <div style="font-size:1.5rem;">{ico}</div>
              <div style="font-weight:600;margin:6px 0 4px;">{title}</div>
              <div style="font-size:0.8rem;color:#7c83a0;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("""
    <div class="main-header">
      <h1>🔮 Placement Readiness Predictor</h1>
      <p>Fill in the student profile below for an AI-powered assessment</p>
    </div>
    """, unsafe_allow_html=True)

    model_name = st.selectbox("🤖 Select ML Model", list(models.keys()), index=0)

    with st.form("predict_form"):
        st.markdown('<div class="section-title">🎓 Academic Profile</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            cgpa = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
        with a2:
            ssc  = st.slider("SSC Marks (%)", 40.0, 100.0, 75.0, 0.5)
        with a3:
            hsc  = st.slider("HSC Marks (%)", 40.0, 100.0, 72.0, 0.5)

        b1, b2 = st.columns(2)
        with b1:
            backlogs = st.number_input("Active Backlogs", 0, 10, 0)
        with b2:
            cgpa_band = st.empty()

        st.markdown('<div class="section-title">💻 Technical Skills</div>', unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        with t1:
            aptitude = st.slider("Aptitude Test Score", 0.0, 100.0, 70.0, 0.5)
        with t2:
            tech     = st.slider("Technical Score", 0.0, 100.0, 65.0, 0.5)
        with t3:
            github   = st.number_input("GitHub Repositories", 0, 50, 3)

        t4, t5 = st.columns(2)
        with t4:
            comp_coding = st.number_input("Competitive Coding (LeetCode/CF level 0–3)", 0, 3, 1)
        with t5:
            projects = st.number_input("Projects Completed", 0, 20, 2)

        st.markdown('<div class="section-title">🌟 Soft Skills</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            soft     = st.slider("Soft Skills Rating (1-5)", 1.0, 5.0, 3.5, 0.1)
        with s2:
            comm     = st.slider("Communication Score (1-5)", 1.0, 5.0, 3.5, 0.1)
        with s3:
            mock     = st.number_input("Mock Interviews Attended", 0, 20, 2)

        st.markdown('<div class="section-title">🏆 Experience & Activities</div>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        with e1:
            internships = st.number_input("Internships", 0, 10, 1)
        with e2:
            workshops   = st.number_input("Workshops / Certifications", 0, 20, 2)
        with e3:
            extra       = st.selectbox("Extracurricular Activities", [1, 0], format_func=lambda x: "Yes" if x else "No")

        placement_tr = st.selectbox("Completed Placement Training?", [1, 0], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("⚡ Predict Placement Readiness", use_container_width=True)

    if submitted:
        student = {
            "CGPA": cgpa, "Internships": internships, "Projects": projects,
            "Workshops_Certifications": workshops, "AptitudeTestScore": aptitude,
            "SoftSkillsRating": soft, "ExtracurricularActivities": extra,
            "PlacementTraining": placement_tr, "SSC_Marks": ssc, "HSC_Marks": hsc,
            "Backlogs": backlogs, "CommunicationScore": comm, "TechnicalScore": tech,
            "MockInterviews": mock, "GitHub_Repos": github, "CompetitiveCoding": comp_coding
        }

        pred, prob_placed, prob_not = predict_student(student, model_name)
        rs = readiness_score_from_dict(student)

        # Result banner
        if pred == 1:
            st.markdown(f"""
            <div class="result-ready">
              <div class="result-title" style="color:#00d4aa;">✅ PLACEMENT READY</div>
              <div class="result-prob">Placement Probability: <b>{prob_placed*100:.1f}%</b> &nbsp;|&nbsp; Model: {model_name}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-not-ready">
              <div class="result-title" style="color:#ff6b6b;">❌ NOT YET READY</div>
              <div class="result-prob">Placement Probability: <b>{prob_placed*100:.1f}%</b> &nbsp;|&nbsp; Model: {model_name}</div>
            </div>
            """, unsafe_allow_html=True)

        # Metrics row
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""<div class="metric-card">
              <div class="val" style="color:#6c63ff;">{prob_placed*100:.1f}%</div>
              <div class="label">Placed Prob.</div></div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""<div class="metric-card">
              <div class="val" style="color:#ff6b6b;">{prob_not*100:.1f}%</div>
              <div class="label">Not Placed Prob.</div></div>""", unsafe_allow_html=True)
        with mc3:
            color = "#00d4aa" if rs >= 60 else "#f5a623" if rs >= 40 else "#ff6b6b"
            st.markdown(f"""<div class="metric-card">
              <div class="val" style="color:{color};">{rs}</div>
              <div class="label">Readiness Score</div></div>""", unsafe_allow_html=True)
        with mc4:
            grade = "A+" if rs >= 80 else "A" if rs >= 70 else "B" if rs >= 60 else "C" if rs >= 45 else "D"
            st.markdown(f"""<div class="metric-card">
              <div class="val">{grade}</div>
              <div class="label">Grade</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns([1.2, 1])

        with left:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_placed * 100,
                number={'suffix': "%", 'font': {'size': 36, 'color': '#e8eaf6'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#7c83a0'},
                    'bar': {'color': "#6c63ff", 'thickness': 0.22},
                    'bgcolor': "#141627",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40],  'color': '#2a1a1a'},
                        {'range': [40, 65], 'color': '#1a1f2a'},
                        {'range': [65, 100],'color': '#131f1b'},
                    ],
                    'threshold': {
                        'line': {'color': "#00d4aa", 'width': 3},
                        'thickness': 0.75,
                        'value': 65
                    }
                },
                title={'text': "Placement Probability", 'font': {'color': '#7c83a0', 'size': 14}}
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with right:
            # Radar chart of student skills
            categories = ['CGPA\n(/10)', 'Aptitude\n(/100)', 'Technical\n(/100)',
                          'Soft Skills\n(/5)', 'Communication\n(/5)', 'Experience']
            vals_norm = [
                cgpa / 10 * 100,
                aptitude,
                tech,
                soft / 5 * 100,
                comm / 5 * 100,
                min((internships * 15 + workshops * 5 + mock * 4 + projects * 5) / 1.5, 100)
            ]
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals_norm + [vals_norm[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(108,99,255,0.2)',
                line=dict(color='#6c63ff', width=2),
                name="Student"
            ))
            fig_radar.update_layout(
                **PLOTLY_LAYOUT,
                polar=dict(
                    bgcolor="#141627",
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1e2235", tickcolor="#7c83a0"),
                    angularaxis=dict(gridcolor="#1e2235", tickcolor="#7c83a0")
                ),
                height=280,
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Improvement recommendations
        st.markdown('<div class="section-title">💡 AI-Powered Improvement Recommendations</div>', unsafe_allow_html=True)

        recs = []
        if cgpa < 7.0:        recs.append(("📚", "CGPA", f"Your CGPA is {cgpa}. Aim for 7.5+ to be competitive. Focus on core subjects."))
        if aptitude < 65:     recs.append(("🧠", "Aptitude", f"Score {aptitude}/100. Practice quantitative & logical reasoning daily. Target 75+."))
        if internships == 0:  recs.append(("💼", "Internships", "No internships found. Apply to at least 1 industry internship (virtual or in-person)."))
        if github < 3:        recs.append(("🐙", "GitHub", f"Only {github} repos. Recruiters check GitHub. Build 5+ meaningful projects."))
        if mock == 0:         recs.append(("🎤", "Mock Interviews", "Attend mock interviews — they dramatically improve placement chances."))
        if backlogs > 0:      recs.append(("⚠️", "Backlogs", f"You have {backlogs} backlog(s). Clear them immediately — they're a red flag."))
        if tech < 60:         recs.append(("💻", "Technical Skills", "Strengthen DSA, DBMS, OS, and computer networks fundamentals."))
        if soft < 3.0:        recs.append(("🗣️", "Soft Skills", "Work on communication, teamwork, and problem-solving presentation."))
        if comp_coding == 0:  recs.append(("⚔️", "Competitive Coding", "Join LeetCode/Codeforces. Even beginner level coding contests add value."))
        if placement_tr == 0: recs.append(("🏫", "Placement Training", "Enroll in your college's placement training program ASAP."))
        if not recs:
            recs.append(("🌟", "Outstanding!", "Your profile is excellent. Focus on mock interviews and company research."))

        for ico, area, msg in recs[:6]:
            st.markdown(f"""
            <div class="tip-box">
              <b>{ico} {area}:</b> {msg}
            </div>
            """, unsafe_allow_html=True)

        # Skill bars
        st.markdown('<div class="section-title">📊 Profile Breakdown</div>', unsafe_allow_html=True)
        skill_items = [
            ("Academic", (cgpa/10*50 + ssc*0.25 + hsc*0.25)/2),
            ("Technical", (tech*0.4 + aptitude*0.3 + github*3 + comp_coding*8)/1.4),
            ("Soft Skills", (soft/5*50 + comm/5*50)),
            ("Experience", min((internships*15 + workshops*5 + projects*5 + mock*3)/1.0, 100)),
            ("Overall Readiness", rs),
        ]
        for name, score in skill_items:
            s = min(max(score, 0), 100)
            color = "#00d4aa" if s >= 65 else "#f5a623" if s >= 45 else "#ff6b6b"
            st.markdown(f"""
            <div class="skill-bar-wrap">
              <div class="skill-bar-label"><span>{name}</span><span>{s:.1f}%</span></div>
              <div class="skill-bar-outer">
                <div class="skill-bar-inner" style="width:{s}%;background:linear-gradient(90deg,{color}99,{color});"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("""
    <div class="main-header">
      <h1>📊 Dataset Analytics & Insights</h1>
      <p>Explore patterns in student placement data</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw.empty:
        st.warning("Dataset not found.")
        st.stop()

    placed = df_raw[df_raw["PlacementStatus"] == 1]
    not_placed = df_raw[df_raw["PlacementStatus"] == 0]

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "📦 Feature Impact", "💰 Package Analysis"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Placement donut
            fig = go.Figure(go.Pie(
                labels=["Placed", "Not Placed"],
                values=[len(placed), len(not_placed)],
                hole=0.65,
                marker=dict(colors=["#00d4aa", "#ff6b6b"]),
                textinfo="label+percent",
                textfont=dict(color="white")
            ))
            fig.add_annotation(text=f"{len(df_raw)}<br>students", x=0.5, y=0.5,
                               font=dict(size=16, color="white"), showarrow=False)
            fig.update_layout(**PLOTLY_LAYOUT, height=320, title="Placement Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # CGPA distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=placed["CGPA"], name="Placed",
                                       marker_color="#00d4aa", opacity=0.7, nbinsx=25))
            fig.add_trace(go.Histogram(x=not_placed["CGPA"], name="Not Placed",
                                       marker_color="#ff6b6b", opacity=0.7, nbinsx=25))
            fig.update_layout(**PLOTLY_LAYOUT, height=320, barmode="overlay",
                              title="CGPA Distribution by Placement",
                              xaxis_title="CGPA", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        # Box plots
        features_to_plot = ["CGPA", "AptitudeTestScore", "TechnicalScore", "SoftSkillsRating"]
        fig = make_subplots(rows=1, cols=4, subplot_titles=features_to_plot)
        colors = {"Placed": "#00d4aa", "Not Placed": "#ff6b6b"}
        for i, feat in enumerate(features_to_plot):
            for status, grp, color in [("Placed", placed, "#00d4aa"), ("Not Placed", not_placed, "#ff6b6b")]:
                fig.add_trace(
                    go.Box(y=grp[feat], name=status, marker_color=color,
                           line_color=color, showlegend=(i == 0)),
                    row=1, col=i+1
                )
        fig.update_layout(**PLOTLY_LAYOUT, height=340, title="Key Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        numeric_cols = [c for c in df_raw.select_dtypes(include=np.number).columns
                        if c not in ["StudentID", "PackageLPA"]]
        corr = df_raw[numeric_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns, y=corr.columns,
            colorscale=[[0,"#ff6b6b"],[0.5,"#141627"],[1,"#00d4aa"]],
            zmid=0, text=np.round(corr.values, 2),
            texttemplate="%{text}", textfont=dict(size=9)
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=520, title="Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        shap_data = {}
        sp = MODELS / "shap_data.pkl"
        if sp.exists():
            sd = joblib.load(sp)
            shap_importance = sd.get("shap_importance", {})
        else:
            shap_importance = metadata.get("shap_importance", {})

        if shap_importance:
            # Flatten any list values (e.g. from older SHAP storage format)
            shap_flat = {
                k: float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)
                for k, v in shap_importance.items()
            }
            si = pd.DataFrame(list(shap_flat.items()), columns=["Feature", "SHAP Value"])
            si = si.sort_values("SHAP Value", ascending=False).head(15)
            si["SHAP Value"] = pd.to_numeric(si["SHAP Value"], errors="coerce").fillna(0.0)
            shap_vals = si["SHAP Value"].tolist()
            fig = go.Figure(go.Bar(
                y=si["Feature"].tolist(), x=shap_vals, orientation='h',
                marker=dict(color=shap_vals,
                            colorscale=[[0,"#2d3561"],[1,"#00d4aa"]], showscale=False),
                text=[f"{v:.4f}" for v in shap_vals], textposition="inside",
                textfont=dict(color="white")
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=460, title="SHAP Feature Importance (Top 15)",
                              xaxis_title="Mean |SHAP value|")
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: CGPA vs AptitudeTestScore
        fig = px.scatter(
            df_raw, x="CGPA", y="AptitudeTestScore",
            color=df_raw["PlacementStatus"].map({1:"Placed", 0:"Not Placed"}),
            color_discrete_map={"Placed": "#00d4aa", "Not Placed": "#ff6b6b"},
            opacity=0.6, title="CGPA vs Aptitude Score",
            size_max=6
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        pkg = placed[placed["PackageLPA"] > 0]["PackageLPA"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pkg, nbinsx=30, marker_color="#6c63ff",
                                   name="Package Distribution"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          title="Salary Package Distribution (Placed Students)",
                          xaxis_title="Package (LPA)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            # Package vs CGPA
            fig = px.scatter(placed, x="CGPA", y="PackageLPA",
                             color="TechnicalScore", color_continuous_scale="Viridis",
                             title="CGPA vs Package (colored by Tech Score)")
            fig.update_layout(**PLOTLY_LAYOUT, height=340)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            bins = pd.cut(placed["PackageLPA"], bins=[0,5,8,12,18], labels=["<5 LPA","5-8 LPA","8-12 LPA","12+ LPA"])
            vc   = bins.value_counts().sort_index()
            fig  = go.Figure(go.Bar(x=vc.index, y=vc.values,
                                    marker_color=["#ff6b6b","#f5a623","#6c63ff","#00d4aa"],
                                    text=vc.values, textposition="outside",
                                    textfont=dict(color="white")))
            fig.update_layout(**PLOTLY_LAYOUT, height=340, title="Package Tier Distribution")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARE MODELS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Compare Models":
    st.markdown("""
    <div class="main-header">
      <h1>⚖️ Model Comparison Dashboard</h1>
      <p>Compare all ML models across key performance metrics</p>
    </div>
    """, unsafe_allow_html=True)

    mets = metadata.get("metrics", {})
    if not mets:
        st.error("No model metrics found. Please train models first.")
        st.stop()

    df_mets = pd.DataFrame([
        {"Model": k, **{kk: vv for kk, vv in v.items() if kk != "cm"}}
        for k, v in mets.items()
    ]).sort_values("roc_auc", ascending=False)

    # Metric selector
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
    selected_m  = st.multiselect("Select metrics to compare:", metric_cols,
                                  default=["accuracy", "f1", "roc_auc"])

    if selected_m:
        fig = go.Figure()
        colors = ["#6c63ff","#00d4aa","#ff6b6b","#f5a623","#a78bfa","#34d399"]
        for i, m in enumerate(selected_m):
            fig.add_trace(go.Bar(
                name=m.replace("_"," ").title(),
                x=df_mets["Model"], y=df_mets[m],
                marker_color=colors[i % len(colors)],
                text=[f"{v}%" for v in df_mets[m]],
                textposition="outside", textfont=dict(color="white", size=10)
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=420, barmode="group",
            yaxis=dict(range=[70, 100], gridcolor="#1e2235"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title="Model Performance Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Radar comparison
    st.markdown('<div class="section-title">🕸️ Multi-Metric Radar</div>', unsafe_allow_html=True)
    radar_models = st.multiselect("Select models for radar chart:", list(mets.keys()),
                                   default=list(mets.keys())[:4])
    if radar_models:
        radar_dims = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        fig_r = go.Figure()
        colors_r = [
            ("108,99,255"), ("0,212,170"), ("255,107,107"), ("245,166,35"),
            ("167,139,250"), ("52,211,153"), ("244,114,182"), ("251,146,60")
        ]
        for i, m in enumerate(radar_models):
            v = mets[m]
            vals = [v.get(d, 0) for d in radar_dims]
            rgb = colors_r[i % len(colors_r)]
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=[d.replace("_"," ").title() for d in radar_dims] + [radar_dims[0].replace("_"," ").title()],
                fill='toself', fillcolor=f"rgba({rgb},0.2)",
                line=dict(color=f"rgb({rgb})", width=2),
                name=m
            ))
        fig_r.update_layout(
            **PLOTLY_LAYOUT, height=450,
            polar=dict(
                bgcolor="#141627",
                radialaxis=dict(visible=True, range=[75, 100], gridcolor="#1e2235"),
                angularaxis=dict(gridcolor="#1e2235")
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # Detailed metrics table
    st.markdown('<div class="section-title">📋 Detailed Metrics Table</div>', unsafe_allow_html=True)
    display_df = df_mets[["Model", "accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"]].copy()
    display_df.columns = ["Model", "Accuracy%", "Precision%", "Recall%", "F1%", "ROC-AUC%", "Brier↓"]
    display_df = display_df.reset_index(drop=True)
    st.dataframe(display_df.style.background_gradient(
        subset=["Accuracy%","Precision%","Recall%","F1%","ROC-AUC%"], cmap="Greens"
    ).background_gradient(subset=["Brier↓"], cmap="Reds_r"), use_container_width=True)

    # Confusion matrices
    st.markdown('<div class="section-title">🗓️ Confusion Matrices</div>', unsafe_allow_html=True)
    cm_models = list(mets.keys())
    n_cols     = 4
    rows_needed = (len(cm_models) + n_cols - 1) // n_cols
    for ri in range(rows_needed):
        cols = st.columns(n_cols)
        for ci in range(n_cols):
            idx = ri * n_cols + ci
            if idx >= len(cm_models):
                break
            mn = cm_models[idx]
            cm = mets[mn].get("cm", [[0,0],[0,0]])
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["Pred:0","Pred:1"], y=["True:0","True:1"],
                colorscale=[[0,"#141627"],[1,"#6c63ff"]],
                text=cm, texttemplate="%{text}", textfont=dict(size=16, color="white"),
                showscale=False
            ))
            fig_cm.update_layout(**PLOTLY_LAYOUT, height=200, title=mn)
            with cols[ci]:
                st.plotly_chart(fig_cm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH PREDICT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch Predict":
    st.markdown("""
    <div class="main-header">
      <h1>📋 Batch Prediction</h1>
      <p>Upload a CSV file and get predictions for all students at once</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-box">
    📌 <b>Required CSV columns:</b> CGPA, Internships, Projects, Workshops_Certifications,
    AptitudeTestScore, SoftSkillsRating, ExtracurricularActivities, PlacementTraining,
    SSC_Marks, HSC_Marks, Backlogs, CommunicationScore, TechnicalScore, MockInterviews,
    GitHub_Repos, CompetitiveCoding
    </div>
    """, unsafe_allow_html=True)

    model_name = st.selectbox("🤖 Model for Batch Prediction", list(models.keys()))

    uploaded = st.file_uploader("Upload student CSV file", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(batch_df)} students")

            eng_df   = engineer_features(batch_df)
            feat_col = [f for f in features if f in eng_df.columns]
            X_batch  = eng_df[feat_col].values

            if scaler:
                X_batch = scaler.transform(X_batch)

            model = models[model_name]
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)[:, 1]

            out_df = batch_df.copy()
            out_df["Predicted_Status"]      = preds
            out_df["Placement_Probability"] = (probs * 100).round(2)
            out_df["Verdict"]               = out_df["Predicted_Status"].map({1:"✅ Ready", 0:"❌ Not Ready"})

            # Summary
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-card">
                  <div class="val" style="color:#00d4aa;">{preds.sum()}</div>
                  <div class="label">Placement Ready</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                  <div class="val" style="color:#ff6b6b;">{len(preds)-preds.sum()}</div>
                  <div class="label">Not Ready</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                  <div class="val">{probs.mean()*100:.1f}%</div>
                  <div class="label">Avg Probability</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability histogram
            fig = go.Figure(go.Histogram(
                x=probs * 100, nbinsx=25, marker_color="#6c63ff",
                opacity=0.85
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                              title="Placement Probability Distribution",
                              xaxis_title="Probability (%)")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(out_df, use_container_width=True)

            csv = out_df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv,
                               file_name="batch_placement_results.csv",
                               mime="text/csv", use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        # Show sample CSV
        sample_cols = FEATURE_COLS
        sample_data = pd.DataFrame([{
            "CGPA": 7.5, "Internships": 1, "Projects": 2,
            "Workshops_Certifications": 2, "AptitudeTestScore": 80,
            "SoftSkillsRating": 4.0, "ExtracurricularActivities": 1,
            "PlacementTraining": 1, "SSC_Marks": 78, "HSC_Marks": 75,
            "Backlogs": 0, "CommunicationScore": 4.0, "TechnicalScore": 72,
            "MockInterviews": 2, "GitHub_Repos": 5, "CompetitiveCoding": 1
        }])
        st.markdown('<div class="section-title">📄 Sample CSV Format</div>', unsafe_allow_html=True)
        st.dataframe(sample_data, use_container_width=True)
        st.download_button("⬇️ Download Sample CSV Template", sample_data.to_csv(index=False),
                           file_name="sample_students.csv", mime="text/csv", use_container_width=True)
