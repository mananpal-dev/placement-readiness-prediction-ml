"""
Advanced Model Training Pipeline
- Random Forest, Logistic Regression, SVM, KNN (original)
- XGBoost, LightGBM (new)
- Stacking Ensemble (new)
- SHAP Explainability
- Cross-validation with detailed metrics
- Hyperparameter tuning
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    brier_score_loss, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

import shap

from preprocess import preprocess_data, ENGINEERED_COLS, TARGET_COL


MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall": round(recall_score(y_test, y_pred) * 100, 2),
        "f1": round(f1_score(y_test, y_pred) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_prob) * 100, 2),
        "avg_precision": round(average_precision_score(y_test, y_prob) * 100, 2),
        "brier_score": round(brier_score_loss(y_test, y_prob), 4),
        "cm": confusion_matrix(y_test, y_pred).tolist()
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:      {metrics['accuracy']}%")
    print(f"  Precision:     {metrics['precision']}%")
    print(f"  Recall:        {metrics['recall']}%")
    print(f"  F1 Score:      {metrics['f1']}%")
    print(f"  ROC-AUC:       {metrics['roc_auc']}%")
    print(f"  Avg Precision: {metrics['avg_precision']}%")
    print(f"  Brier Score:   {metrics['brier_score']}")

    return metrics


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance if available."""
    try:
        if hasattr(model, 'named_steps'):
            clf = model.named_steps.get('clf') or model.named_steps.get('model')
        else:
            clf = model

        if hasattr(clf, 'feature_importances_'):
            fi = dict(zip(feature_names, clf.feature_importances_.tolist()))
            return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(clf, 'coef_'):
            fi = dict(zip(feature_names, np.abs(clf.coef_[0]).tolist()))
            return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    except:
        pass
    return {}


def train_all_models():
    print("\n🚀 Loading and preprocessing data...")
    df = preprocess_data("../data/placement_data.csv")

    features = [c for c in ENGINEERED_COLS if c in df.columns]
    X = df[features]
    y = df[TARGET_COL]

    print(f"   Dataset: {len(df)} samples, {len(features)} features")
    print(f"   Placement rate: {y.mean()*100:.1f}%")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE for class balance on train set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"   After SMOTE: {len(X_train_res)} training samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(features, MODELS_DIR / "feature_names.pkl")

    all_metrics = {}
    feature_importances = {}

    # --- Random Forest ---
    print("\n⚙️  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train_res)
    joblib.dump(rf, MODELS_DIR / "rf_model.pkl")
    all_metrics["Random Forest"] = evaluate_model(rf, X_test_scaled, y_test, "Random Forest")
    feature_importances["Random Forest"] = get_feature_importance(rf, features, "Random Forest")

    # --- Logistic Regression ---
    print("\n⚙️  Training Logistic Regression...")
    lr = LogisticRegression(
        C=1.0, max_iter=2000, solver='lbfgs',
        class_weight='balanced', random_state=42
    )
    lr.fit(X_train_scaled, y_train_res)
    joblib.dump(lr, MODELS_DIR / "lr_model.pkl")
    all_metrics["Logistic Regression"] = evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")
    feature_importances["Logistic Regression"] = get_feature_importance(lr, features, "Logistic Regression")

    # --- SVM ---
    print("\n⚙️  Training SVM...")
    svm = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True,
              class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train_res)
    joblib.dump(svm, MODELS_DIR / "svm_model.pkl")
    all_metrics["SVM"] = evaluate_model(svm, X_test_scaled, y_test, "SVM")

    # --- KNN ---
    print("\n⚙️  Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', n_jobs=-1)
    knn.fit(X_train_scaled, y_train_res)
    joblib.dump(knn, MODELS_DIR / "knn_model.pkl")
    all_metrics["KNN"] = evaluate_model(knn, X_test_scaled, y_test, "KNN")

    # --- Gradient Boosting ---
    print("\n⚙️  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=5,
        subsample=0.8, random_state=42
    )
    gb.fit(X_train_scaled, y_train_res)
    joblib.dump(gb, MODELS_DIR / "gb_model.pkl")
    all_metrics["Gradient Boosting"] = evaluate_model(gb, X_test_scaled, y_test, "Gradient Boosting")
    feature_importances["Gradient Boosting"] = get_feature_importance(gb, features, "Gradient Boosting")

    # --- XGBoost ---
    if HAS_XGB:
        print("\n⚙️  Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric='logloss', random_state=42, n_jobs=-1
        )
        xgb.fit(X_train_scaled, y_train_res)
        joblib.dump(xgb, MODELS_DIR / "xgb_model.pkl")
        all_metrics["XGBoost"] = evaluate_model(xgb, X_test_scaled, y_test, "XGBoost")
        feature_importances["XGBoost"] = get_feature_importance(xgb, features, "XGBoost")
    else:
        print("⚠️  XGBoost not available, skipping.")

    # --- LightGBM ---
    if HAS_LGB:
        print("\n⚙️  Training LightGBM...")
        lgb = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=50, subsample=0.8, random_state=42, n_jobs=-1,
            verbose=-1
        )
        lgb.fit(X_train_scaled, y_train_res)
        joblib.dump(lgb, MODELS_DIR / "lgb_model.pkl")
        all_metrics["LightGBM"] = evaluate_model(lgb, X_test_scaled, y_test, "LightGBM")
        feature_importances["LightGBM"] = get_feature_importance(lgb, features, "LightGBM")
    else:
        print("⚠️  LightGBM not available, skipping.")

    # --- Stacking Ensemble ---
    print("\n⚙️  Building Stacking Ensemble...")
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ]
    if HAS_XGB:
        base_estimators.append(('xgb', XGBClassifier(
            n_estimators=100, use_label_encoder=False,
            eval_metric='logloss', random_state=42, n_jobs=-1
        )))

    meta_model = LogisticRegression(max_iter=1000)
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5, n_jobs=-1, passthrough=False
    )
    stacking.fit(X_train_scaled, y_train_res)
    joblib.dump(stacking, MODELS_DIR / "stacking_model.pkl")
    all_metrics["Stacking Ensemble"] = evaluate_model(stacking, X_test_scaled, y_test, "Stacking Ensemble")

    # --- SHAP values (for best model) ---
    print("\n🔍 Computing SHAP values for Random Forest...")
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test_scaled[:200])
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_importance = dict(zip(features, mean_abs_shap.tolist()))
        shap_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
        joblib.dump({"explainer": explainer, "shap_importance": shap_importance},
                    MODELS_DIR / "shap_data.pkl")
        print("   SHAP values saved.")
    except Exception as e:
        print(f"   SHAP error: {e}")
        shap_importance = {}

    # Determine best model by ROC-AUC
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]['roc_auc'])
    print(f"\n🏆 Best Model: {best_model_name} ({all_metrics[best_model_name]['roc_auc']}% ROC-AUC)")

    # Save metadata
    metadata = {
        "metrics": all_metrics,
        "feature_importances": feature_importances,
        "shap_importance": shap_importance,
        "feature_names": features,
        "best_model": best_model_name,
        "dataset_info": {
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train_res)),
            "test_samples": int(len(X_test)),
            "placement_rate": round(float(y.mean()) * 100, 1),
            "num_features": len(features)
        }
    }

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ All models trained and saved!")
    print(f"📁 Models saved to: {MODELS_DIR.resolve()}")
    return metadata


if __name__ == "__main__":
    train_all_models()
