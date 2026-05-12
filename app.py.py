import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
from catboost import CatBoostClassifier
import joblib
import os

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #c0392b;
        text-align: center;
        padding: 1rem 0 0.3rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #fff5f5, #fff);
        border: 1px solid #f5c6cb;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high {
        background-color: #f8d7da;
        border-left: 5px solid #c0392b;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #721c24;
    }
    .risk-low {
        background-color: #d4edda;
        border-left: 5px solid #27ae60;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #155724;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🫀 CardioRisk AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">CSS 324 · Introduction to Machine Learning · Final Project<br>Cardiovascular Disease Prediction using CatBoost</div>', unsafe_allow_html=True)

# ─── Sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "📊 Data Explorer",
    "📈 EDA",
    "🤖 Model Comparison",
    "🔮 Predict Risk"
])

DATASET_PATH = "70000rowdataset.csv"
MODEL_PATH   = "final_cvd_model.pkl"

# ─── Data loading & preprocessing (cached) ──────────────────────────────────
@st.cache_data
def load_and_clean(path):
    df = pd.read_csv(path, sep=';')
    df = df.drop_duplicates()
    df['age_years'] = (df['age'] / 365).astype(int)
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    df = df[(df['height'] >= 120) & (df['height'] <= 220)]
    df = df[(df['weight'] >= 30)  & (df['weight'] <= 200)]
    df = df[(df['ap_hi'] >= 80)   & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 40)   & (df['ap_lo'] <= 160)]
    df = df[df['ap_hi'] > df['ap_lo']]
    return df

@st.cache_resource
def train_models(df):
    drop_cols = [c for c in ['id', 'age'] if c in df.columns]
    X = df.drop(['cardio'] + drop_cols, axis=1)
    y = df['cardio']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    def eval_model(m, Xtr, Xte, name, scaled=False):
        Xtr_ = X_train_sc if scaled else Xtr
        Xte_ = X_test_sc  if scaled else Xte
        m.fit(Xtr_, y_train)
        pred = m.predict(Xte_)
        prob = m.predict_proba(Xte_)[:,1]
        return {
            'Model': name,
            'Accuracy':  round(accuracy_score(y_test, pred)*100, 2),
            'Precision': round(precision_score(y_test, pred)*100, 2),
            'Recall':    round(recall_score(y_test, pred)*100, 2),
            'F1 Score':  round(f1_score(y_test, pred)*100, 2),
            'ROC AUC':   round(roc_auc_score(y_test, prob), 4),
        }

    results = []
    results.append(eval_model(LogisticRegression(max_iter=1000, random_state=42),
                               X_train, X_test, 'Logistic Regression', scaled=True))
    results.append(eval_model(DecisionTreeClassifier(max_depth=6, random_state=42),
                               X_train, X_test, 'Decision Tree'))
    results.append(eval_model(RandomForestClassifier(n_estimators=100, random_state=42),
                               X_train, X_test, 'Random Forest'))
    results.append(eval_model(KNeighborsClassifier(n_neighbors=5),
                               X_train, X_test, 'KNN', scaled=True))
    results.append(eval_model(AdaBoostClassifier(n_estimators=100, random_state=42),
                               X_train, X_test, 'AdaBoost'))
    results.append(eval_model(GradientBoostingClassifier(n_estimators=100, random_state=42),
                               X_train, X_test, 'Gradient Boosting'))

    cat = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                              verbose=0, random_state=42)
    results.append(eval_model(cat, X_train, X_test, 'CatBoost'))

    # Best model (CatBoost)
    best = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                               verbose=0, random_state=42)
    best.fit(X_train, y_train)

    return pd.DataFrame(results), best, scaler, X.columns.tolist(), X_test, y_test

# ─── Load data ───────────────────────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    st.error(f"Dataset not found at `{DATASET_PATH}`. Please place `70000rowdataset.csv` in the same folder as `app.py`.")
    st.stop()

df = load_and_clean(DATASET_PATH)
results_df, best_model, scaler, feature_cols, X_test, y_test = train_models(df)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
**CardioRisk AI** is a machine-learning system trained to predict cardiovascular
disease (CVD) from clinical and lifestyle features. Cardiovascular disease is the
leading cause of death worldwide — early, data-driven detection can save lives.

**Pipeline summary:**
- **Dataset:** 70,000-row cardiovascular health records (Kaggle)
- **Cleaning:** removed duplicates, outliers in height/weight/blood pressure
- **Feature engineering:** `age_years` from days, BMI calculation
- **Models trained:** 7 classifiers compared; **CatBoost** selected as best
- **Best ROC AUC:** ≈ 0.80

**Features used:**
`age_years`, `gender`, `height`, `weight`, `BMI`,
`ap_hi` (systolic BP), `ap_lo` (diastolic BP),
`cholesterol`, `gluc`, `smoke`, `alco`, `active`
""")

    with col2:
        total  = len(df)
        cardio = int(df['cardio'].sum())
        healthy = total - cardio
        st.metric("Total Patients", f"{total:,}")
        st.metric("With CVD", f"{cardio:,}", delta=f"{cardio/total*100:.1f}%", delta_color="inverse")
        st.metric("Healthy", f"{healthy:,}", delta=f"{healthy/total*100:.1f}%")

    st.markdown('<div class="section-title">Dataset Quick Peek</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown('<div class="section-title">Dataset Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().T.round(2), use_container_width=True)

    st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Feature', 'Missing Count']
    st.dataframe(missing, use_container_width=True)

    st.markdown('<div class="section-title">Feature Distribution</div>', unsafe_allow_html=True)
    num_cols = ['age_years', 'height', 'weight', 'BMI', 'ap_hi', 'ap_lo']
    chosen = st.selectbox("Select feature", num_cols)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[chosen], kde=True, ax=axes[0], color='#e74c3c')
    axes[0].set_title(f'Distribution of {chosen}')
    sns.boxplot(x='cardio', y=chosen, data=df, ax=axes[1],
                palette=['#27ae60', '#e74c3c'])
    axes[1].set_xticklabels(['No CVD', 'CVD'])
    axes[1].set_title(f'{chosen} by CVD Status')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA":
    st.markdown('<div class="section-title">Target Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['cardio'].value_counts()
        ax.pie(counts, labels=['No CVD', 'CVD'], autopct='%1.1f%%',
               colors=['#27ae60', '#e74c3c'], startangle=90)
        ax.set_title('CVD Distribution')
        st.pyplot(fig); plt.close()

    with col2:
        pct = df['cardio'].value_counts(normalize=True)*100
        st.metric("No CVD", f"{pct[0]:.1f}%")
        st.metric("CVD", f"{pct[1]:.1f}%")
        st.info("The dataset is well-balanced, ideal for classification tasks.")

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    num_df = df.select_dtypes(include=np.number)
    sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Lifestyle Factors vs CVD</div>', unsafe_allow_html=True)
    cat_features = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']
    fig, axes = plt.subplots(1, len(cat_features), figsize=(18, 4))
    for ax, feat in zip(axes, cat_features):
        sns.countplot(x=feat, hue='cardio', data=df, ax=ax,
                      palette=['#27ae60', '#e74c3c'])
        ax.set_title(feat.capitalize())
        ax.legend(title='CVD', labels=['No', 'Yes'])
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
**Key EDA Findings:**
1. Older patients show significantly higher CVD prevalence.
2. Elevated systolic blood pressure is strongly associated with CVD.
3. Higher BMI correlates with increased cardiovascular risk.
4. Cholesterol level is a powerful predictor.
5. Physical activity appears protective.
""")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)

    sorted_df = results_df.sort_values('ROC AUC', ascending=False).reset_index(drop=True)
    st.dataframe(sorted_df.style.highlight_max(
        subset=['Accuracy','Precision','Recall','F1 Score','ROC AUC'],
        color='#d4edda'), use_container_width=True)

    st.markdown('<div class="section-title">ROC AUC Comparison</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#e74c3c' if m == 'CatBoost' else '#3498db' for m in sorted_df['Model']]
    bars = ax.barh(sorted_df['Model'], sorted_df['ROC AUC'], color=colors)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel('ROC AUC Score')
    ax.set_title('Model Comparison — ROC AUC (CatBoost = 🏆 Best)')
    for bar, val in zip(bars, sorted_df['ROC AUC']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Confusion Matrix — CatBoost</div>', unsafe_allow_html=True)
    preds  = best_model.predict(X_test)
    probs  = best_model.predict_proba(X_test)[:,1]
    cm = confusion_matrix(y_test, preds)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                    xticklabels=['No CVD','CVD'], yticklabels=['No CVD','CVD'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig); plt.close()

    with col2:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'CatBoost AUC={auc_val:.4f}')
        ax.plot([0,1],[0,1],'k--', lw=1)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve'); ax.legend()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Feature Importance — CatBoost</div>', unsafe_allow_html=True)
    drop_cols = [c for c in ['id','age'] if c in df.columns]
    X_all = df.drop(['cardio'] + drop_cols, axis=1)
    imp_df = pd.DataFrame({
        'Feature': X_all.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp_df['Feature'], imp_df['Importance'], color='#e74c3c')
    ax.set_title('Feature Importances (CatBoost)')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Risk":
    st.markdown('<div class="section-title">Patient Risk Assessment</div>', unsafe_allow_html=True)
    st.info("Enter patient details below to predict cardiovascular disease risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Biometrics**")
        age_years = st.slider("Age (years)", 18, 90, 45)
        gender    = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        height    = st.slider("Height (cm)", 120, 220, 170)
        weight    = st.slider("Weight (kg)", 30, 200, 75)
        bmi       = round(weight / ((height / 100) ** 2), 2)
        st.metric("Calculated BMI", bmi,
                  delta="Normal" if 18.5 <= bmi <= 24.9 else ("Overweight" if bmi < 30 else "Obese"))

    with col2:
        st.markdown("**Blood Pressure & Lab**")
        ap_hi      = st.slider("Systolic BP (ap_hi)", 80, 250, 120)
        ap_lo      = st.slider("Diastolic BP (ap_lo)", 40, 160, 80)
        cholesterol= st.selectbox("Cholesterol", [1, 2, 3],
                                   format_func=lambda x: {1:"Normal",2:"Above Normal",3:"Well Above Normal"}[x])
        gluc       = st.selectbox("Glucose", [1, 2, 3],
                                   format_func=lambda x: {1:"Normal",2:"Above Normal",3:"Well Above Normal"}[x])

    with col3:
        st.markdown("**Lifestyle**")
        smoke  = st.radio("Smoking?",   [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        alco   = st.radio("Alcohol?",   [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        active = st.radio("Physically active?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    if ap_hi <= ap_lo:
        st.warning("⚠️ Systolic BP should be greater than Diastolic BP.")

    st.divider()

    if st.button("🔍 Predict Cardiovascular Risk", type="primary", use_container_width=True):
        input_dict = {
            'age_years': age_years, 'gender': gender, 'height': height,
            'weight': weight, 'ap_hi': ap_hi, 'ap_lo': ap_lo,
            'cholesterol': cholesterol, 'gluc': gluc, 'smoke': smoke,
            'alco': alco, 'active': active, 'BMI': bmi
        }
        # Align to feature order used in training
        input_df = pd.DataFrame([input_dict])[feature_cols]

        proba = best_model.predict_proba(input_df)[0][1]
        pred  = best_model.predict(input_df)[0]

        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f'<div class="risk-high">⚠️ HIGH RISK of Cardiovascular Disease — Probability: {proba*100:.1f}%</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ LOW RISK of Cardiovascular Disease — Probability: {proba*100:.1f}%</div>',
                        unsafe_allow_html=True)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(6, 3))
        bar_color = '#e74c3c' if proba >= 0.5 else '#27ae60'
        ax.barh(['Risk'], [proba], color=bar_color, height=0.4)
        ax.barh(['Risk'], [1 - proba], left=[proba], color='#ecf0f1', height=0.4)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Predicted Probability')
        ax.set_title(f'CVD Risk Score: {proba*100:.1f}%')
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("**⚕️ Disclaimer:** This tool is for educational purposes only and does not replace professional medical advice.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("""
**CSS 324 · Final Project**  
Cardiovascular Disease Prediction  
Model: CatBoost Classifier  
Dataset: 70,000 patient records
""")