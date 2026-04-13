import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Scoring — ML Project",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    .stApp { background: #0f1117; }

    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 50%, #1a1f2e 100%);
        border: 1px solid #2a2f3e;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(99,179,237,0.06) 0%, transparent 60%),
                    radial-gradient(circle at 80% 20%, rgba(154,117,234,0.06) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .hero p { color: #718096; font-size: 1rem; margin: 0; }
    .hero .accent { color: #63b3ed; }

    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #63b3ed;
    }
    .metric-card .lbl { font-size: 0.78rem; color: #718096; margin-top: 0.2rem; letter-spacing: 0.05em; text-transform: uppercase; }

    .result-default {
        background: linear-gradient(135deg, #2d1515, #1a0f0f);
        border: 1px solid #e53e3e;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        color: #fc8181;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .result-safe {
        background: linear-gradient(135deg, #0f2d1a, #0a1f12);
        border: 1px solid #38a169;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        color: #68d391;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a2f3e;
    }
    div[data-testid="stSlider"] label { color: #a0aec0 !important; font-size: 0.85rem; }
    div[data-testid="stNumberInput"] label { color: #a0aec0 !important; font-size: 0.85rem; }
    .stSelectbox label { color: #a0aec0 !important; font-size: 0.85rem; }
    .stButton > button {
        background: linear-gradient(135deg, #3182ce, #2b6cb0);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #4299e1, #3182ce); transform: translateY(-1px); }
    .sidebar-info {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.82rem;
        color: #718096;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# ── Feature names ────────────────────────────────────────────────
FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]

# ── Hero header ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💳 Credit <span class="accent">Scoring</span> Dashboard</h1>
    <p>ML-powered default risk prediction · GiveMeSomeCredit dataset · XGBoost Champion Model</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Client Profile")
    st.markdown("---")

    st.markdown('<p class="section-title">💰 Financial</p>', unsafe_allow_html=True)
    monthly_income  = st.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=5400, step=100)
    debt_ratio      = st.slider("Debt Ratio", 0.0, 1.0, 0.35, 0.01)
    revolving_util  = st.slider("Revolving Utilization", 0.0, 1.0, 0.30, 0.01)

    st.markdown('<p class="section-title">👤 Personal</p>', unsafe_allow_html=True)
    age             = st.slider("Age", 18, 100, 45)
    dependents      = st.number_input("Number of Dependents", 0, 20, 0)

    st.markdown('<p class="section-title">🏦 Credit History</p>', unsafe_allow_html=True)
    open_credit     = st.number_input("Open Credit Lines & Loans", 0, 50, 8)
    real_estate     = st.number_input("Real Estate Loans or Lines", 0, 20, 1)
    late_30_59      = st.number_input("Times 30-59 Days Late", 0, 20, 0)
    late_60_89      = st.number_input("Times 60-89 Days Late", 0, 20, 0)
    late_90         = st.number_input("Times 90+ Days Late", 0, 20, 0)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Risk")

    st.markdown("""
    <div class="sidebar-info">
        <b>Model :</b> XGBoost (Optimized)<br>
        <b>Dataset :</b> GiveMeSomeCredit<br>
        <b>Metric :</b> AUC-ROC<br>
        <b>Threshold :</b> 0.50
    </div>
    """, unsafe_allow_html=True)

# ── Build input vector ───────────────────────────────────────────
input_data = pd.DataFrame([[
    revolving_util, age, late_30_59, debt_ratio, monthly_income,
    open_credit, late_90, real_estate, late_60_89, dependents
]], columns=FEATURES)

# ── Main content ─────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Prediction", "🔬 SHAP Explanation"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        # Scale only if model needs it (LR/KNN), XGBoost doesn't
        model_name = type(model).__name__
        if model_name in ['LogisticRegression', 'KNeighborsClassifier']:
            X_input = scaler.transform(input_data)
        else:
            X_input = input_data.values

        proba    = model.predict_proba(X_input)[0][1]
        pred     = int(proba >= 0.5)
        risk_pct = proba * 100

        # ── Result banner ────────────────────────────────────────
        col_res, col_gauge = st.columns([1.2, 1])
        with col_res:
            if pred == 1:
                st.markdown(f"""
                <div class="result-default">
                    ⚠️ HIGH RISK — Default Likely<br>
                    <span style="font-size:0.85rem; font-weight:400; color:#fc8181cc;">
                    Probability of default: {risk_pct:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    ✅ LOW RISK — Good Payer<br>
                    <span style="font-size:0.85rem; font-weight:400; color:#68d391cc;">
                    Probability of default: {risk_pct:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)

        with col_gauge:
            # Gauge chart
            fig, ax = plt.subplots(figsize=(4, 2.2), facecolor='none')
            color = '#e53e3e' if pred == 1 else '#38a169'
            ax.barh([0], [1], color='#2a2f3e', height=0.4, zorder=1)
            ax.barh([0], [proba], color=color, height=0.4, zorder=2)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='#718096', fontsize=9)
            ax.axvline(0.5, color='#718096', linestyle='--', lw=1, zorder=3)
            ax.set_facecolor('none')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"Risk Score: {risk_pct:.1f}%", color='#e2e8f0', fontsize=11, pad=8)
            plt.tight_layout()
            st.pyplot(fig, transparent=True)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metric cards ─────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (f"{risk_pct:.1f}%", "Default Probability"),
            ("High" if pred == 1 else "Low", "Risk Level"),
            (f"${monthly_income:,}", "Monthly Income"),
            (f"{age} yrs", "Client Age"),
        ]
        for col, (val, lbl) in zip([c1, c2, c3, c4], cards):
            col.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Feature values summary ────────────────────────────────
        st.markdown('<p class="section-title">📋 Input Summary</p>', unsafe_allow_html=True)
        summary_df = input_data.T.rename(columns={0: 'Value'})
        summary_df['Value'] = summary_df['Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
        st.dataframe(summary_df, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color:#4a5568;">
            <div style="font-size:3rem;">💳</div>
            <div style="font-size:1.1rem; margin-top:1rem;">Fill in the client profile on the left,<br>then click <b style="color:#63b3ed">Predict Risk</b></div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — SHAP Explanation
# ════════════════════════════════════════════════════════════════
with tab2:
    if predict_btn:
        st.markdown('<p class="section-title">🔬 SHAP Feature Contribution</p>', unsafe_allow_html=True)

        with st.spinner("Computing SHAP values..."):
            try:
                model_type = type(model).__name__

                if model_type in ['XGBClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']:
                    explainer   = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_data)
                    if isinstance(shap_values, list):
                        sv = shap_values[1][0]
                    else:
                        sv = shap_values[0]

                elif model_type == 'LogisticRegression':
                    # LinearExplainer needs background data — we use zeros as reference
                    background  = np.zeros((1, len(FEATURES)))
                    explainer   = shap.LinearExplainer(model, background, feature_perturbation="interventional")
                    shap_values = explainer.shap_values(scaler.transform(input_data))
                    sv = shap_values[0]

                else:
                    # KNN or unknown → use KernelExplainer (slower but universal)
                    background  = np.zeros((1, len(FEATURES)))
                    explainer   = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(input_data.values, nsamples=100)
                    sv = shap_values[1][0]

                # ── SHAP bar chart ───────────────────────────────
                shap_df = pd.DataFrame({
                    'Feature': FEATURES,
                    'SHAP':    sv
                }).sort_values('SHAP', key=abs, ascending=True)

                fig, ax = plt.subplots(figsize=(9, 5), facecolor='none')
                colors_shap = ['#fc8181' if v > 0 else '#68d391' for v in shap_df['SHAP']]
                bars = ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors_shap, edgecolor='none', height=0.6)

                ax.axvline(0, color='#718096', lw=1)
                ax.set_xlabel('SHAP Value  (+ increases risk, − decreases risk)', color='#a0aec0', fontsize=9)
                ax.set_title('Feature Contributions to Prediction', color='#e2e8f0', fontsize=12, fontweight='bold', pad=12)
                ax.tick_params(colors='#a0aec0', labelsize=9)
                ax.set_facecolor('none')
                for spine in ax.spines.values():
                    spine.set_color('#2a2f3e')

                red_patch   = mpatches.Patch(color='#fc8181', label='Increases default risk')
                green_patch = mpatches.Patch(color='#68d391', label='Decreases default risk')
                ax.legend(handles=[red_patch, green_patch], loc='lower right',
                          facecolor='#1a1f2e', edgecolor='#2a2f3e',
                          labelcolor='#a0aec0', fontsize=8)

                plt.tight_layout()
                st.pyplot(fig, transparent=True)
                plt.close()

                # ── Top factors ───────────────────────────────────
                st.markdown('<p class="section-title">🔝 Top Risk Factors</p>', unsafe_allow_html=True)
                top3 = shap_df.reindex(shap_df['SHAP'].abs().sort_values(ascending=False).index).head(3)
                for _, row in top3.iterrows():
                    direction = "🔴 Increases" if row['SHAP'] > 0 else "🟢 Decreases"
                    val = input_data[row['Feature']].values[0]
                    st.markdown(f"""
                    <div style="background:#1a1f2e; border:1px solid #2a2f3e; border-radius:8px;
                                padding:0.8rem 1.2rem; margin-bottom:0.5rem; font-size:0.9rem; color:#a0aec0;">
                        {direction} risk &nbsp;·&nbsp;
                        <b style="color:#e2e8f0">{row['Feature']}</b> = {val:.3f}
                        &nbsp;·&nbsp; SHAP = <b style="color:{'#fc8181' if row['SHAP']>0 else '#68d391'}">{row['SHAP']:+.4f}</b>
                    </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
                
    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color:#4a5568;">
            <div style="font-size:3rem;">🔬</div>
            <div style="font-size:1.1rem; margin-top:1rem;">Run a prediction first to see<br>the <b style="color:#63b3ed">SHAP explanation</b></div>
        </div>
        """, unsafe_allow_html=True)
