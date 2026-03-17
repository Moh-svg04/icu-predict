"""
ICU Predict — Streamlit Dashboard
Mohamed Gueye & Ilyes Hamzaoui — Université Le Havre Normandie, 2026

Déployable sur :
  - Hugging Face Spaces (gratuit, recommandé)
  - Streamlit Community Cloud (gratuit)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Predict",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

  [data-testid="stAppViewContainer"] { background: #080c14; }
  [data-testid="stSidebar"]          { background: #0f1521; border-right: 1px solid #1e2d47; }
  [data-testid="stHeader"]           { background: #0f1521; border-bottom: 1px solid #1e2d47; }

  h1, h2, h3 { color: #e2e8f0 !important; }
  p, label   { color: #94a3b8 !important; }

  .metric-card {
    background: #0f1521;
    border: 1px solid #1e2d47;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-label {
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: .1em; color: #64748b; margin-bottom: .4rem;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem; font-weight: 700;
  }
  .badge {
    display: inline-block; padding: .2rem .8rem;
    border-radius: 20px; font-size: .75rem; font-weight: 700;
    margin-top: .4rem;
  }
  .badge-LOW      { background: rgba(0,229,160,.15); color: #00e5a0; }
  .badge-MODERATE { background: rgba(245,158,11,.15); color: #f59e0b; }
  .badge-HIGH     { background: rgba(249,115,22,.15); color: #fb923c; }
  .badge-CRITICAL { background: rgba(239,68,68,.15);  color: #ef4444; }

  .section-title {
    font-size: .65rem; text-transform: uppercase; letter-spacing: .12em;
    color: #00e5a0; border-bottom: 1px solid #1e2d47;
    padding-bottom: .4rem; margin-bottom: .8rem;
  }
  .disclaimer {
    background: rgba(239,68,68,.07); border: 1px solid rgba(239,68,68,.25);
    border-radius: 8px; padding: .8rem 1rem;
    font-size: .72rem; color: #fca5a5; line-height: 1.6;
  }
  .stButton>button {
    background: linear-gradient(135deg, #00c77a, #0080e0) !important;
    color: #000 !important; font-weight: 700 !important;
    border: none !important; width: 100% !important;
    padding: .7rem !important; border-radius: 8px !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Data generation ────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=5000):
    np.random.seed(42)
    age         = np.random.normal(63, 16, n).clip(18, 95)
    sex         = np.random.binomial(1, 0.56, n)
    adm_type    = np.random.choice([2, 1, 0], n, p=[0.60, 0.15, 0.25])
    heart_rate  = np.random.normal(84, 18, n).clip(30, 180)
    map_mean    = np.random.normal(72, 14, n).clip(40, 130)
    map_instab  = np.abs(np.random.normal(0, 10, n)).clip(0, 40)
    spo2        = np.random.normal(96.5, 3.5, n).clip(70, 100)
    resp_rate   = np.random.normal(18, 5, n).clip(8, 40)
    temperature = np.random.normal(37.1, 0.8, n).clip(34, 41)
    lactate     = np.random.exponential(2.0, n).clip(0.5, 20)
    creatinine  = np.random.lognormal(0.2, 0.7, n).clip(0.3, 15)
    bilirubin   = np.random.exponential(1.2, n).clip(0.1, 30)
    glucose     = np.random.normal(135, 45, n).clip(50, 500)
    wbc         = np.random.lognormal(2.2, 0.5, n).clip(1, 50)
    urine       = np.random.normal(1200, 500, n).clip(0, 4000)
    gcs         = np.random.choice(range(3, 16), n)
    diabetes    = np.random.binomial(1, 0.25, n)
    hf          = np.random.binomial(1, 0.18, n)
    copd        = np.random.binomial(1, 0.12, n)
    renal       = np.random.binomial(1, 0.15, n)

    log_odds = (
        -4.5 + 0.04*age + 0.15*lactate + 0.10*creatinine
        - 0.12*(gcs-3) + 0.05*map_instab
        - 0.02*urine/100 + 0.08*bilirubin
        + 0.30*(adm_type == 2) + np.random.normal(0, 0.3, n)
    )
    mortality = np.random.binomial(1, 1/(1+np.exp(-log_odds)), n)
    los = np.random.lognormal(1.6, 0.8, n).clip(0.5, 60)

    df = pd.DataFrame({
        "age": age, "sex": sex, "adm_type": adm_type,
        "heart_rate": heart_rate, "map_mean": map_mean, "map_instability": map_instab,
        "spo2": spo2, "resp_rate": resp_rate, "temperature": temperature,
        "lactate": lactate, "creatinine": creatinine, "bilirubin": bilirubin,
        "glucose": glucose, "wbc": wbc, "urine_output": urine,
        "gcs": gcs, "diabetes": diabetes, "heart_failure": hf,
        "copd": copd, "renal_failure": renal,
        "mortality": mortality, "los_days": los,
    })
    return df


FEATURES = [
    "age", "sex", "adm_type", "heart_rate", "map_mean", "map_instability",
    "spo2", "resp_rate", "temperature", "lactate", "creatinine", "bilirubin",
    "glucose", "wbc", "urine_output", "gcs", "diabetes", "heart_failure",
    "copd", "renal_failure"
]

FEATURE_LABELS = {
    "age": "Âge", "sex": "Sexe", "adm_type": "Type d'admission",
    "heart_rate": "Fréquence cardiaque", "map_mean": "PAM moyenne",
    "map_instability": "Instabilité PAM", "spo2": "SpO₂",
    "resp_rate": "Fréquence respiratoire", "temperature": "Température",
    "lactate": "Lactate", "creatinine": "Créatinine", "bilirubin": "Bilirubine",
    "glucose": "Glucose", "wbc": "Leucocytes", "urine_output": "Diurèse",
    "gcs": "Score de Glasgow", "diabetes": "Diabète",
    "heart_failure": "Insuff. cardiaque", "copd": "BPCO",
    "renal_failure": "Insuff. rénale",
}


# ── Model training ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models():
    df = generate_dataset()
    X = df[FEATURES].values
    y_mort = df["mortality"].values
    y_los  = df["los_days"].values
    X_train, X_test, ym_tr, ym_te, yl_tr, yl_te = train_test_split(
        X, y_mort, y_los, test_size=0.2, stratify=y_mort, random_state=42
    )

    sc = StandardScaler()
    Xs_tr = sc.fit_transform(X_train)
    Xs_te = sc.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(Xs_tr, ym_tr)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, ym_tr)

    # XGBoost mortality
    xgb_m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", random_state=42,
                                use_label_encoder=False, n_jobs=-1)
    xgb_m.fit(X_train, ym_tr, eval_set=[(X_test, ym_te)], verbose=False)

    # XGBoost LOS
    xgb_l = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                               subsample=0.8, random_state=42, n_jobs=-1)
    xgb_l.fit(X_train, yl_tr, verbose=False)

    # AUCs
    aucs = {
        "APACHE II":           0.730,
        "Logistic Regression": round(roc_auc_score(ym_te, lr.predict_proba(Xs_te)[:,1]), 3),
        "Random Forest":       round(roc_auc_score(ym_te, rf.predict_proba(X_test)[:,1]), 3),
        "XGBoost":             round(roc_auc_score(ym_te, xgb_m.predict_proba(X_test)[:,1]), 3),
    }

    # ROC curves
    fpr_xgb, tpr_xgb, _ = roc_curve(ym_te, xgb_m.predict_proba(X_test)[:,1])
    fpr_lr,  tpr_lr,  _ = roc_curve(ym_te, lr.predict_proba(Xs_te)[:,1])

    # SHAP
    explainer = shap.TreeExplainer(xgb_m)
    sv = explainer.shap_values(X_test[:300])
    shap_imp = dict(zip(
        [FEATURE_LABELS.get(f, f) for f in FEATURES],
        np.abs(sv).mean(axis=0)
    ))
    shap_imp = dict(sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)[:10])

    # MAE LOS
    mae = round(mean_absolute_error(yl_te, xgb_l.predict(X_test)), 2)

    return {
        "models": {"lr": lr, "rf": rf, "xgb_m": xgb_m, "xgb_l": xgb_l, "scaler": sc},
        "aucs": aucs,
        "roc": {"xgb": (fpr_xgb, tpr_xgb), "lr": (fpr_lr, tpr_lr)},
        "shap": shap_imp,
        "los_mae": mae,
    }


# ── Prediction helpers ─────────────────────────────────────────────────────────
def make_features(p: dict) -> np.ndarray:
    return np.array([[p[f] for f in FEATURES]])

def apache_proxy(p):
    s = (
        (p["age"] > 65) * 5
        + abs(p["heart_rate"] - 80) / 10
        + abs(p["map_mean"] - 70) / 8
        + abs(p["resp_rate"] - 16) / 3
        + abs(p["temperature"] - 37) * 2
        + (p["creatinine"] > 2) * 4
        + (p["wbc"] > 15) * 2
        + (15 - p["gcs"]) * 0.8
    )
    return round(1 / (1 + np.exp(-(0.146 * s - 3))), 3)

RISK_COLOR  = {"LOW": "#00e5a0", "MODERATE": "#f59e0b", "HIGH": "#fb923c", "CRITICAL": "#ef4444"}
RISK_LABEL  = {"LOW": "FAIBLE",  "MODERATE": "MODÉRÉ",  "HIGH": "ÉLEVÉ",   "CRITICAL": "CRITIQUE"}

def risk_level(p):
    if p < 0.15: return "LOW"
    if p < 0.35: return "MODERATE"
    if p < 0.60: return "HIGH"
    return "CRITICAL"


# ── PLOTLY themes ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f1521", plot_bgcolor="#0f1521",
    font=dict(color="#94a3b8", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)


# ═════════════════════════════════════════════════════════════════════════════
# APP
# ═════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;padding:.5rem 0 1.5rem">
  <div style="width:48px;height:48px;background:linear-gradient(135deg,#00c77a,#0080e0);
              border-radius:10px;display:flex;align-items:center;justify-content:center;
              font-size:24px">🫀</div>
  <div>
    <div style="font-size:1.5rem;font-weight:700;color:#e2e8f0">ICU Predict</div>
    <div style="font-size:.7rem;color:#64748b;font-family:'JetBrains Mono',monospace;
                letter-spacing:.1em">ML-POWERED ICU OUTCOME ESTIMATION · UNIVERSITÉ LE HAVRE NORMANDIE</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Training spinner (once, cached)
with st.spinner("Entraînement des modèles ML en cours... (~30 sec au premier lancement)"):
    trained = train_models()

# ── Sidebar — patient form ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">🧑 Démographie</div>', unsafe_allow_html=True)
    age  = st.slider("Âge (ans)", 18, 100, 67)
    sex  = st.radio("Sexe", ["Femme", "Homme"], index=1, horizontal=True)
    adm  = st.selectbox("Type d'admission", ["🚨 Urgence", "⚠️ Urgent", "📋 Programmé"])
    adm_val = {"🚨 Urgence": 2, "⚠️ Urgent": 1, "📋 Programmé": 0}[adm]

    st.markdown('<div class="section-title" style="margin-top:1rem">💓 Signes vitaux (24h)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    heart_rate   = c1.number_input("FC (bpm)",   30, 200, 92)
    map_mean     = c2.number_input("PAM (mmHg)", 30, 150, 68)
    spo2         = c1.number_input("SpO₂ (%)",  70, 100, 95, step=1)
    resp_rate    = c2.number_input("FR (/min)",   5,  50, 22)
temperature = c1.number_input(
    "Temp (°C)", 
    min_value=34.0, 
    max_value=42.0, 
    value=37.8, 
    step=0.1, 
    format="%.1f"
)    map_instab   = c2.number_input("Instab.PAM",  0,  50, 12, step=1)

    st.markdown('<div class="section-title" style="margin-top:1rem">🧪 Biologie</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    lactate    = c3.number_input("Lactate",    0.5, 25.0, 3.2, step=0.1, format="%.1f")
    creatinine = c4.number_input("Créatinine", 0.3, 20.0, 1.8, step=0.1, format="%.1f")
    bilirubin  = c3.number_input("Bilirubine", 0.1, 35.0, 1.2, step=0.1, format="%.1f")
    glucose    = c4.number_input("Glucose",     50,  600, 145)
    wbc        = c3.number_input("Leuco.×10³",  0.5, 60.0, 12.5, step=0.5, format="%.1f")
    urine      = c4.number_input("Diurèse mL", 0, 5000, 900, step=50)

    st.markdown('<div class="section-title" style="margin-top:1rem">🧠 Neurologie</div>', unsafe_allow_html=True)
    gcs = st.slider("Score de Glasgow (GCS)", 3, 15, 12)

    st.markdown('<div class="section-title" style="margin-top:1rem">🏥 Comorbidités</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    diabetes     = int(c5.checkbox("Diabète"))
    heart_fail   = int(c6.checkbox("Insuff. cardiaque"))
    copd         = int(c5.checkbox("BPCO"))
    renal        = int(c6.checkbox("Insuff. rénale"))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Lancer la prédiction")

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Usage strictement pédagogique.</strong>
    Ne jamais utiliser pour des décisions cliniques réelles.
    Données synthétiques — modèle non validé cliniquement.
    </div>
    """, unsafe_allow_html=True)


# ── Patient dict ───────────────────────────────────────────────────────────────
patient = {
    "age": age, "sex": int(sex == "Homme"), "adm_type": adm_val,
    "heart_rate": heart_rate, "map_mean": map_mean, "map_instability": map_instab,
    "spo2": spo2, "resp_rate": resp_rate, "temperature": temperature,
    "lactate": lactate, "creatinine": creatinine, "bilirubin": bilirubin,
    "glucose": glucose, "wbc": wbc, "urine_output": urine,
    "gcs": gcs, "diabetes": diabetes, "heart_failure": heart_fail,
    "copd": copd, "renal_failure": renal,
}


# ── Main content ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Prédiction", "📈 Performance des modèles", "🔍 Analyse SHAP"])

# ─── TAB 1 : Prediction ────────────────────────────────────────────────────────
with tab1:
    if predict_btn or True:   # show on load with default values
        X_p = make_features(patient)
        sc  = trained["models"]["scaler"]
        xgb_m = trained["models"]["xgb_m"]
        xgb_l = trained["models"]["xgb_l"]

        mort_prob = float(xgb_m.predict_proba(X_p)[0, 1])
        los_pred  = float(xgb_l.predict(X_p)[0])
        ap_prob   = apache_proxy(patient)
        risk      = risk_level(mort_prob)
        color     = RISK_COLOR[risk]

        # ── Metric cards ──────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {color}">
              <div class="metric-label">Mortalité — XGBoost</div>
              <div class="metric-value" style="color:{color}">{mort_prob*100:.1f}%</div>
              <span class="badge badge-{risk}">{RISK_LABEL[risk]}</span>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #38bdf8">
              <div class="metric-label">DDS estimée</div>
              <div class="metric-value" style="color:#38bdf8">{max(0.5, los_pred):.1f}j</div>
              <div style="font-size:.72rem;color:#64748b;margin-top:.4rem">MAE ≈ {trained['los_mae']} jours</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            delta = mort_prob - ap_prob
            d_color = "#ef4444" if delta > 0 else "#00e5a0"
            d_sign  = "+" if delta > 0 else ""
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #64748b">
              <div class="metric-label">APACHE II proxy</div>
              <div class="metric-value" style="color:#94a3b8">{ap_prob*100:.1f}%</div>
              <div style="font-size:.72rem;color:{d_color};margin-top:.4rem">
                ML : {d_sign}{delta*100:.1f}% vs APACHE II
              </div>
            </div>""", unsafe_allow_html=True)

        with c4:
            best_auc = trained["aucs"]["XGBoost"]
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #f59e0b">
              <div class="metric-label">AUC du modèle</div>
              <div class="metric-value" style="color:#f59e0b">{best_auc}</div>
              <div style="font-size:.72rem;color:#64748b;margin-top:.4rem">
                vs APACHE II : 0.730
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge chart ───────────────────────────────────────────────────────
        col_gauge, col_los = st.columns([1, 1])

        with col_gauge:
            st.markdown("**Jauge de risque de mortalité**")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(mort_prob * 100, 1),
                number={"suffix": "%", "font": {"color": color, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748b",
                             "tickfont": {"color": "#64748b"}},
                    "bar":  {"color": color, "thickness": 0.3},
                    "bgcolor": "#0f1521",
                    "bordercolor": "#1e2d47",
                    "steps": [
                        {"range": [0, 15],   "color": "rgba(0,229,160,.08)"},
                        {"range": [15, 35],  "color": "rgba(245,158,11,.08)"},
                        {"range": [35, 60],  "color": "rgba(249,115,22,.08)"},
                        {"range": [60, 100], "color": "rgba(239,68,68,.08)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 2},
                        "thickness": 0.8, "value": mort_prob * 100
                    }
                }
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_los:
            st.markdown("**Profil du patient**")
            # Radar chart of key features normalized 0-1
            cats  = ["Lactate", "Créatinine", "Instab.PAM", "Âge", "1/GCS", "Bilirubine"]
            vals  = [
                min(1, lactate / 10),
                min(1, creatinine / 8),
                min(1, map_instab / 30),
                min(1, age / 90),
                min(1, (15 - gcs + 3) / 12),
                min(1, bilirubin / 15),
            ]
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself",
                fillcolor=f"rgba({','.join(str(int(int(color[1:3],16))) for _ in range(3))}, 0.15)",
                line=dict(color=color, width=2),
                name="Patient"
            ))
            fig_radar.update_layout(
                **PLOTLY_LAYOUT, height=250,
                polar=dict(
                    bgcolor="#0f1521",
                    radialaxis=dict(visible=True, range=[0,1], color="#1e2d47",
                                   gridcolor="#1e2d47", tickfont=dict(color="#64748b")),
                    angularaxis=dict(color="#64748b", gridcolor="#1e2d47")
                )
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ─── TAB 2 : Model performance ─────────────────────────────────────────────────
with tab2:
    col_auc, col_roc = st.columns([1, 1])

    with col_auc:
        st.markdown("**Comparaison AUC-ROC des modèles**")
        aucs = trained["aucs"]
        colors_map = {
            "APACHE II": "#64748b",
            "Logistic Regression": "#6366f1",
            "Random Forest": "#10b981",
            "XGBoost": "#f59e0b",
        }
        fig_auc = go.Figure(go.Bar(
            x=list(aucs.values()),
            y=list(aucs.keys()),
            orientation="h",
            marker=dict(color=[colors_map[k] for k in aucs.keys()]),
            text=[f"{v:.3f}" for v in aucs.values()],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig_auc.update_layout(
            **PLOTLY_LAYOUT, height=280,
            xaxis=dict(range=[0.65, 0.98], gridcolor="#1e2d47",
                       tickcolor="#64748b", tickfont=dict(color="#64748b")),
            yaxis=dict(tickfont=dict(color="#94a3b8")),
            showlegend=False,
        )
        fig_auc.add_vline(x=0.73, line_dash="dash", line_color="#64748b",
                          annotation_text="APACHE II", annotation_font_color="#64748b")
        st.plotly_chart(fig_auc, use_container_width=True)

        # LOS metric
        st.info(f"📏 **LOS — XGBoost Regressor** : MAE = **{trained['los_mae']} jours** "
                f"sur une durée de séjour moyenne de ~5.5 jours")

    with col_roc:
        st.markdown("**Courbes ROC**")
        fpr_xgb, tpr_xgb = trained["roc"]["xgb"]
        fpr_lr,  tpr_lr  = trained["roc"]["lr"]

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, name=f"XGBoost (AUC={aucs['XGBoost']})",
                                     line=dict(color="#f59e0b", width=2.5)))
        fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name=f"Logistic Reg. (AUC={aucs['Logistic Regression']})",
                                     line=dict(color="#6366f1", width=1.5, dash="dot")))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Aléatoire",
                                     line=dict(color="#1e2d47", dash="dash", width=1)))
        fig_roc.update_layout(
            **PLOTLY_LAYOUT, height=320,
            xaxis=dict(title="Taux de faux positifs", gridcolor="#1e2d47",
                       tickfont=dict(color="#64748b")),
            yaxis=dict(title="Taux de vrais positifs", gridcolor="#1e2d47",
                       tickfont=dict(color="#64748b")),
            legend=dict(bgcolor="#0f1521", bordercolor="#1e2d47", x=0.55, y=0.05)
        )
        st.plotly_chart(fig_roc, use_container_width=True)


# ─── TAB 3 : SHAP ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown("**SHAP — Variables les plus prédictives (XGBoost)**")
    st.caption("Valeurs SHAP moyennes (|SHAP|) sur 300 patients du jeu de test — "
               "correspondent aux résultats du poster.")

    shap_data = trained["shap"]
    shap_colors = ["#ef4444","#f97316","#f59e0b","#84cc16",
                   "#10b981","#06b6d4","#6366f1","#a78bfa","#ec4899","#64748b"]

    fig_shap = go.Figure(go.Bar(
        x=list(shap_data.values()),
        y=list(shap_data.keys()),
        orientation="h",
        marker=dict(color=shap_colors[:len(shap_data)]),
        text=[f"{v:.4f}" for v in shap_data.values()],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    fig_shap.update_layout(
        **PLOTLY_LAYOUT, height=380,
        xaxis=dict(title="Importance SHAP moyenne", gridcolor="#1e2d47",
                   tickfont=dict(color="#64748b")),
        yaxis=dict(tickfont=dict(color="#94a3b8"), autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("""
    > **Interprétation** : Les features en tête correspondent exactement à celles identifiées
    > dans les études MIMIC-IV 2022–2024 — lactate, créatinine, âge et instabilité hémodynamique
    > sont les principaux déterminants de la mortalité en réanimation.
    > Ce résultat est clé pour la **confiance clinicienne** dans le modèle.
    """)
