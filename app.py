
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import textwrap
import streamlit.components.v1 as components

# ==========================================
# CARDIOGUARD AI - PREMIUM WEB APPLICATION
# ==========================================

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "best_model.pkl"
REPORT = BASE / "reports" / "model_results"
SHAP_DIR = BASE / "reports" / "shap"

st.set_page_config(
    page_title="CardioGuard AI | Heart Intelligence",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        return exc

model = load_model(str(MODEL_PATH))
model_ready = model is not None and not isinstance(model, Exception)

# ==========================================
# PREMIUM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #07111F;
    --panel: #101F32;
    --panel2: #14283D;
    --cyan: #37D6E8;
    --teal: #16C7B7;
    --muted: #9BB0C5;
    --white: #F4FAFF;
}

.stApp {
    background: radial-gradient(ellipse at 15% 0%, #12324A 0%, #07111F 48%, #07111F 100%);
    color: var(--white);
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--white) !important;
}

p, li, label {
    color: #C1D0DF;
}

[data-testid="stSidebar"] {display: none;}
[data-testid="collapsedControl"] {display: none;}
.top-brand {font-family:'Outfit',sans-serif;font-size:30px;font-weight:800;color:#4BE0E5;}
.top-caption {font-size:11px;letter-spacing:2px;color:#9BB0C5;margin-top:-4px;}
div[role="radiogroup"] {gap:8px; flex-wrap:wrap;}
div[role="radiogroup"] label {background:#102438;border:1px solid #294B60;padding:9px 15px;border-radius:999px;}
div[role="radiogroup"] label:hover {border-color:#37D6E8;}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    background: linear-gradient(115deg, rgba(12,35,55,.97), rgba(9,67,79,.86)),
    url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1500&q=85');
    background-size: cover;
    background-position: center;
    border: 1px solid #28566C;
    border-radius: 25px;
    padding: 48px 42px;
    min-height: 330px;
    box-shadow: 0 15px 50px rgba(0,0,0,.25);
    animation: appear .8s ease;
}

.hero-tag {
    color: #53E5E5;
    font-weight: 700;
    letter-spacing: 2px;
    font-size: 12px;
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(35px, 5vw, 58px);
    font-weight: 800;
    line-height: 1.1;
    color: white;
    margin-top: 15px;
}

.hero-text {
    font-size: 17px;
    color: #D0E3F1;
    max-width: 600px;
    line-height: 1.8;
}

.section-label {
    color: #43DCE1;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: 800;
}

.section-heading {
    font-size: 30px;
    font-weight: 800;
    margin: 7px 0 10px 0;
    color: white;
}

.info-card {
    background: linear-gradient(145deg, #142B41, #0E1D30);
    border: 1px solid #28445A;
    border-radius: 18px;
    padding: 23px;
    min-height: 145px;
    transition: all .25s ease;
    box-shadow: 0 8px 25px rgba(0,0,0,.12);
}

.info-card:hover {
    transform: translateY(-5px);
    border-color: #39D9DE;
    box-shadow: 0 12px 35px rgba(25,200,210,.12);
}

.card-icon {
    font-size: 27px;
    margin-bottom: 12px;
}

.card-title {
    font-size: 18px;
    font-weight: 800;
    color: #F4FAFF;
}

.card-text {
    font-size: 14px;
    color: #A9BED0;
    line-height: 1.7;
    margin-top: 8px;
}

.metric-box {
    background: linear-gradient(140deg, #15324A, #102438);
    border: 1px solid #31556C;
    border-radius: 17px;
    padding: 20px;
    text-align: left;
}

.metric-number {
    font-family: 'Outfit', sans-serif;
    color: #49DDE4;
    font-size: 29px;
    font-weight: 800;
}

.metric-label {
    color: #AFC2D3;
    font-size: 13px;
}

.guide-box {
    background: #102438;
    border-left: 4px solid #27CED0;
    border-radius: 10px;
    padding: 17px 20px;
    margin: 12px 0;
}

.heart-pulse {
    font-size: 90px;
    text-align: center;
    animation: heartbeat 1.4s infinite;
    filter: drop-shadow(0 0 20px rgba(41,220,230,.5));
}

@keyframes heartbeat {
    0%, 100% {transform: scale(1);}
    12% {transform: scale(1.12);}
    25% {transform: scale(1);}
    38% {transform: scale(1.09);}
    50% {transform: scale(1);}
}

@keyframes appear {
    from {opacity: 0; transform: translateY(18px);}
    to {opacity: 1; transform: translateY(0);}
}

div.stButton > button {
    background: linear-gradient(90deg, #0EA5B7, #16C7B7);
    color: #031622 !important;
    border: none;
    border-radius: 10px;
    padding: 12px 22px;
    font-weight: 800;
    transition: all .2s ease;
}

div.stButton > button:hover {
    box-shadow: 0 0 22px rgba(39,214,220,.35);
    transform: translateY(-2px);
}

.stForm {
    background: #102438;
    border: 1px solid #294B60;
    border-radius: 17px;
    padding: 20px;
}

[data-testid="stMetric"] {
    background: #12283D;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #29465B;
}

[data-testid="stMetricValue"] {
    color: #47DDE4;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #13283B;
    border-color: #36566B;
}

.stTabs [data-baseweb="tab"] {
    color: #C6D7E5;
}

hr {
    border-color: #27445A;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# REUSABLE COMPONENTS
# ==========================================

def heading(label, title, description=None):
    st.markdown(
        f'<div class="section-label">{label}</div>'
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )
    if description:
        st.write(description)


def card(icon, title, description):
    st.markdown(f"""
    <div class="info-card">
        <div class="card-icon">{icon}</div>
        <div class="card-title">{title}</div>
        <div class="card-text">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(number, label):
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-number">{number}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# TOP NAVIGATION
# ==========================================

nav_left, nav_right = st.columns([2.2, 5.8])
with nav_left:
    st.markdown('<div class="top-brand">❤️ CardioGuard AI</div><div class="top-caption">HEART INTELLIGENCE</div>', unsafe_allow_html=True)
with nav_right:
    page = st.radio(
        "MAIN MENU",
        [
            "Home",
            "Heart Health Guide",
            "AI Prediction",
            "Model Analytics",
            "Explainable AI",
            "About Project"
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="top_navigation"
    )

st.divider()
if model_ready:
    st.caption("● ML model loaded  ·  Educational Research Edition")
elif isinstance(model, Exception):
    st.warning(f"ML model could not be loaded: {model}. Check that models/best_model.pkl exists and was created with compatible library versions.")
else:
    st.warning("ML model file not found. Add models/best_model.pkl to enable AI Prediction. Other sections remain available.")


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">ARTIFICIAL INTELLIGENCE • HEALTH RESEARCH</div>
        <div class="hero-title">
            Understand the Heart.<br>
            Explore the Power of AI.
        </div>
        <div class="hero-text">
            Discover an educational machine learning platform
            that explores heart disease classification,
            predictive analytics and explainable artificial intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    a, b, c, d = st.columns(4)

    with a:
        metric_card("303", "Dataset records")

    with b:
        metric_card("13", "Input features")

    with c:
        metric_card("05", "ML algorithms")

    with d:
        metric_card("88.52%", "Recorded test accuracy")

    st.write("")
    heading(
        "EXPLORE THE PLATFORM",
        "Intelligence meets heart health",
        "Explore the different modules available in CardioGuard AI."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        card(
            "🫀",
            "Heart Health Guide",
            "Learn about heart anatomy, chambers, blood circulation, "
            "coronary arteries and common heart conditions."
        )

    with c2:
        card(
            "🧠",
            "AI Classification",
            "Explore the machine learning model using educational "
            "sample inputs and view its classification output."
        )

    with c3:
        card(
            "📊",
            "Model Analytics",
            "Explore algorithm comparisons, classification metrics, "
            "confusion matrices and ROC curves."
        )

    st.write("")

    c1, c2 = st.columns(2)

    with c1:
        card(
            "🔍",
            "Explainable AI",
            "Explore SHAP visualizations to understand how "
            "features contribute to model predictions."
        )

    with c2:
        card(
            "🧬",
            "Research and Technology",
            "Understand the dataset, methodology, technologies "
            "and limitations of this research project."
        )

    st.write("")
    heading("HEARTBEAT VISUAL", "Every heartbeat matters")

    st.markdown("""
    <div class="info-card" style="text-align:center;">
        <div class="heart-pulse">❤️</div>
        <div style="font-size:20px;font-weight:800;color:#4BE0E5;">
            Heart Health Awareness
        </div>
        <p style="color:#AFC2D3;">
            Learn how the heart works and why cardiovascular
            health awareness is important.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.info(
        "CardioGuard AI is an educational research prototype, "
        "not a clinically validated medical diagnostic application."
    )


# ==========================================
# HEART HEALTH EDUCATION
# ==========================================

elif page == "Heart Health Guide":

    heading(
        "KNOW YOUR HEART",
        "The human heart",
        "The heart is a muscular organ that continuously pumps blood "
        "throughout the body."
    )

    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d"
        "?auto=format&fit=crop&w=1400&q=85",
        use_container_width=True
    )

    st.markdown("""
    <div class="guide-box">
        <b>What is the heart?</b><br>
        The heart is part of the circulatory system. It pumps blood
        to the lungs to receive oxygen and then sends oxygen-rich
        blood to the rest of the body.
    </div>
    """, unsafe_allow_html=True)

    heading("01 • HEART ANATOMY", "Four chambers of the heart")

    st.write(
        "The human heart has four chambers. The upper chambers are "
        "called atria and the lower chambers are called ventricles."
    )

    # Simplified educational heart chamber diagram
    components.html("""
    <div class="info-card">
    <svg viewBox="0 0 600 390" width="100%"
         xmlns="http://www.w3.org/2000/svg">

        <text x="150" y="27" fill="#4BE0E5" font-size="17"
              text-anchor="middle" font-weight="bold">
              RIGHT SIDE
        </text>

        <text x="450" y="27" fill="#4BE0E5" font-size="17"
              text-anchor="middle" font-weight="bold">
              LEFT SIDE
        </text>

        <rect x="65" y="55" width="180" height="115" rx="35"
              fill="#167FA3" stroke="#61DDF0" stroke-width="3"/>
        <rect x="65" y="195" width="180" height="135" rx="35"
              fill="#145C8A" stroke="#61DDF0" stroke-width="3"/>

        <rect x="355" y="55" width="180" height="115" rx="35"
              fill="#B74756" stroke="#FF9DAB" stroke-width="3"/>
        <rect x="355" y="195" width="180" height="135" rx="35"
              fill="#A92D49" stroke="#FF9DAB" stroke-width="3"/>

        <text x="155" y="110" fill="white" font-size="18"
              text-anchor="middle" font-weight="bold">
              RIGHT ATRIUM
        </text>
        <text x="155" y="132" fill="#C8F4FF" font-size="13"
              text-anchor="middle">
              Upper chamber
        </text>

        <text x="155" y="250" fill="white" font-size="17"
              text-anchor="middle" font-weight="bold">
              RIGHT VENTRICLE
        </text>
        <text x="155" y="274" fill="#C8F4FF" font-size="13"
              text-anchor="middle">
              Pumps blood to lungs
        </text>

        <text x="445" y="110" fill="white" font-size="18"
              text-anchor="middle" font-weight="bold">
              LEFT ATRIUM
        </text>
        <text x="445" y="132" fill="#FFE2E7" font-size="13"
              text-anchor="middle">
              Upper chamber
        </text>

        <text x="445" y="250" fill="white" font-size="17"
              text-anchor="middle" font-weight="bold">
              LEFT VENTRICLE
        </text>
        <text x="445" y="274" fill="#FFE2E7" font-size="13"
              text-anchor="middle">
              Pumps blood to body
        </text>

        <path d="M155 170 V195" stroke="#54DFF1"
              stroke-width="5" stroke-linecap="round"/>
        <path d="M445 170 V195" stroke="#FF9BA9"
              stroke-width="5" stroke-linecap="round"/>

        <path d="M245 120 H355" stroke="#D8E8F5"
              stroke-width="2" stroke-dasharray="7 7"/>

        <text x="300" y="365" fill="#9BB0C5" font-size="13"
              text-anchor="middle">
              Simplified educational chamber diagram
        </text>
    </svg>
    </div>
  """, height=850, scrolling=False)

    chambers = [
        (
            "Right Atrium",
            "Receives oxygen-poor blood returning from the body "
            "through the vena cava."
        ),
        (
            "Right Ventricle",
            "Pumps oxygen-poor blood to the lungs through "
            "the pulmonary artery."
        ),
        (
            "Left Atrium",
            "Receives oxygen-rich blood from the lungs "
            "through the pulmonary veins."
        ),
        (
            "Left Ventricle",
            "Pumps oxygen-rich blood through the aorta "
            "to the rest of the body."
        )
    ]

    for title, description in chambers:
        with st.expander(title):
            st.write(description)

    st.divider()

    heading("02 • BLOOD CIRCULATION", "How does blood flow?")

    circulation = [
        ("1", "Body to heart",
         "Oxygen-poor blood returns to the right atrium."),
        ("2", "Heart to lungs",
         "The right ventricle sends blood to the lungs."),
        ("3", "Lungs to heart",
         "Blood receives oxygen and returns to the left atrium."),
        ("4", "Heart to body",
         "The left ventricle pumps oxygen-rich blood through the aorta.")
    ]

    for number, title, description in circulation:
        st.markdown(f"""
        <div class="info-card" style="margin-bottom:12px;">
            <span style="color:#45DCE4;font-size:24px;font-weight:800;">
                {number}.
            </span>
            <span style="font-size:19px;font-weight:800;color:white;">
                {title}
            </span>
            <div class="card-text">{description}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    heading("03 • BLOOD VESSELS", "Arteries, veins and capillaries")

    col1, col2, col3 = st.columns(3)

    with col1:
        card(
            "🔴",
            "Arteries",
            "Carry blood away from the heart. Most arteries carry "
            "oxygen-rich blood."
        )

    with col2:
        card(
            "🔵",
            "Veins",
            "Carry blood towards the heart. Most veins carry "
            "oxygen-poor blood."
        )

    with col3:
        card(
            "🩸",
            "Capillaries",
            "Tiny blood vessels where oxygen, nutrients and waste "
            "are exchanged with tissues."
        )

    st.divider()

    heading("04 • CORONARY CIRCULATION",
            "How does the heart receive its own blood supply?")

    st.write("""
    The heart muscle also needs oxygen and nutrients to work.
    Coronary arteries supply oxygen-rich blood to the heart muscle.
    """)

    st.markdown("""
    <div class="guide-box">
        <b>Left coronary artery:</b> Supplies blood to much of the
        left side of the heart muscle through its branches.
    </div>
    <div class="guide-box">
        <b>Right coronary artery:</b> Supplies blood to areas of the
        right side and other parts of the heart.
    </div>
    <div class="guide-box">
        <b>Coronary artery disease:</b> Plaque buildup can narrow
        coronary arteries and reduce blood flow to the heart muscle.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    heading("05 • HEART CONDITIONS", "Common cardiovascular conditions")

    conditions = [
        (
            "Coronary Artery Disease",
            "Narrowing of coronary arteries due to plaque buildup."
        ),
        (
            "Heart Attack",
            "Occurs when blood flow to part of the heart muscle "
            "is severely reduced or blocked."
        ),
        (
            "Heart Failure",
            "A condition in which the heart cannot pump blood "
            "as effectively as the body needs."
        ),
        (
            "Arrhythmia",
            "A problem involving the rate or rhythm of the heartbeat."
        ),
        (
            "Hypertension",
            "Persistently elevated blood pressure that can increase "
            "cardiovascular risk."
        )
    ]

    for title, description in conditions:
        with st.expander(title):
            st.write(description)

    st.divider()

    heading("06 • AWARENESS", "Risk factors and healthy habits")

    left, right = st.columns(2)

    with left:
        st.markdown("### Common risk factors")
        st.write("""
        - High blood pressure
        - High cholesterol
        - Diabetes
        - Smoking and tobacco exposure
        - Physical inactivity
        - Family history
        - Unhealthy dietary patterns
        """)

    with right:
        st.markdown("### Healthy habits")
        st.write("""
        - Stay physically active
        - Eat a balanced diet
        - Avoid tobacco
        - Get adequate sleep
        - Attend regular health checkups
        - Follow medical advice
        """)

    st.warning(
        "Chest discomfort, breathlessness or other concerning "
        "symptoms need medical assessment. Sudden chest discomfort, "
        "breathing difficulty or fainting can require emergency care. "
        "Contact local emergency services; in India, call 112."
    )

    st.caption(
        "Educational information only. It is not a substitute "
        "for professional medical advice."
    )


# ==========================================
# AI PREDICTION
# ==========================================

elif page == "AI Prediction":

    if not model_ready:
        st.error("Prediction is unavailable because the trained model could not be loaded. Verify models/best_model.pkl and your installed package versions.")
        st.stop()

    if not hasattr(model, "predict_proba"):
        st.error("The loaded model does not support predict_proba. Use the saved pipeline/model that supports probability output, or remove the probability display.")
        st.stop()

    heading(
        "MACHINE LEARNING",
        "Heart Disease Classification",
        "Explore the trained model using sample inputs based on "
        "the UCI Heart Disease dataset."
    )

    st.warning(
        "This is a research demonstration, not a personal health "
        "assessment. Do not enter real patient information or use "
        "the output for medical decisions."
    )

    with st.form("prediction_form"):

        st.markdown("### Personal and dataset attributes")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age", 1, 100, 45
            )

            sex = st.selectbox(
                "Sex",
                ["Female (0)", "Male (1)"]
            )

            cp = st.selectbox(
                "Chest pain type",
                [
                    "Typical angina (1)",
                    "Atypical angina (2)",
                    "Non-anginal pain (3)",
                    "Asymptomatic (4)"
                ]
            )

            trestbps = st.number_input(
                "Resting blood pressure", 50, 250, 120
            )

            chol = st.number_input(
                "Serum cholesterol", 50, 700, 220
            )

            fbs = st.selectbox(
                "Fasting blood sugar",
                ["False (0)", "True (1)"]
            )

            restecg = st.selectbox(
                "Resting ECG",
                [
                    "Normal (0)",
                    "ST-T abnormality (1)",
                    "LV hypertrophy (2)"
                ]
            )

        with col2:
            thalach = st.number_input(
                "Maximum heart rate achieved", 40, 250, 150
            )

            exang = st.selectbox(
                "Exercise-induced angina",
                ["No (0)", "Yes (1)"]
            )

            oldpeak = st.number_input(
                "Oldpeak", 0.0, 10.0, 1.0, step=0.1
            )

            slope = st.selectbox(
                "ST segment slope",
                [
                    "Upsloping (1)",
                    "Flat (2)",
                    "Downsloping (3)"
                ]
            )

            ca = st.selectbox(
                "Number of major vessels (ca)",
                [0, 1, 2, 3]
            )

            thal = st.selectbox(
                "Thal category",
                [
                    "Normal (3)",
                    "Fixed defect (6)",
                    "Reversible defect (7)"
                ]
            )

        submitted = st.form_submit_button(
            "Run AI Classification",
            use_container_width=True
        )

    if submitted:

        values = {
            "age": float(age),
            "sex": int(sex[-2]),
            "cp": int(cp[-2]),
            "trestbps": float(trestbps),
            "chol": float(chol),
            "fbs": int(fbs[-2]),
            "restecg": int(restecg[-2]),
            "thalach": float(thalach),
            "exang": int(exang[-2]),
            "oldpeak": float(oldpeak),
            "slope": int(slope[-2]),
            "ca": float(ca),
            "thal": int(thal[-2])
        }

        input_df = pd.DataFrame([values])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        heading("MODEL OUTPUT", "Classification result")

        if prediction == 1:
            st.error(
                "Positive class: the model classified this "
                "sample as belonging to the positive category."
            )
        else:
            st.success(
                "Negative class: the model classified this "
                "sample as belonging to the negative category."
            )

        st.metric(
            "Model positive-class probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        st.info(
            "The probability is a model output, not a clinically "
            "validated estimate of an individual's chance of disease."
        )


# ==========================================
# MODEL ANALYTICS
# ==========================================

elif page == "Model Analytics":

    heading(
        "PERFORMANCE CENTRE",
        "Model Analytics",
        "Compare the five trained machine learning algorithms."
    )

    csv_path = REPORT / "model_comparison.csv"

    if csv_path.exists():

        results = pd.read_csv(csv_path)

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        charts = [
            ("Accuracy Comparison", "accuracy_comparison.png"),
            ("Performance Metrics", "metrics_comparison.png"),
            ("Confusion Matrices", "confusion_matrices.png"),
            ("ROC Curve Comparison", "roc_curve_comparison.png")
        ]

        for title, filename in charts:

            path = REPORT / filename

            if path.exists():
                st.subheader(title)
                st.image(str(path), use_container_width=True)

    else:
        st.error("Model comparison report not found.")


# ==========================================
# EXPLAINABLE AI
# ==========================================

elif page == "Explainable AI":

    heading(
        "TRANSPARENCY IN AI",
        "Explainable Artificial Intelligence",
        "Understand how model features contribute to the "
        "classification output using SHAP."
    )

    st.markdown("""
    <div class="guide-box">
        <b>What is SHAP?</b><br>
        SHAP stands for SHapley Additive exPlanations. It is a
        method that explains model predictions by assigning
        contribution values to individual features.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Understanding the three SHAP visualizations")

    with st.expander("1. SHAP Summary Plot"):
        st.write(
            "Shows how different feature values influence predictions "
            "across the selected samples. Each dot represents a sample."
        )

    with st.expander("2. SHAP Feature Importance"):
        st.write(
            "Ranks features using their average absolute SHAP values "
            "across the explained samples."
        )

    with st.expander("3. SHAP Waterfall Plot"):
        st.write(
            "Explains the feature contributions for one selected "
            "sample, starting from the model's baseline output."
        )

    charts = [
        ("Feature Impact Summary", "shap_summary.png"),
        ("Global Feature Importance", "shap_bar.png"),
        ("Individual Sample Explanation", "shap_waterfall.png")
    ]

    for title, filename in charts:

        path = SHAP_DIR / filename

        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)

    importance_path = SHAP_DIR / "shap_feature_importance.csv"

    if importance_path.exists():

        st.subheader("Feature Importance Data")

        shap_data = pd.read_csv(importance_path)

        st.dataframe(
            shap_data,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "SHAP explains model behavior, not biological causes "
        "or clinical risk."
    )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    heading(
        "OUR RESEARCH",
        "About CardioGuard AI",
        "An educational machine learning project exploring "
        "heart disease classification."
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">OUR MISSION</div>
        <div class="hero-title" style="font-size:39px;">
            Bringing AI and<br>Heart Health Education Together.
        </div>
        <div class="hero-text">
            CardioGuard AI demonstrates how machine learning,
            data analysis and explainable AI can be combined
            in a healthcare education project.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    heading("PROJECT OBJECTIVE", "Why this project?")

    st.write("""
    The objective is to develop and evaluate machine learning
    classification models using the UCI Heart Disease dataset.

    The project compares different algorithms and uses SHAP
    to explore the contribution of input features to model outputs.
    """)

    heading("DATASET", "UCI Heart Disease Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("303", "Records")

    with col2:
        metric_card("13", "Input features")

    with col3:
        metric_card("2", "Binary target classes")

    st.write("")

    heading("TECHNOLOGY STACK", "Tools behind the project")

    tech = [
        ("🐍", "Python", "Core programming language"),
        ("🐼", "Pandas", "Data manipulation"),
        ("🧮", "Scikit-learn", "Machine learning"),
        ("📊", "Matplotlib", "Data visualization"),
        ("🔍", "SHAP", "Model explainability"),
        ("🌐", "Streamlit", "Interactive web application")
    ]

    cols = st.columns(3)

    for i, (icon, name, description) in enumerate(tech):
        with cols[i % 3]:
            card(icon, name, description)

    st.write("")

    heading("RESEARCH LIMITATIONS", "Important information")

    st.write("""
    - The dataset contains only 303 records.
    - The model has been evaluated using a limited test sample.
    - The binary target combines original categories 1–4.
    - Model performance does not establish clinical validity.
    - The system is not approved or intended for diagnosis.
    - Predictions should never replace professional medical care.
    """)

    st.markdown("""
    <div class="guide-box">
        <b>Project Disclaimer</b><br>
        CardioGuard AI is an educational research prototype.
        It is not a medical device, diagnostic service or
        treatment recommendation system.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:15px;">
    <div style="font-size:20px;font-weight:800;color:#45DCE4;">
        ❤️ CardioGuard AI
    </div>
    <div style="color:#91A9BB;font-size:12px;margin-top:8px;">
        Intelligent Heart Health Research • Built with Python & AI
    </div>
    <div style="color:#71899D;font-size:11px;margin-top:7px;">
        Educational Prototype | Not for Medical Diagnosis
    </div>
</div>
""", unsafe_allow_html=True)
