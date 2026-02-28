import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import StringIO

# ─── PAGE CONFIGURATION ───
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔍",
    layout="wide"
)

# ─── CUSTOM CSS ───
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        background-color: #8B2500;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        width: 100%;
    }
    .stButton>button:hover { background-color: #5C1A00; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .high-risk { color: #FF0000; font-size: 24px; font-weight: bold; }
    .medium-risk { color: #FFA500; font-size: 24px; font-weight: bold; }
    .low-risk { color: #008000; font-size: 24px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ───
@st.cache_resource
def load_model():
    with open('xgboost_churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

FEATURES = ['Age', 'Gender', 'Tenure', 'Total_Spend',
            'Usage_Frequency', 'Risk_Score', 'Engagement_Score']

# ─── HELPER FUNCTIONS ───
def engineer_features(data):
    df = pd.DataFrame([data])
    df['Risk_Score'] = df['Payment_Delay'] * 0.6 + df['Support_Calls'] * 0.4
    df['Engagement_Score'] = df['Usage_Frequency'] * 0.7
    return df[FEATURES]

def get_risk_level(probability):
    if probability > 0.60:
        return "🔴 High Risk"
    elif probability > 0.30:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"

def get_risk_color(probability):
    if probability > 0.60:
        return "high-risk"
    elif probability > 0.30:
        return "medium-risk"
    else:
        return "low-risk"

# ─── HEADER ───
st.markdown("<h1 style='text-align:center; color:#8B2500;'>🔍 Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:grey;'>Powered by XGBoost | Accuracy: 89.24% | AUC-ROC: 0.9557</p>", unsafe_allow_html=True)
st.markdown("---")

# ─── TABS ───
tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📂 Batch CSV Prediction"])

# ══════════════════════════════════════════
# TAB 1 — SINGLE CUSTOMER PREDICTION
# ══════════════════════════════════════════
with tab1:
    st.subheader("Enter Customer Details")
    st.markdown("Fill in the customer information below to predict churn probability.")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", min_value=18, max_value=65, value=35)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=60, value=12)
        total_spend = st.number_input("Total Spend ($)", min_value=100.0,
                                       max_value=1000.0, value=500.0, step=10.0)

    with col2:
        usage_frequency = st.slider("Usage Frequency", min_value=1, max_value=30, value=15)
        support_calls = st.slider("Support Calls", min_value=0, max_value=10, value=3)
        payment_delay = st.slider("Payment Delay (Days)", min_value=0, max_value=30, value=5)

    with col3:
        st.markdown("#### Customer Summary")
        st.info(f"""
        **Age:** {age}
        **Gender:** {gender}
        **Tenure:** {tenure} months
        **Total Spend:** ${total_spend}
        **Usage Frequency:** {usage_frequency}
        **Support Calls:** {support_calls}
        **Payment Delay:** {payment_delay} days
        """)

    st.markdown("")
    predict_btn = st.button("🔍 Predict Churn", key="single_predict")

    if predict_btn:
        # Prepare data
        gender_encoded = 1 if gender == "Male" else 0
        data = {
            'Age': age, 'Gender': gender_encoded, 'Tenure': tenure,
            'Total_Spend': total_spend, 'Usage_Frequency': usage_frequency,
            'Support_Calls': support_calls, 'Payment_Delay': payment_delay
        }

        df = engineer_features(data)
        scaled = scaler.transform(df)
        probability = float(model.predict_proba(scaled)[0][1])
        prediction = int(model.predict(scaled)[0])
        risk_level = get_risk_level(probability)
        risk_class = get_risk_color(probability)

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(label="Churn Prediction",
                      value="Will Churn ⚠️" if prediction == 1 else "Will Stay ✅")
        with r2:
            st.metric(label="Churn Probability",
                      value=f"{probability * 100:.2f}%")
        with r3:
            st.metric(label="Risk Level", value=risk_level)

        # Progress bar
        st.markdown("**Churn Probability Gauge:**")
        st.progress(probability)

        # Recommendation
        st.markdown("---")
        st.subheader("💡 Recommendation")
        if probability > 0.60:
            st.error("""
            🚨 **Immediate Action Required!**
            This customer is at HIGH risk of churning.
            - Contact customer immediately with a retention offer
            - Offer a discount or plan upgrade
            - Assign a dedicated account manager
            """)
        elif probability > 0.30:
            st.warning("""
            ⚠️ **Monitor This Customer**
            This customer is at MEDIUM risk of churning.
            - Send a re-engagement email campaign
            - Offer product training or support
            - Check in with a satisfaction survey
            """)
        else:
            st.success("""
            ✅ **Customer Looks Healthy**
            This customer is at LOW risk of churning.
            - Continue standard engagement
            - Consider upselling opportunities
            - Reward loyalty with exclusive offers
            """)

# ══════════════════════════════════════════
# TAB 2 — BATCH CSV PREDICTION
# ══════════════════════════════════════════
with tab2:
    st.subheader("Upload CSV for Batch Prediction")
    st.markdown("Upload your customer CSV file to score all customers at once.")

    st.info("""
    **Required CSV columns:**
    CustomerID, Age, Gender, Tenure, Usage_Frequency,
    Support_Calls, Payment_Delay, Total_Spend
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded {len(df):,} customers successfully**")

        # Clean Total_Spend
        if df['Total_Spend'].dtype == object:
            df['Total_Spend'] = df['Total_Spend'].str.replace('$', '', regex=False)\
                                                  .str.replace(',', '', regex=False)\
                                                  .str.strip()
            df['Total_Spend'] = pd.to_numeric(df['Total_Spend'], errors='coerce')

        # Encode Gender
        if df['Gender'].dtype == object:
            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

        # Engineer features
        df['Risk_Score'] = df['Payment_Delay'] * 0.6 + df['Support_Calls'] * 0.4
        df['Engagement_Score'] = df['Usage_Frequency'] * 0.7

        if st.button("🚀 Run Batch Prediction", key="batch_predict"):
            with st.spinner("Scoring all customers..."):
                scaled = scaler.transform(df[FEATURES])
                probabilities = model.predict_proba(scaled)[:, 1]
                predictions = model.predict(scaled)

                df['Churn_Prediction'] = predictions
                df['Churn_Probability'] = probabilities.round(4)
                df['Risk_Level'] = df['Churn_Probability'].apply(
                    lambda x: "High Risk" if x > 0.60
                    else "Medium Risk" if x > 0.30
                    else "Low Risk"
                )

            st.markdown("---")
            st.subheader("📊 Batch Prediction Summary")

            total = len(df)
            high = int((df['Risk_Level'] == 'High Risk').sum())
            medium = int((df['Risk_Level'] == 'Medium Risk').sum())
            low = int((df['Risk_Level'] == 'Low Risk').sum())
            avg_prob = df['Churn_Probability'].mean()

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Customers", f"{total:,}")
            m2.metric("🔴 High Risk", f"{high:,}")
            m3.metric("🟠 Medium Risk", f"{medium:,}")
            m4.metric("🟢 Low Risk", f"{low:,}")
            m5.metric("Avg Churn Probability", f"{avg_prob * 100:.2f}%")

            st.markdown("---")
            st.subheader("🏆 Top 20 Highest Risk Customers")
            top20 = df[['CustomerID', 'Churn_Probability', 'Risk_Level']]\
                      .sort_values('Churn_Probability', ascending=False)\
                      .head(20)
            st.dataframe(top20, use_container_width=True)

            # Download results
            st.markdown("---")
            st.subheader("⬇️ Download Full Results")
            output = df[['CustomerID', 'Churn_Probability', 'Risk_Level']]\
                       .sort_values('Churn_Probability', ascending=False)
            csv = output.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="churn_predictions_results.csv",
                mime="text/csv"
            )

# ─── FOOTER ───
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:grey; font-size:12px;'>
Customer Churn Prediction App | Built with XGBoost & Streamlit | Accuracy: 89.24%
</p>
""", unsafe_allow_html=True)
