import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

xgb_model = load_model()

# Default values for other features
feature_defaults = {
    'Guaranteed_Approved__Loan_Clean': 1000000,
    'Loan_Approved_Gross': 1000000,
    'Gross_Amount_Disbursed': 950000,
    'ChargedOff_Amount': 0,
    'Year_Of_Commitment': 2021,
    'Date_Of_Disbursement': '2021-01-01',
    'Commitment_Date': '2020-12-01'
}

# Derived features calculator
def compute_derived_features(inputs):
    df = pd.DataFrame([inputs])

    df['ChargedOff_Amount_Clean'] = df['ChargedOff_Amount']
    df['Loan_Approved_Gross_Clean'] = df['Loan_Approved_Gross']
    df['Gross_Amount_Disbursed_Clean'] = df['Gross_Amount_Disbursed']

    df['ChargedOff_to_Approved'] = df['ChargedOff_Amount_Clean'] / df['Guaranteed_Approved__Loan_Clean']
    df['Disbursed_to_Approved'] = df['Gross_Amount_Disbursed_Clean'] / df['Guaranteed_Approved__Loan_Clean']

    df['Year_Of_Commitment'] = pd.to_numeric(df['Year_Of_Commitment'])
    df['Commitment_Date'] = pd.to_datetime(df['Commitment_Date'])
    df['Date_Of_Disbursement'] = pd.to_datetime(df['Date_Of_Disbursement'])

    df['Loan_Age_Days'] = (pd.Timestamp.today() - df['Date_Of_Disbursement']).dt.days
    df['Processing_Delay_Days'] = (df['Date_Of_Disbursement'] - df['Commitment_Date']).dt.days
    df['Commitment_to_Disbursement_Ratio'] = df['Processing_Delay_Days'] / df['Loan_Age_Days']

    expected_features = xgb_model.get_booster().feature_names
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    return df[expected_features]

# Load sample data for dashboard
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'Year_Of_Commitment': np.random.randint(2015, 2023, n),
        'ChargedOff_to_Approved': np.random.rand(n),
        'Disbursed_to_Approved': np.random.rand(n),
        'Classification_Code': np.random.choice(['A', 'B', 'C'], n),
        'State_Of_Bank': np.random.choice(['NY', 'CA', 'TX', 'VA'], n),
        'Default': np.random.randint(0, 2, n),
        'Loan_Age_Days': np.random.randint(30, 2000, n)
    })
    return df

sample_data = load_sample_data()

# Tabs
tabs = st.tabs(["📈 Predict Default", "📊 Dashboard"])

# Tab 1 - Predict
with tabs[0]:
    st.title("📈 Predict Loan Default")
    st.write("Enter loan and borrower details below:")

    col1, col2 = st.columns(2)
    with col1:
        loan_amount = st.number_input("Guaranteed Approved Loan Amount", value=feature_defaults['Guaranteed_Approved__Loan_Clean'])
        disbursed = st.number_input("Gross Amount Disbursed", value=feature_defaults['Gross_Amount_Disbursed'])
        charged_off = st.number_input("Charged Off Amount", value=feature_defaults['ChargedOff_Amount'])
        loan_approved = st.number_input("Loan Approved Gross", value=feature_defaults['Loan_Approved_Gross'])
    with col2:
        year_commitment = st.number_input("Year of Commitment", value=feature_defaults['Year_Of_Commitment'])
        disbursement_date = st.date_input("Date of Disbursement", value=pd.to_datetime(feature_defaults['Date_Of_Disbursement']))
        commitment_date = st.date_input("Commitment Date", value=pd.to_datetime(feature_defaults['Commitment_Date']))

    if st.button("Predict"):
        user_inputs = {
            'Guaranteed_Approved__Loan_Clean': loan_amount,
            'Gross_Amount_Disbursed': disbursed,
            'ChargedOff_Amount': charged_off,
            'Loan_Approved_Gross': loan_approved,
            'Year_Of_Commitment': year_commitment,
            'Date_Of_Disbursement': disbursement_date,
            'Commitment_Date': commitment_date
        }

        input_df = compute_derived_features(user_inputs)
        prediction = xgb_model.predict(input_df)[0]
        probability = xgb_model.predict_proba(input_df)[0][1]

        st.subheader("📌 Prediction Result")
        if prediction == 1:
            st.error(f"❌ Loan is likely to default. Risk score: {probability:.2%}")
        else:
            st.success(f"✅ Loan is unlikely to default. Risk score: {probability:.2%}")

# Tab 2 - Dashboard
with tabs[1]:
    st.title("📊 Dashboard - Loan Feature Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Default Rate by Classification Code")
        fig1, ax1 = plt.subplots()
        grouped = sample_data.groupby('Classification_Code')['Default'].mean().reset_index()
        sns.barplot(data=grouped, x='Classification_Code', y='Default', ax=ax1, palette='pastel')
        ax1.set_ylabel("Average Default Rate")
        st.pyplot(fig1)

    with col2:
        st.markdown("#### Distribution of Loan Age Days")
        fig2, ax2 = plt.subplots()
        sns.histplot(sample_data['Loan_Age_Days'], bins=20, kde=True, color='#86bf91', ax=ax2)
        ax2.set_xlabel("Loan Age (days)")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### ChargedOff_to_Approved by State")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=sample_data, x='State_Of_Bank', y='ChargedOff_to_Approved', palette='Set3', ax=ax3)
        st.pyplot(fig3)

    with col4:
        st.markdown("#### Disbursed vs Charged Off Ratio")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(data=sample_data, x='Disbursed_to_Approved', y='ChargedOff_to_Approved', hue='Default', palette='coolwarm', ax=ax4)
        st.pyplot(fig4)

    st.markdown("#### Feature Correlation Heatmap")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    corr = sample_data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax5, linewidths=0.5)
    st.pyplot(fig5)

    st.markdown("---")
    st.caption("Visual insights generated from simulated data. Plug in your own data for real-world use.")
