# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
from gemini import generate_gemini_response

# Load trained model
model = joblib.load("dropout_model.pkl")

# Set page config
st.set_page_config(
    page_title="🎓 School Dropout Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("🎓 School Dropout Predictor with AI Copilot")
st.markdown("---")

# Tabs layout for navigation
tab1, tab2 = st.tabs(["📋 Manual Prediction", "📤 Bulk Upload"])

# ======= MANUAL PREDICTION TAB =======
with tab1:
    st.header("📋 Enter Student Data")
    st.markdown("Fill out the form below to predict dropout risk and get AI feedback.")

    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            student_id = st.text_input("Student ID")
            age = st.number_input("Age", min_value=10, max_value=30, value=18)

        with col2:
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=5.0, step=0.1)
            attendance = st.number_input("Attendance Rate", min_value=0, max_value=100, value=80)
            behavioral = st.number_input("Behavioral Rating", min_value=0, max_value=100, value=70)

        with col3:
            study_time = st.number_input("Study Time", min_value=0, value=10)
            parental_support = st.selectbox("Parental Support", ["YES", "NO"])
            extra_class = st.selectbox("Extra Paid Class", ["YES", "NO"])

        submit = st.form_submit_button("Predict")

    if submit:
        # Convert 'Yes'/'No' to binary
        parental_support_binary = 1 if parental_support.upper() == "YES" else 0
        extra_class_binary = 1 if extra_class.upper() == "YES" else 0

        # Build input DataFrame
        input_df = pd.DataFrame({
            "Age": [age],
            "CGPA": [cgpa],
            "Attendance Rate": [attendance],
            "Behavioural Rating": [behavioral],
            "Study Time": [study_time],
            "Parental Support": [parental_support_binary],
            "Extra Paid Class": [extra_class_binary]
        })

        # Align columns to model
        expected_features = model.feature_names_in_
        input_df = input_df[expected_features]

        # 💡 Debug: Show input vs model expectation
        st.write("✅ Model expects these features:", list(expected_features))
        st.write("🧾 You provided these features:", list(input_df.columns))

        # Predict
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100

        st.markdown("---")
        st.subheader(f"🎯 Dropout Risk Score: **{prob:.2f}%**")
        if prediction:
            st.error("❌ This student is at **risk of dropping out**.")
        else:
            st.success("✅ This student is **not at immediate risk**.")

        # AI Copilot
        st.markdown("### 🤖 Gemini AI Copilot Suggestion")
        ai_response = generate_gemini_response(input_df.iloc[0].to_dict(), student_id)
        st.info(ai_response)

# ======= BULK PREDICTION TAB =======
with tab2:
    st.header("📤 Upload Dataset for Bulk Prediction")
    st.markdown("Upload a CSV with the same column structure as the training data.")

    # Sample data preview
    try:
        sample_data = pd.read_csv("MODEL TRAINING DATASET.csv").head()
        st.markdown("### 📌 Sample Format:")
        st.dataframe(sample_data)
    except:
        st.warning("Sample data not found. Please upload your CSV file.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df_encoded = df.copy()

        # Standardize YES/NO
        df_encoded['Parental Support'] = df_encoded['Parental Support'].astype(str).str.upper()
        df_encoded['Extra Paid Class'] = df_encoded['Extra Paid Class'].astype(str).str.upper()

        # Map YES/NO to 1/0
        df_encoded['Parental Support'] = df_encoded['Parental Support'].map({'YES': 1, 'NO': 0})
        df_encoded['Extra Paid Class'] = df_encoded['Extra Paid Class'].map({'YES': 1, 'NO': 0})

        # Drop target if exists
        X = df_encoded.drop(columns=['Dropout'], errors='ignore')

        # Align columns
        expected_features = model.feature_names_in_
        X = X[expected_features]

        # Predict
        df['Dropout Prediction'] = model.predict(X)
        df['Dropout Risk (%)'] = model.predict_proba(X)[:, 1] * 100

        # Generate AI Suggestions
        df['AI Advice'] = [
            generate_gemini_response(row.to_dict(), row.get('Student ID', None))
            for _, row in X.iterrows()
        ]

        st.markdown("---")
        st.subheader("📊 Prediction Results")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "predictions.csv", "text/csv")

# ======= FOOTER LINKS =======
st.markdown("---")
st.markdown("💡 [Calculate CGPA Online](https://cgpacalculator.com.ng/)")
