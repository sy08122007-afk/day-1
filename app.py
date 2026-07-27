import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

from backend.ml.predict import UniversalPredictor

st.set_page_config(
    page_title="ChurnSense AI",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.main{
    background:#F5F7FA;
}

h1{
    color:#1565C0;
}

.stButton>button{
    background:#1565C0;
    color:white;
    border-radius:8px;
    height:50px;
    width:220px;
    font-size:18px;
}

.stDownloadButton>button{
    background:#2E7D32;
    color:white;
    border-radius:8px;
}

div[data-testid="metric-container"]{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 0px 10px rgba(0,0,0,.15);
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("🤖 ChurnSense AI")

st.sidebar.markdown("---")

st.sidebar.subheader("Project")

st.sidebar.info("""
Customer Churn Prediction
using Machine Learning
""")

st.sidebar.markdown("### Technology")

st.sidebar.write("""
✔ Python

✔ Streamlit

✔ Pandas

✔ Scikit-Learn

✔ Random Forest

✔ Plotly
""")

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.title("📊 Customer Churn Prediction Dashboard")

st.write("""
Upload any compatible customer dataset
(CSV or Excel) and predict churn.
""")

st.markdown("---")

predictor = UniversalPredictor()

uploaded_file = st.file_uploader(
    "📂 Upload CSV or Excel",
    type=["csv","xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.markdown("---")

    col1,col2=st.columns(2)

    col1.metric(
        "Rows",
        len(df)
    )

    col2.metric(
        "Columns",
        len(df.columns)
    )

    st.markdown("---")

    if st.button("🚀 Predict Churn"):

        results = predictor.predict_dataframe(
            df.copy()
        )

        if "Probability" in results.columns:
            results.rename(
                columns={
                    "Probability":"Churn Probability (%)"
                },
                inplace=True
            )

        if "Prediction" in results.columns:

            results["Prediction"] = results[
                "Prediction"
            ].replace({
                1:"Yes",
                0:"No",
                True:"Yes",
                False:"No",
                "Yes":"Yes",
                "No":"No"
            })
            st.success("✅ Prediction Completed Successfully!")

        total = len(results)
        churn = (results["Prediction"] == "Yes").sum()
        safe = (results["Prediction"] == "No").sum()

        st.markdown("## 📈 Dashboard")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "👥 Total Customers",
            total
        )

        c2.metric(
            "⚠️ Likely to Churn",
            churn
        )

        c3.metric(
            "✅ Safe Customers",
            safe
        )

        st.markdown("---")

        summary = (
            results["Prediction"]
            .value_counts()
            .reset_index()
        )

        summary.columns = [
            "Prediction",
            "Count"
        ]

        left, right = st.columns(2)

        with left:

            st.subheader("📊 Bar Chart")

            fig = px.bar(
                summary,
                x="Prediction",
                y="Count",
                color="Prediction",
                text="Count",
                title="Customer Churn Prediction"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            st.subheader("🥧 Pie Chart")

            fig2 = px.pie(
                summary,
                names="Prediction",
                values="Count",
                hole=0.45,
                title="Prediction Distribution"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        st.markdown("---")

        if "MonthlyCharges" in results.columns:

            st.subheader("💰 Monthly Charges")

            fig3 = px.histogram(
                results,
                x="MonthlyCharges",
                color="Prediction",
                nbins=20,
                title="Monthly Charges Distribution"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

        if "Contract" in results.columns:

            st.subheader("📄 Contract Types")

            contract = (
                results["Contract"]
                .value_counts()
                .reset_index()
            )

            contract.columns = [
                "Contract",
                "Customers"
            ]

            fig4 = px.bar(
                contract,
                x="Contract",
                y="Customers",
                color="Contract",
                text="Customers"
            )

            st.plotly_chart(
                fig4,
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("📋 Prediction Summary")

        st.table(summary)

        st.markdown("---")

        st.subheader("📑 Prediction Results")

        if "Churn Probability (%)" in results.columns:

            results["Confidence (%)"] = results[
                "Churn Probability (%)"
            ]

        st.dataframe(
            results,
            use_container_width=True,
            height=500
        )

        st.markdown("---")

        st.subheader("📌 Insights")

        churn_percentage = round(
            (churn / total) * 100,
            2
        )

        safe_percentage = round(
            (safe / total) * 100,
            2
        )

        insight1, insight2 = st.columns(2)

        with insight1:

            st.info(
                f"""
                **Likely to Churn**

                {churn} Customers

                ({churn_percentage}%)
                """
            )

        with insight2:

            st.success(
                f"""
                **Safe Customers**

                {safe} Customers

                ({safe_percentage}%)
                """
            )

        st.markdown("---")

        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

        st.balloons()

st.markdown("---")

st.markdown(
    """
    <center>

    ### 🤖 ChurnSense AI

    Universal Customer Churn Prediction

    **Algorithm:** Random Forest Classifier

    **Frontend:** Streamlit

    **Backend:** Python

    **Version:** Universal AI Edition

    ©️ 2026 All Rights Reserved

    </center>
    """,
    unsafe_allow_html=True
)