
import streamlit as st
import os
import pandas as pd
import base64

st.set_page_config(layout="wide")
st.title("☀️ Solar Forecast Dashboard with Report Integration")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Forecasting", "📈 Report"])

if section == "Forecasting":
    st.info("🚧 This placeholder is for the model forecast UI. You can integrate your existing logic here.")

elif section == "📈 Report":
    st.header("📈 Performance Report")

    # Show metrics if available
    metrics_path = "outputs/metrics.json"
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.subheader("Model Performance Metrics")
        st.metric("MAE (W/m²)", f"{metrics['MAE']:.2f}")
        st.metric("RMSE (W/m²)", f"{metrics['RMSE']:.2f}")
        st.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")
        st.metric("R²", f"{metrics['R²']:.3f}")
    else:
        st.warning("⚠️ No metrics file found at outputs/metrics.json")

    # Display plots
    for label, filename in [
        ("Forecast vs Actual (95% CI)", "forecast_vs_actual_ci.png"),
        ("Residuals Plot", "residuals_plot.png"),
        ("SHAP Feature Importance", "feature_importance_shap.png")
    ]:
        path = os.path.join("outputs", filename)
        if os.path.exists(path):
            st.subheader(label)
            st.image(path, use_column_width=True)
        else:
            st.warning(f"❌ Missing plot: {filename}")

    # Display HTML report preview
    html_path = "outputs/final_report.html"
    if os.path.exists(html_path):
        st.subheader("📄 Full HTML Report")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
            st.components.v1.html(html, height=800, scrolling=True)

        with open(html_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/html;base64,{b64}" download="final_report.html">📥 Download Full Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("❌ final_report.html not found.")
