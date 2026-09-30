import time

import dill
import pandas as pd
import streamlit as st

from phishing.predictor import ModelResolver
from phishing.url_predictor import predictor

with open("templates/style.css", encoding="utf-8") as style_file:
    st.markdown(f"<style>{style_file.read()}</style>", unsafe_allow_html=True)

with open("templates/index.html", encoding="utf-8") as layout_file:
    st.markdown(layout_file.read(), unsafe_allow_html=True)

model_resolver = ModelResolver()
model_path = model_resolver.get_latest_model_path()
feature_name_path = model_resolver.get_latest_model_feature_names_file_path()


@st.cache_resource
def load_model():
    """Load the saved phishing model from disk."""
    with open(model_path, "rb") as model_file:
        return dill.load(model_file)


st.markdown(
    """
    <style>
        .container {
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.sidebar.markdown("<h2 class='sidebar-title'>Navigation</h2>", unsafe_allow_html=True)
selected_page = st.sidebar.selectbox("Select a page", ["Home", "History"])

if selected_page == "Home":
    st.sidebar.markdown("<h2 class='sidebar-title'>Prediction Inputs</h2>", unsafe_allow_html=True)

    url_input = st.sidebar.text_area(
        "Enter URLs to Detect",
        placeholder="example1.com, example2.com",
    )

    if st.sidebar.button("🔎 Predict URLs"):
        if url_input:
            url_list = [url.strip() for url in url_input.split(",") if url.strip()]

            with st.spinner("🔍 Predicting... Please wait."):
                for url in url_list:
                    try:
                        prediction, prediction_probability = predictor(
                            model=model,
                            url=url,
                            model_feature_names_file_path=feature_name_path,
                        )
                        confidence = max(prediction_probability[0]) * 100
                        time.sleep(0.2)

                        st.session_state.prediction_history.append(
                            {
                                "URL": url,
                                "Prediction": "Phishing" if prediction == 1 else "Safe",
                                "Confidence": f"{confidence:.2f}%",
                            }
                        )

                        if prediction == 1:
                            st.markdown(
                                f"<div class='result danger'>🚨 <strong>Phishing Detected!</strong><br>"
                                f"URL: <code>{url}</code><br>Confidence: {confidence:.2f}%</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div class='result safe'>✅ <strong>Safe URL</strong><br>"
                                f"URL: <code>{url}</code><br>Confidence: {confidence:.2f}%</div>",
                                unsafe_allow_html=True,
                            )

                    except Exception as exc:
                        st.markdown(
                            f"<div class='result error'>❌ <strong>Error:</strong> Could not process "
                            f"<code>{url}</code><br>{exc}</div>",
                            unsafe_allow_html=True,
                        )

if selected_page == "History":
    st.sidebar.markdown("<h2 class='sidebar-title'>Prediction History</h2>", unsafe_allow_html=True)

    if st.session_state.prediction_history:
        prediction_history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(prediction_history_df, width=1000, height=500)
    else:
        st.markdown("No predictions made yet. Please make predictions on the Home page.")

st.markdown(
    """
    <footer style="text-align: center; margin-top: 30px;">
        <p>Created by Rajat Singh | <a href="https://github.com/Rajatsingh64/phishing_domain_detection.git" target="_blank">GitHub Repo</a> | Powered by Code Interactive</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
