import streamlit as st
import pandas as pd
import dill
import time
from phishing.url_predictor import predictor
from phishing.predictor import ModelResolver

# === Load custom styling ===
with open("templates/style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# === Load main HTML layout ===
with open("templates/index.html", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

resolver=ModelResolver()
model_path=resolver.get_latest_model_path()
model_feature_names_file_path=resolver.get_latest_top_features_file_path()

# === Load model ===
@st.cache_resource
def load_model():
    """Load and return the trained phishing detection model from file."""
    with open(model_path, "rb") as file:
        return dill.load(file)

# Add custom CSS to create space after the container
st.markdown("""
    <style>
        /* Add bottom margin to the container */
        .container {
            margin-bottom: 10px; /* Adjust this value for more or less space */
        }
    </style>
""", unsafe_allow_html=True)

model = load_model()

# Initialize prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# === Sidebar UI ===
st.sidebar.markdown("<h2 class='sidebar-title'>Navigation</h2>", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.selectbox('Select a page', ['Home', 'History'])

# === Home Page ===
if page == 'Home':
    st.sidebar.markdown("<h2 class='sidebar-title'>Prediction Inputs</h2>", unsafe_allow_html=True)

    # Text area for multiple URLs
    urls = st.sidebar.text_area(
        "Enter URLs to Detect", 
        placeholder="example1.com, example2.com"
    )

    # Predict button
    if st.sidebar.button("🔎 Predict URLs"):
        if urls:
            url_list = [u.strip() for u in urls.split(",") if u.strip()]

            with st.spinner("🔍 Predicting... Please wait."):
                for url in url_list:
                    try:
                        # Make prediction using the model
                        prediction, prediction_prob = predictor(model=model, #loaded model
                                                                url=url ,
                                                                 model_feature_names_file_path=model_feature_names_file_path #path of trained features names pickle file
                                                        )
                        confidence = max(prediction_prob[0]) * 100
                        time.sleep(0.2)  # Optional delay to simulate processing time
                        # Store result in session state
                        st.session_state.prediction_history.append({
                                'URL': url,
                                'Prediction': 'Phishing' if prediction == 1 else 'Safe' ,
                                'Confidence': f"{confidence:.2f}%"
                            })


                        # Show result with appropriate styling
                        if prediction == 1:
                            st.markdown(
                                f"<div class='result danger'>🚨 <strong>Phishing Detected!</strong><br>"
                                f"URL: <code>{url}</code><br>Confidence: {confidence:.2f}%</div>",
                                unsafe_allow_html=True
                            )
                        else :
                            st.markdown(
                                f"<div class='result safe'>✅ <strong>Safe URL</strong><br>"
                                f"URL: <code>{url}</code><br>Confidence: {confidence:.2f}%</div>",
                                unsafe_allow_html=True
                            )
                        
                    except Exception as e:
                        # Display error message if URL processing fails
                        st.markdown(
                            f"<div class='result error'>❌ <strong>Error:</strong> Could not process "
                            f"<code>{url}</code><br>{e}</div>",
                            unsafe_allow_html=True
                        )

# === History Page ===
if page == 'History':
    st.sidebar.markdown("<h2 class='sidebar-title'>Prediction History</h2>", unsafe_allow_html=True)

    # Display prediction history if available
    if st.session_state.prediction_history:
        df_history = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_history , width=1000 , height=500)
    else:
        st.markdown("No predictions made yet. Please make predictions on the Home page.")

# === Footer ===
st.markdown("""
    <footer style="text-align: center; margin-top: 30px;">
        <p>Created by Rajat Singh | <a href="https://github.com/Rajatsingh64/phishing_domain_detection.git" target="_blank">GitHub Repo</a> | Powered by Code Interactive</p>
    </footer>
""", unsafe_allow_html=True)
