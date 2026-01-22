import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "https://emotion-classifier-597500488480.europe-west1.run.app")

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Emotion Classifier")
st.write("Upload a facial image to detect the emotion expressed")

with st.sidebar:
    st.header("Configuration")

    try:
        response = requests.get(f"{BACKEND_URL}/models/", timeout=300)
        if response.status_code == 200:
            models_data = response.json()
            available_models = models_data.get("available_models", {})
            current_model = models_data.get("current_model", "unknown")

            if available_models:
                st.subheader("Available Models")

                model_options = list(available_models.keys())
                index = (
                    model_options.index(current_model)
                    if current_model in model_options
                    else 0
                )
                selected = st.radio(
                    "Select Model:",
                    model_options,
                    index=index,
                    horizontal=True
                )
                st.session_state.selected_model = selected

                st.info(f"**Selected:** {selected.upper()}")

                model_info = available_models[selected]
                if model_info.get("validation_loss"):
                    st.metric("Validation Loss", f"{model_info['validation_loss']:.4f}")

                st.caption(f"File: {model_info.get('filename', 'N/A')}")
                st.caption(f"Path: {model_info['path']}")
            else:
                st.warning("No trained models found in models/ directory")
                st.info("Train models first with: `python train_all_models.py`")
                st.session_state.selected_model = None
        else:
            st.error(f"Could not fetch models: {response.status_code}")
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")

    st.divider()

    st.info(f"**Backend URL:** {BACKEND_URL}")

    if st.button("Check API Status"):
        try:
            response = requests.get(f"{BACKEND_URL}/", timeout=300)
            if response.status_code == 200:
                data = response.json()
                model_name = data.get("model", "unknown")
                st.success("API is online")
            else:
                st.error(f"API error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API")
        except Exception as e:
            st.error(f"Error: {str(e)}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        display_image = image.copy()
        max_width = 300
        aspect_ratio = image.height / image.width
        new_height = int(max_width * aspect_ratio)
        display_image = display_image.resize((max_width, new_height))

        st.image(display_image, caption="Uploaded Image")
        st.caption(f"File: {uploaded_file.name} | Size: {uploaded_file.size} bytes")

with col2:
    st.subheader("Prediction")

    if uploaded_file is not None:
        if st.button("Predict Emotion", key="predict_btn"):
            with st.spinner("Analyzing image..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    headers = {
                        "Authorization": "dtu",
                        "Accept": "application/json"
                    }

                    params = {}
                    if st.session_state.selected_model:
                        params["model_name"] = st.session_state.selected_model

                    response = requests.post(
                        f"{BACKEND_URL}/predict/",
                        files=files,
                        headers=headers,
                        params=params,
                        timeout=300
                    )

                    if response.status_code == 200:
                        result = response.json()

                        emotion = result.get("emotion", "Unknown").upper()
                        confidence = result.get("confidence", 0)

                        st.success(f"### Detected: {emotion}")

                        col_conf, col_time = st.columns(2)
                        with col_conf:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        with col_time:
                            st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))

                        probabilities = result.get("probabilities", {})
                        if probabilities:
                            st.subheader("Emotion Probabilities")

                            prob_data = pd.DataFrame([
                                {"Emotion": e.capitalize(), "Probability": prob}
                                for e, prob in probabilities.items()
                            ]).sort_values("Probability", ascending=False)

                            st.bar_chart(
                                data=prob_data.set_index("Emotion"),
                                height=300
                            )

                            st.dataframe(
                                prob_data.assign(**{"Probability": prob_data["Probability"].apply(lambda x: f"{x*100:.2f}%")}),
                                width=600,
                                hide_index=True
                            )

                    elif response.status_code == 401:
                        st.error("Authentication Failed: Invalid credentials")
                    elif response.status_code == 400:
                        error_data = response.json()
                        error_msg = error_data.get("message", "Bad request")
                        st.error(f"Invalid Image: {error_msg}")
                    elif response.status_code == 500:
                        error_data = response.json()
                        error_msg = error_data.get("message", "Internal server error")
                        st.error(f"Server Error: {error_msg}")
                    else:
                        st.error(f"Unexpected error: {response.status_code}")
                        st.text(response.text)

                except requests.exceptions.Timeout:
                    st.error("Request Timeout: API took too long to respond")
                except requests.exceptions.ConnectionError:
                    st.error(f"Connection Error: Cannot reach API at {BACKEND_URL}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    else:
        st.info("Upload an image to get started")

