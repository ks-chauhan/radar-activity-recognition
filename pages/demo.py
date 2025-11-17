import os
import sys

# Add project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from utils.DEMO_predictor import predict
from utils.STREAMLIT_visualizer import plot_topk_confidence


st.title("MODEL TESTING")

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state["results"] = []


# FORM SECTION
with st.form("predict_form"):
    accepted_files = st.file_uploader(
        "Choose Image Files",
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg']
    )
    submitted = st.form_submit_button("PREDICT")


# PROCESS PREDICTION
if submitted:
    st.session_state["results"] = []   # Reset previous results

    if not accepted_files:
        st.error("No RD MAP images passed for testing")
    else:
        for file in accepted_files:
            file.seek(0)  # reset file pointer

            try:
                result = predict(file, 3)
                st.session_state["results"].append((file.name, result))
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")


# DISPLAY RESULTS
for filename, result in st.session_state["results"]:
    with st.expander(filename):
        st.write(f"**Predicted Class:** {result['predicted_class']}")
        st.write(f"**Confidence:** {result['confidence']:.2f}%")

        st.write("### Top-K Predictions")
        fig = plot_topk_confidence(result['top_k_predictions'])
        st.pyplot(fig)
