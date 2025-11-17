import streamlit as st

st.set_page_config(page_title="Radar Activity Recognizer", layout="wide")

# Title Section
st.title(" Radar Human Activity Recognizer")

st.markdown(
    """
Welcome to the **Radar Human Activity Recognition Application**.

This tool demonstrates:
- The **CNN models** trained on **Range–Doppler (RD) Maps**
- The **dataset** used for model training
- The **model architecture** and visualization
- A full **demo section** where you can upload RD Map images and view predictions
"""
)

st.divider()

# Navigation Guide
st.subheader(" Navigation Guide")

st.markdown(
    """
-  **DEMO Page:** Test the model using RD Map images  
-  **Data Info:** Learn about the dataset structure and activity classes  
-  **Model Architecture:** View diagrams and summaries of the CNN model  
"""
)
