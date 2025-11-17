import os
import sys

# Add project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from torchinfo import summary
from models.radar_net import PowerfulRadarNet128

st.title("CNN Models Architecture")

st.markdown("""
### **PowerfulRadarNet128 Overview**

This model is built for radar-based human activity recognition using **128×128 range–Doppler images**.  
It combines a **pretrained ResNet18 backbone** with a lightweight **fully-connected classifier head**.


### **1. Input**
- 3-channel radar image (128 × 128)
- Preprocessing handled by transforms before feeding into the network


### **2. Feature Extractor – ResNet18 (Pretrained)**
- Loaded with **ImageNet1K weights**
- All convolutional + residual blocks are retained
- The final fully-connected (FC) layer is **removed**
- Output shape after backbone: **(batch, 512, 1, 1)**

This gives a strong feature representation even with a small dataset.


### **3. Classification Head**
A simple but effective 3-layer MLP:

---

            """)

model = PowerfulRadarNet128()

with st.expander("Detailed Model Architecture"):
    model = PowerfulRadarNet128()
    st.text(summary(model, input_size = (1, 3, 128, 128)))
    st.image("utils/cnn_model_architecture.png")