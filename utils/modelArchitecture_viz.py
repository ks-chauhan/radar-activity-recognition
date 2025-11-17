import os
import sys

# Add project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torchviz import make_dot
from models.radar_net import PowerfulRadarNet128

print("Generating architecture PNG...")

model = PowerfulRadarNet128()
img_path = "cnn_model_architecture.png"

if not os.path.exists(img_path):
    x = torch.randn(1, 3, 128, 128)
    dot = make_dot(model(x), params=dict(model.named_parameters()))
    dot.render("cnn_model_architecture", format="png")

print("Saved:", img_path)
