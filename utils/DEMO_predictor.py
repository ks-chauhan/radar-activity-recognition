import os
import sys

# Add project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from models.radar_net import PowerfulRadarNet128
from inference.preprocessor import RDMapPreprocessor
from utils.config import Config


def predict(image_input, top_k=3):
    try:
        # Load image
        image_input = Image.open(image_input).convert("RGB")

        config = Config()
        activities = config.ACTIVITIES

        # Initialize model
        model = PowerfulRadarNet128(
            num_classes=config.NUM_CLASSES,
            dropout_rate=0.3
        )
        model_path = "saved_models/RadarNet_128x128_BEST_MODEL.pth"

        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to("cpu")

        # Preprocess
        preprocessor = RDMapPreprocessor()
        input_tensor, original_img, original_array = preprocessor.preprocess_image(image_input)
        input_tensor = input_tensor.to("cpu")

        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            predicted_idx = int(np.argmax(probs))

            top_k_indices = probs.argsort()[::-1][:top_k]
            top_k_predictions = [
                {"class": activities[idx], "confidence": float(probs[idx] * 100)}
                for idx in top_k_indices
            ]

        return {
            "predicted_class": activities[predicted_idx],
            "confidence": float(probs[predicted_idx] * 100),
            "top_k_predictions": top_k_predictions,
            "original_image": original_array,
            "input_tensor": input_tensor
        }

    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")
