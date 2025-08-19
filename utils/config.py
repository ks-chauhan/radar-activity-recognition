"""
Configuration settings for radar inference
"""
import os
from pathlib import Path


class Config:
    """Configuration class for radar inference pipeline"""
    
    # Model settings
    INPUT_SIZE = 128
    NUM_CLASSES = 11
    DROPOUT_RATE = 0.3
    
    # Activities (can be loaded from model info)
    ACTIVITIES = [
        "Away", "Bend", "Crawl", "Kneel", "Limp", 
        "Pick", "SStep", "Scissor", "Sit", "Toes", "Towards"
    ]
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "saved_models"
    MODEL_PATH = MODELS_DIR / "RadarNet_128x128_BEST_MODEL.pth"
    MODEL_INFO_PATH = MODELS_DIR / "RadarNet_128x128_BEST_MODEL_info.json"
    
    # Device settings
    DEVICE = "cuda" if hasattr(os, 'environ') and 'CUDA_VISIBLE_DEVICES' in os.environ else "cpu"
    
    # Visualization settings
    FIGURE_SIZE = (15, 10)
    DPI = 150
    
    @classmethod
    def update_from_model_info(cls, model_info):
        """Update config from loaded model info"""
        cls.NUM_CLASSES = model_info.get('num_classes', cls.NUM_CLASSES)
        cls.ACTIVITIES = model_info.get('activities', cls.ACTIVITIES)
        cls.INPUT_SIZE = int(model_info.get('input_size', '128x128').split('x')[0])
