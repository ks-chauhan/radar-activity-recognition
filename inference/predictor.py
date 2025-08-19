"""
Main prediction class for radar activity recognition
"""
import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path

from models import PowerfulRadarNet128
from utils import Config
from .preprocessor import RDMapPreprocessor


class RadarPredictor:
    """Main predictor class for radar activity recognition"""
    
    def __init__(self, model_path=None, model_info_path=None, device=None):
        """
        Initialize the predictor
        
        Args:
            model_path (str, optional): Path to model weights
            model_info_path (str, optional): Path to model info JSON
            device (str, optional): Device to run inference on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set paths
        self.model_path = Path(model_path) if model_path else Config.MODEL_PATH
        self.model_info_path = Path(model_info_path) if model_info_path else Config.MODEL_INFO_PATH
        
        # Load model info and setup
        self.model_info = self._load_model_info()
        self.activities = self.model_info['activities']
        self.num_classes = len(self.activities)
        
        # Initialize model and preprocessor
        self.model = self._load_model()
        self.preprocessor = RDMapPreprocessor(target_size=Config.INPUT_SIZE)
        
        print(f" Radar Predictor initialized")
        print(f" Model accuracy: {self.model_info['best_accuracy']:.2f}%")
        print(f"  Activities: {', '.join(self.activities)}")
        print(f" Device: {self.device}")
    
    def _load_model_info(self):
        """Load model information from JSON file"""
        try:
            with open(self.model_info_path, 'r') as f:
                model_info = json.load(f)
            return model_info
        except Exception as e:
            raise RuntimeError(f"Error loading model info: {str(e)}")
    
    def _load_model(self):
        """Load the trained model"""
        try:
            # Initialize model
            model = PowerfulRadarNet128(
                num_classes=self.num_classes,
                dropout_rate=0.3
            )
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def predict(self, image_input, return_probabilities=True, top_k=3):
        """
        Make prediction on a single image
        
        Args:
            image_input: Image input (path, numpy array, or PIL Image)
            return_probabilities (bool): Whether to return class probabilities
            top_k (int): Number of top predictions to return
            
        Returns:
            dict: Prediction results containing:
                - predicted_class: Top predicted class name
                - confidence: Confidence score for top prediction
                - top_k_predictions: List of top-k predictions
                - probabilities: Class probabilities (if requested)
        """
        try:
            # Preprocess image
            input_tensor, original_img, original_array = self.preprocessor.preprocess_image(image_input)
            input_tensor = input_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                
            # Get results
            probs_np = probabilities.cpu().numpy()[0]
            predicted_idx = np.argmax(probs_np)
            
            # Get top-k predictions
            top_k_indices = np.argsort(probs_np)[::-1][:top_k]
            top_k_predictions = [
                {
                    'class': self.activities[idx],
                    'confidence': float(probs_np[idx]),
                    'percentage': f"{probs_np[idx]*100:.2f}%"
                }
                for idx in top_k_indices
            ]
            
            # Prepare results
            results = {
                'predicted_class': self.activities[predicted_idx],
                'confidence': float(probs_np[predicted_idx]),
                'percentage': f"{probs_np[predicted_idx]*100:.2f}%",
                'top_k_predictions': top_k_predictions,
                'original_image': original_array,
                'input_tensor': input_tensor.cpu()
            }
            
            if return_probabilities:
                results['probabilities'] = {
                    activity: float(prob) 
                    for activity, prob in zip(self.activities, probs_np)
                }
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
    
    def predict_batch(self, image_list, return_probabilities=False):
        """
        Make predictions on a batch of images
        
        Args:
            image_list (list): List of image inputs
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            list: List of prediction results for each image
        """
        try:
            # Preprocess batch
            batch_tensor, original_images = self.preprocessor.batch_preprocess(image_list)
            batch_tensor = batch_tensor.to(self.device)
            
            # Make predictions
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = F.softmax(logits, dim=1)
            
            # Process results for each image
            results = []
            probs_np = probabilities.cpu().numpy()
            
            for i, (probs, original_img) in enumerate(zip(probs_np, original_images)):
                predicted_idx = np.argmax(probs)
                
                result = {
                    'predicted_class': self.activities[predicted_idx],
                    'confidence': float(probs[predicted_idx]),
                    'percentage': f"{probs[predicted_idx]*100:.2f}%",
                    'original_image': original_img
                }
                
                if return_probabilities:
                    result['probabilities'] = {
                        activity: float(prob) 
                        for activity, prob in zip(self.activities, probs)
                    }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Error during batch prediction: {str(e)}")
    
    def get_model_info(self):
        """Get detailed model information"""
        return {
            'model_architecture': 'PowerfulRadarNet128 with ResNet18 backbone',
            'input_size': f"{Config.INPUT_SIZE}x{Config.INPUT_SIZE}",
            'num_classes': self.num_classes,
            'activities': self.activities,
            'accuracy': self.model_info['best_accuracy'],
            'epoch_achieved': self.model_info['epoch_achieved'],
            'last_updated': self.model_info['last_updated'],
            'total_parameters': self.model_info['total_parameters'],
            'device': self.device
        }