"""
Image preprocessing pipeline for RD Map images
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path


class RDMapPreprocessor:
    """Preprocessing pipeline for RD Map images"""
    
    def __init__(self, target_size=128):
        """
        Initialize preprocessor
        
        Args:
            target_size (int): Target image size for model input
        """
        self.target_size = target_size
        
        # Define transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )  # ImageNet normalization
        ])
    
    def preprocess_image(self, image_input):
        """
        Preprocess various types of image inputs
        
        Args:
            image_input: Can be:
                - str/Path: path to image file
                - numpy array: image array
                - PIL Image: PIL image object
        
        Returns:
            tuple: (preprocessed_tensor, original_image, original_array)
        """
        try:
            # Handle different input types
            if isinstance(image_input, (str, Path)):
                # Load from file path
                image_path = Path(image_input)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                
                original_img = Image.open(image_path).convert('RGB')
                
            elif isinstance(image_input, np.ndarray):
                # Convert numpy array to PIL
                original_array = image_input.copy()
                
                if image_input.dtype == np.uint8:
                    if len(image_input.shape) == 3:
                        original_img = Image.fromarray(image_input)
                    else:
                        # Convert grayscale to RGB
                        rgb_array = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                        original_img = Image.fromarray(rgb_array)
                else:
                    # Normalize to 0-255 if needed
                    img_normalized = self._normalize_array(image_input)
                    original_img = Image.fromarray(img_normalized)
                
            elif isinstance(image_input, Image.Image):
                # Already a PIL image
                original_img = image_input.convert('RGB')
                
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Convert PIL to numpy for original display
            original_array = np.array(original_img)
            
            # Apply preprocessing transforms
            preprocessed_tensor = self.transform(original_img)
            
            # Add batch dimension
            preprocessed_tensor = preprocessed_tensor.unsqueeze(0)
            
            return preprocessed_tensor, original_img, original_array
            
        except Exception as e:
            raise RuntimeError(f"Error preprocessing image: {str(e)}")
    
    def _normalize_array(self, array):
        """Normalize array to 0-255 uint8 range"""
        if array.max() <= 1.0:
            # Assume it's already normalized
            normalized = (array * 255).astype(np.uint8)
        else:
            # Normalize using min-max
            normalized = ((array - array.min()) / 
                         (array.max() - array.min()) * 255).astype(np.uint8)
        
        # Convert to RGB if grayscale
        if len(normalized.shape) == 2:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        
        return normalized
    
    def batch_preprocess(self, image_list):
        """
        Preprocess a batch of images
        
        Args:
            image_list (list): List of image inputs
            
        Returns:
            tuple: (batch_tensor, original_images_list)
        """
        batch_tensors = []
        original_images = []
        
        for img in image_list:
            tensor, orig_img, orig_array = self.preprocess_image(img)
            batch_tensors.append(tensor.squeeze(0))  # Remove batch dim
            original_images.append(orig_array)
        
        # Stack into batch
        batch_tensor = torch.stack(batch_tensors)
        
        return batch_tensor, original_images
