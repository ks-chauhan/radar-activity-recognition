"""
Main entry point for radar activity recognition inference
"""
import argparse
import sys
from pathlib import Path

from inference import RadarPredictor
from utils import RadarVisualizer


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Radar Activity Recognition Inference')
    parser.add_argument('image_path', type=str, help='Path to RD Map image')
    parser.add_argument('--model-path', type=str, help='Path to model weights')
    parser.add_argument('--model-info-path', type=str, help='Path to model info JSON')
    parser.add_argument('--output-dir', type=str, default='./outputs', 
                       help='Output directory for visualizations')
    parser.add_argument('--top-k', type=int, default=3, 
                       help='Number of top predictions to show')
    parser.add_argument('--save-viz', action='store_true', 
                       help='Save visualization plots')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f" Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize predictor
        print(" Initializing Radar Activity Recognition...")
        predictor = RadarPredictor(
            model_path=args.model_path,
            model_info_path=args.model_info_path,
            device=args.device
        )
        
        # Make prediction
        print(f" Analyzing RD Map: {image_path.name}")
        results = predictor.predict(
            image_path, 
            return_probabilities=True, 
            top_k=args.top_k
        )
        
        # Display results
        print("\n" + "="*60)
        print(" PREDICTION RESULTS")
        print("="*60)
        print(f" Image: {image_path.name}")
        print(f" Predicted Activity: {results['predicted_class']}")
        print(f" Confidence: {results['percentage']}")
        print(f"\n Top-{args.top_k} Predictions:")
        for i, pred in enumerate(results['top_k_predictions'], 1):
            print(f"   {i}. {pred['class']}: {pred['percentage']}")
        
        # Model info
        model_info = predictor.get_model_info()
        print(f"\n Model Info:")
        print(f"    Accuracy: {model_info['accuracy']:.2f}%")
        print(f"     Architecture: {model_info['model_architecture']}")
        print(f"    Input Size: {model_info['input_size']}")
        print(f"    Device: {model_info['device']}")
        
        # Visualization
        print(f"\n Generating visualizations...")
        visualizer = RadarVisualizer()
        
        save_path = None
        if args.save_viz:
            save_path = output_dir / f"{image_path.stem}_prediction.png"
        
        visualizer.plot_prediction_results(results, save_path=save_path)
        
        print(" Analysis complete!")
        
    except Exception as e:
        print(f" Error during inference: {str(e)}")
        sys.exit(1)


def demo_batch_inference():
    """Demo function for batch inference"""
    try:
        # Initialize predictor
        predictor = RadarPredictor()
        visualizer = RadarVisualizer()
        
        # Example with multiple images (replace with actual paths)
        image_paths = [
            "path/to/rd_map_1.png",
            "path/to/rd_map_2.png", 
            "path/to/rd_map_3.png"
        ]
        
        # Filter existing paths
        existing_paths = [p for p in image_paths if Path(p).exists()]
        
        if not existing_paths:
            print(" No valid image paths found for batch demo")
            return
        
        print(f" Running batch inference on {len(existing_paths)} images...")
        
        # Batch prediction
        batch_results = predictor.predict_batch(existing_paths, return_probabilities=True)
        
        # Display results
        for i, (path, result) in enumerate(zip(existing_paths, batch_results)):
            print(f"\n Image {i+1}: {Path(path).name}")
            print(f" Predicted: {result['predicted_class']} ({result['percentage']})")
        
        # Visualize batch results
        visualizer.plot_batch_results(batch_results, save_path="batch_predictions.png")
        
    except Exception as e:
        print(f" Error during batch inference: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_batch_inference()
    else:
        main()