"""
Visualization utilities for radar predictions
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class RadarVisualizer:
    """Visualization utilities for radar activity recognition"""
    
    def __init__(self, figsize=(15, 10), dpi=150):
        """
        Initialize visualizer
        
        Args:
            figsize (tuple): Default figure size
            dpi (int): Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_prediction_results(self, results, save_path=None):
        """
        Visualize prediction results
        
        Args:
            results (dict): Prediction results from RadarPredictor
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(' Radar Activity Recognition Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Original RD Map
        ax1 = axes[0, 0]
        ax1.imshow(results['original_image'])
        ax1.set_title(' Original RD Map', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Add prediction text
        pred_text = f"Predicted: {results['predicted_class']}\nConfidence: {results['percentage']}"
        ax1.text(0.02, 0.98, pred_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, fontweight='bold')
        
        # Plot 2: Top-K Predictions Bar Chart
        ax2 = axes[0, 1]
        top_k = results['top_k_predictions']
        classes = [pred['class'] for pred in top_k]
        confidences = [pred['confidence'] for pred in top_k]
        
        bars = ax2.barh(classes, confidences, color='skyblue', alpha=0.8)
        ax2.set_xlabel('Confidence Score')
        ax2.set_title(f' Top-{len(top_k)} Predictions', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 1)
        
        # Add percentage labels
        for bar, conf in zip(bars, confidences):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{conf*100:.1f}%', va='center', fontsize=9)
        
        # Plot 3: All Class Probabilities (if available)
        if 'probabilities' in results:
            ax3 = axes[1, 0]
            activities = list(results['probabilities'].keys())
            probs = list(results['probabilities'].values())
            
            bars = ax3.bar(range(len(activities)), probs, color='lightcoral', alpha=0.7)
            ax3.set_xticks(range(len(activities)))
            ax3.set_xticklabels(activities, rotation=45, ha='right')
            ax3.set_ylabel('Probability')
            ax3.set_title(' All Class Probabilities', fontsize=12, fontweight='bold')
            
            # Highlight predicted class
            predicted_idx = activities.index(results['predicted_class'])
            bars[predicted_idx].set_color('darkgreen')
            bars[predicted_idx].set_alpha(1.0)
        else:
            ax3 = axes[1, 0]
            ax3.text(0.5, 0.5, 'Probabilities not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(' Class Probabilities', fontsize=12, fontweight='bold')
        
        # Plot 4: Confidence Meter
        ax4 = axes[1, 1]
        confidence = results['confidence']
        
        # Create a semi-circle confidence meter
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        ax4.plot(x, y, 'k-', linewidth=2)
        ax4.fill_between(x, 0, y, alpha=0.3, color='lightgray')
        
        # Add confidence arc
        conf_theta = np.linspace(0, confidence * np.pi, 50)
        conf_x = r * np.cos(conf_theta)
        conf_y = r * np.sin(conf_theta)
        ax4.fill_between(conf_x, 0, conf_y, alpha=0.7, 
                        color='green' if confidence > 0.8 else 'orange' if confidence > 0.6 else 'red')
        
        # Add confidence text
        ax4.text(0, 0.2, f'{confidence*100:.1f}%', ha='center', va='center', 
                fontsize=20, fontweight='bold')
        ax4.text(0, -0.1, 'Confidence', ha='center', va='center', fontsize=12)
        
        ax4.set_xlim(-1.2, 1.2)
        ax4.set_ylim(-0.2, 1.2)
        ax4.set_aspect('equal')
        ax4.axis('off')
        ax4.set_title(' Prediction Confidence', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f" Visualization saved to: {save_path}")
        
        plt.show()
    
    def plot_batch_results(self, batch_results, save_path=None):
        """
        Visualize batch prediction results
        
        Args:
            batch_results (list): List of prediction results
            save_path (str, optional): Path to save the plot
        """
        n_images = len(batch_results)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), dpi=self.dpi)
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(' Batch Prediction Results', fontsize=16, fontweight='bold')
        
        for i, (result, ax) in enumerate(zip(batch_results, axes)):
            ax.imshow(result['original_image'])
            ax.set_title(f"#{i+1}: {result['predicted_class']}\n{result['percentage']}", 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide extra subplots
        for j in range(n_images, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f" Batch visualization saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
            class_names (list): List of class names
            save_path (str, optional): Path to save the plot
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8), dpi=self.dpi)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(' Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f" Confusion matrix saved to: {save_path}")
        
        plt.show()
