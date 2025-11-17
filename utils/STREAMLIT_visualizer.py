import matplotlib.pyplot as plt
import numpy as np

def plot_topk_confidence(top_k_predictions):
    labels = [x['class'] for x in top_k_predictions]
    values = [x['confidence'] for x in top_k_predictions]

    fig, ax = plt.subplots()
    ax.barh(labels, values)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top-K predictions")
    ax.invert_yaxis()

    return fig