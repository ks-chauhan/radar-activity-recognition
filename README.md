# Radar Activity Recognition


## 🎯 Overview

A complete pipeline for human activity recognition using radar Range-Doppler (RD) Map images from the **CI4R-MULTI3** dataset. Achieves **90.42% accuracy** on 11 human activities using a ResNet18-based architecture.

## 📊 Performance

- **Accuracy**: 90.42%
- **Classes**: 11 activities
- **Model**: PowerfulRadarNet128 (11.3M parameters)
- **Input Size**: 128×128 RGB images

## 🏷️ Activity Classes

The model recognizes 11 activities from the CI4R-MULTI3 dataset:

Away, Bend, Crawl, Kneel, Limp, Pick, SStep, Scissor, Sit, Toes, Towards

## 🚀 Quick Start

### Installation

git clone https://github.com/ks-chauhan/radar-activity-recognition.git
cd radar-activity-recognition
pip install -r requirements.txt


### Usage

Single prediction
python main.py "path/to/rd_map.png" --save-viz

Python API
from inference import RadarPredictor
predictor = RadarPredictor()
results = predictor.predict("rd_map.png")
print(f"Activity: {results['predicted_class']} ({results['percentage']})")

## 📚 Citation

### Dataset

@inproceedings{gurbuz2020cross,
title={Cross-frequency training with adversarial learning for radar micro-Doppler signature classification},
author={Gurbuz, Sevgi Z. and Rahman, M. Mahbubur and Kurtoglu, Emre and Macks, Trevor and Fioranelli, Francesco},
booktitle={Radar Sensor Technology XXIV},
year={2020},
doi={10.1117/12.2559155}
}


## 🔍 API Reference

Initialize predictor
predictor = RadarPredictor()

Single prediction
results = predictor.predict(image_path, return_probabilities=True, top_k=3)

Batch prediction
batch_results = predictor.predict_batch(image_list)


## 📄 License

MIT License. Dataset usage subject to CI4R-MULTI3 terms.

## 🙏 Acknowledgments

- **Dataset**: [CI4R-MULTI3](https://github.com/ci4r/CI4R-MULTI3) multi-frequency radar dataset
- **Research**: Laboratory of Computational Intelligence for RADAR (CI4R)
