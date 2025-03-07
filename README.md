# Multi-task Learning of Objects and Parts

## Overview
This project explores a multi-task learning approach for object and feature classification using Convolutional Neural Networks (CNNs). It builds upon the **AlexNet** architecture to classify images based on both conceptual categories and semantic features, leveraging datasets like CSLB Property Norms and THINGS.

## Features
- **Unified Model**: Performs both **concept classification** (single-label) and **feature classification** (multi-label).
- **Concept Model**: Classifies images into predefined object categories.
- **Feature Model**: Identifies semantic properties of objects from images.
- **Dataset Processing**: Maps CSLB Property Norms with THINGS dataset to train CNN models.
- **Property Norms Infilling**: Uses GPT-4 to augment feature datasets.
- **Representational Similarity Matrices (RSMs)**: Analyzes activation similarities across models.

## Dataset
The project uses two primary datasets:
1. **CSLB Property Norms**: A dataset with semantic features describing object concepts.
2. **THINGS Dataset**: A large-scale image dataset containing 1,854 object concepts.

### Dataset Preprocessing
- **Feature Augmentation**: Augments the CSLB feature matrix by infilling missing property norms.
- **Concept Mapping**: Aligns CSLB concepts with THINGS dataset images.
- **Filtering**: Extracts relevant images from THINGS and restructures them for training.

## Architecture
This project modifies **AlexNet** to support multi-task learning:
- **Unified Model**: Includes two fully connected output layers for both concept and feature classification.
- **Concept Model**: Outputs a probability distribution over object categories.
- **Feature Model**: Outputs a probability distribution over multiple semantic attributes.

### Model Training
- **Concept Classification**: Uses CrossEntropyLoss for multi-class classification.
- **Feature Classification**: Uses MSELoss instead of BCELoss for multi-label classification, considering inter-label dependencies.
- **Optimizer**: Adam optimizer with learning rate decay.
- **Cross-validation**: Implements Stratified K-Fold cross-validation for model robustness.

## Installation
### Requirements
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn nltk scikit-learn openai
```

### Setup
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/your-repo/multi-task-learning.git
cd multi-task-learning
```

## Usage
### Data Preparation
Run the following script to preprocess datasets:
```bash
python preprocess.py
```

### Training
Train the models using:
```bash
python train.py --model unified  # For Unified Model
python train.py --model concept  # For Concept Model
python train.py --model feature  # For Feature Model
```

### Evaluation
Evaluate model performance using:
```bash
python evaluate.py --model unified
```

### Inference
To test model predictions on a new image:
```bash
python predict.py --image sample.jpg --model unified
```

## Results
- The **Unified Model** effectively captures both concept and feature relationships.
- The **Concept Model** achieves high accuracy in classification tasks.
- The **Feature Model** successfully predicts multiple attributes per image.
- **RSM Analysis** shows that the Unified Model integrates feature and concept learning effectively.

## Citation
If you use this project, please cite:
```
@thesis{weng2024multi,
  title={Multi-task Learning of Objects and Parts},
  author={Hailin Weng},
  year={2024},
  institution={Queen's University Belfast}
}
```

## License
MIT License
