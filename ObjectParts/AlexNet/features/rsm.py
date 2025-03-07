import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import random

# Import your custom AlexNet model and dataset
from model_features import AlexNet  # Make sure to import correctly based on your project structure
from dataset_features import CustomImageDataset  # Make sure to import correctly based on your project structure

# Set random seed for reproducibility
random.seed(42)

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_features=2725, init_weights=False).to(device)
model.load_state_dict(torch.load('./AlexNet_Features.pth'))
model.eval()

# Load the dataset and namemap
feature_vectors = pd.read_csv('../../../CSLB/updated_feature_matrix.csv', index_col=0)
with open('../../../MAPPING/updated_namemap_verified.json') as nm_file:
    namemap = json.load(nm_file)

# Randomly select 60 concepts
selected_concepts = random.sample(list(namemap.keys()), 40)

# Create the dataset
dataset = CustomImageDataset(
    img_dir='../../../things_data/filtered_object_images',
    feature_vectors=feature_vectors,
    namemap={concept: concept for concept in selected_concepts},  # Use concept names directly as directory names
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
)

loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# Create a mapping from index to concept name
index_to_concept = {idx: concept for concept, idx in dataset.concept_to_index.items()}

def calculate_rsm(activations):
    # Flatten the activations and calculate the Representational Similarity Matrix (RSM)
    activations_flat = activations.view(activations.size(0), -1).cpu().numpy()
    return cosine_similarity(activations_flat)

def generate_rsm_heatmap(rsm, labels, layer_name):
    # Generate and save the heatmap of the RSM
    plt.figure(figsize=(20, 20))
    sns.heatmap(rsm, cmap='viridis', square=True, annot=False, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.title(f'RSM Heatmap for {layer_name}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'rsm/RSM_{layer_name}.png', bbox_inches='tight')
    plt.close()
    np.save(f'rsm/RSM_{layer_name}.npy', rsm)

# Generate RSM heatmaps
with torch.no_grad():
    for images, concept_labels, _ in loader:
        images = images.to(device)
        features, activations = model(images)  # Adjust here based on actual return values

        for layer_index, layer_activation in enumerate(activations):
            # Initialize a dictionary to collect activations for each selected concept
            concept_activations = {concept: [] for concept in selected_concepts}

            # Assuming there's a way to determine the concept name for each image based on concept_labels
            for concept_idx, activation in zip(concept_labels.tolist(), layer_activation):
                concept_name = index_to_concept.get(concept_idx)  # Obtain the concept name
                if concept_name:
                    concept_activations[concept_name].append(activation.cpu())

            # Calculate average activations and generate RSM heatmap
            if any(concept_activations.values()):
                activations_tensor = torch.cat([torch.mean(torch.stack(acts), dim=0).unsqueeze(0)
                                                for acts in concept_activations.values() if acts], dim=0)
                labels = [concept for concept, acts in concept_activations.items() if acts]
                rsm = calculate_rsm(activations_tensor)
                generate_rsm_heatmap(rsm, labels, f"Layer_{layer_index + 1}_Features_RSM")

print("RSM heatmaps have been generated for all selected layers.")
