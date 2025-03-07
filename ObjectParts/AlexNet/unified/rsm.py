import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from model import AlexNet  # Make sure to import correctly according to your project structure
from dataset import CustomImageDataset  # Make sure to import correctly according to your project structure
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import random

# Set the random seed for reproducibility
random.seed(42)

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_concepts=473, num_features=2725, init_weights=False).to(device)
model.load_state_dict(torch.load('./AlexNet.pth'))
model.eval()

# Load the dataset and namemap
feature_vectors = pd.read_csv('../../../CSLB/updated_feature_matrix.csv', index_col=0)
with open('../../../MAPPING/updated_namemap_verified.json') as nm_file:
    namemap = json.load(nm_file)

# Randomly select 60 concepts
selected_concepts = random.sample(list(namemap.keys()), 40)

# Namemap containing only the selected concepts
selected_namemap = {k: v for k, v in namemap.items() if k in selected_concepts}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the dataset based on the selection criteria
dataset = CustomImageDataset(
    img_dir='../../../things_data/filtered_object_images',
    feature_vectors=feature_vectors,
    namemap=selected_namemap,
    transform=transform
)

loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# Create a mapping from index to concept name
index_to_concept = {idx: concept for concept, idx in dataset.concept_to_index.items()}

def calculate_rsm(activations):
    # Flatten the activations and calculate the Representational Similarity Matrix (RSM)
    activations_flat = activations.view(activations.size(0), -1).cpu().numpy()
    rsm = cosine_similarity(activations_flat)
    return rsm

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

# Generate RSM heatmaps for each layer of activations
with torch.no_grad():
    for images, concept_labels, _ in loader:
        images = images.to(device)
        concepts, features, activations = model(images)

        for layer_index, layer_activation in enumerate(activations):
            # Initialize a dictionary to collect activations for each selected concept
            concept_activations = {concept: [] for concept in selected_concepts}
            for concept_idx, activation in zip(concept_labels.tolist(), layer_activation):
                concept_name = index_to_concept[concept_idx]  # Get the concept name using the reverse mapping
                concept_activations[concept_name].append(activation.cpu())

            # Check if the activation list for each concept is empty before calculating the average
            average_activations = {}
            for concept, acts in concept_activations.items():
                if acts:  # Check if the list is not empty
                    average_activations[concept] = torch.mean(torch.stack(acts), dim=0)

            if not average_activations:
                # Skip the current layer if all concept activation lists are empty
                continue

            activations_tensor = torch.stack(list(average_activations.values()))
            labels = list(average_activations.keys())

            rsm = calculate_rsm(activations_tensor)
            generate_rsm_heatmap(rsm, labels, f"Layer_{layer_index + 1}")

print("RSM heatmaps have been generated for all selected layers.")
