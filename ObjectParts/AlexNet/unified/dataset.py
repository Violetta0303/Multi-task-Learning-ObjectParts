import os
import pandas as pd
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import random
import json

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, feature_vectors, namemap, transform=None):
        self.img_dir = img_dir
        self.feature_vectors = feature_vectors
        self.namemap = namemap
        self.transform = transform
        self.img_features = []
        # Use the keys of namemap (CSLB dataset labels) for indexing
        # self.concept_to_index = {value: idx for idx, value in enumerate(namemap.values())}
        self.concept_to_index = {key: idx for idx, key in enumerate(namemap.keys())}

        for concept in feature_vectors.index:
            if concept in namemap:
                # concept_dir = os.path.join(img_dir, namemap[concept]) 这样获得的是值,而不是键
                concept_dir = os.path.join(img_dir, concept)
                if os.path.isdir(concept_dir):
                    for img_file in os.listdir(concept_dir):
                        img_path = os.path.join(concept_dir, img_file)
                        self.img_features.append((img_path, feature_vectors.loc[concept].values))

    def __getitem__(self, idx):
        img_path, feature_vector = self.img_features[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Extract the folder name (THINGS label) and find its index
        concept = os.path.basename(os.path.dirname(img_path))
        concept_label = self.concept_to_index[concept]
        return image, concept_label, torch.tensor(feature_vector)

    def __len__(self):
        return len(self.img_features)


# Path to image directory, feature vectors and namemap file
img_dir = '../../../things_data/filtered_object_images'
feature_vectors = pd.read_csv('../../../CSLB/updated_feature_matrix.csv', index_col=0)
namemap_dir = '../../../MAPPING/updated_namemap_verified.json'  # Assume this is a dictionary loaded elsewhere
with open(namemap_dir, 'r') as file:
    namemap = json.load(file)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create an instance of the dataset
dataset = CustomImageDataset(img_dir=img_dir, feature_vectors=feature_vectors, namemap=namemap, transform=transform)


# Function to visualize an image and its feature labels
def visualize_sample(image, feature_vector, feature_names):
    # Check if the image is a torch.Tensor and convert it for visualization
    if isinstance(image, torch.Tensor):
        # Convert from CxHxW to HxWxC for visualization
        image = image.permute(1, 2, 0).numpy()  # Assuming image data is in the range [0, 1]
    plt.imshow(image)  # Display the image
    plt.title("Sample Image")
    plt.show()

    # Filter and display feature names where the feature vector is 1
    positive_features = [name for value, name in zip(feature_vector, feature_names) if value == 1]
    print("Positive Features:", ", ".join(positive_features))


# Load feature names from the DataFrame (assuming it's globally accessible)
feature_names = feature_vectors.columns.tolist()

# Test __len__ method to print the total number of samples in the dataset
print("Total number of samples in the dataset:", len(dataset))

# Test __getitem__ to fetch and visualize a random sample
random_index = random.randint(0, len(dataset) - 1)
sample_image, sample_concept_label, sample_feature_vector = dataset[random_index]

# Since we are using ToTensor in transform, the image is already a tensor, we directly pass it
visualize_sample(sample_image, sample_feature_vector.numpy(), feature_names)

# Print out the concept label index to verify it matches with the dataset
print("Concept label index for the sample:", sample_concept_label)


