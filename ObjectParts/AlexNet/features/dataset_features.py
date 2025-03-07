import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, feature_vectors, namemap, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_features = []

        # Filter feature_vectors to include only those concepts present in namemap
        filtered_feature_vectors = feature_vectors[feature_vectors.index.isin(namemap.keys())]

        # Create a mapping from concept names (keys of namemap) to indices
        self.concept_to_index = {concept: idx for idx, concept in enumerate(namemap.keys())}

        for concept in filtered_feature_vectors.index:
            concept_dir = os.path.join(img_dir, concept)  # Use concept directly if namemap keys are concept names
            if os.path.isdir(concept_dir):
                for img_file in os.listdir(concept_dir):
                    img_path = os.path.join(concept_dir, img_file)
                    # Use concept index from concept_to_index, ensuring it exists
                    self.img_features.append((img_path, self.concept_to_index[concept], filtered_feature_vectors.loc[concept].values))

    def __getitem__(self, idx):
        img_path, concept_idx, feature_vector = self.img_features[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, concept_idx, torch.tensor(feature_vector)

    def __len__(self):
        return len(self.img_features)
