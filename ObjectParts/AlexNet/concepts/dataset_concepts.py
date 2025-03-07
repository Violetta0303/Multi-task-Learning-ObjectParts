import os
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, namemap, transform=None):
        self.img_dir = img_dir
        self.namemap = namemap
        self.transform = transform
        self.images = []
        # Using the keys of namemap for indexing
        self.concept_to_index = {concept: idx for idx, concept in enumerate(namemap.keys())}

        for concept in namemap.keys():  # Use keys here to iterate over concepts
            concept_dir = os.path.join(img_dir, concept)  # Assuming the folder names are the same as concept names
            if os.path.isdir(concept_dir):
                for img_file in os.listdir(concept_dir):
                    img_path = os.path.join(concept_dir, img_file)
                    self.images.append((img_path, concept))  # Store the concept name directly

    def __getitem__(self, idx):
        img_path, concept = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        concept_label = self.concept_to_index[concept]  # Use the concept name directly to find its index
        return image, concept_label

    def __len__(self):
        return len(self.images)
