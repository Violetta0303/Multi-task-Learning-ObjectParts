import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from model_concepts import AlexNet
from dataset_concepts import CustomImageDataset
from sklearn.metrics.pairwise import cosine_similarity

# Load name mapping
with open('../../../MAPPING/updated_namemap_verified.json') as nm_file:
    namemap = json.load(nm_file)

# Define transformations
# data_transform = {
#     "train": transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ]),
#     "val": transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
# }

# Define transformations with added data augmentation for training
data_transform = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Add random vertical flip
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # Randomly apply a Gaussian blur with a 50% chance
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # Randomly erase parts of the input image
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize the dataset
full_dataset = CustomImageDataset(img_dir='../../../things_data/filtered_object_images', namemap=namemap,
                                  transform=None)

# K-Fold Cross-Validation setup
num_folds = 5
# kf = KFold(n_splits=num_folds, shuffle=True)
y = [full_dataset[i][1] for i in range(len(full_dataset))]
kf = StratifiedKFold(n_splits=num_folds, random_state=None, shuffle=True)

# Training parameters
epochs = 20
batch_size = 128
save_path = './AlexNet_Concepts.pth'

all_train_losses = []
all_val_losses = []
all_train_accuracies = []
all_val_accuracies = []

# Initialize the best accuracy
best_acc = 0.0

# Initialize variables for Early Stopping
best_val_loss = float('inf')  # Initial best validation loss set to infinity
patience = 5  # Allowed number of epochs without improvement
patience_counter = 0  # Tracks epochs without improvement

# Instantiate a gradient scaler for dynamically scaling the gradient to prevent gradient underflow at low precision
scaler = GradScaler()

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset, y)):
    print(f"Starting fold {fold + 1}/{num_folds}")

    # Create subsets for the current fold
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    # Apply transformations
    train_subset.dataset.transform = data_transform["train"]
    val_subset.dataset.transform = data_transform["val"]

    # DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize model, loss function
    net = AlexNet(num_concepts=473, init_weights=True).to(device)
    loss_function = nn.CrossEntropyLoss()

    # Initialize the optimizer with L2 regularization (weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=0.00005, weight_decay=1e-5)

    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    fold_train_losses, fold_val_losses, fold_train_accuracies, fold_val_accuracies = [], [], [], []

    # Before starting the epoch, create a mapping from index back to the concept labels
    index_to_concept = {idx: concept for concept, idx in full_dataset.concept_to_index.items()}

    for epoch in range(epochs):
        net.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Convert indices to concept names
            labels_concept_names = [index_to_concept[label.item()] for label in labels.cpu().numpy()]

            # Use autocast context manager to run the forward pass in mixed precision
            with torch.cuda.amp.autocast():
                outputs, activations = net(inputs)
                loss = loss_function(outputs, labels)

            # Scale the loss and call backward
            scaler.scale(loss).backward()

            # Unscale the gradients and step
            scaler.step(optimizer)

            # Update the scale for the next iteration
            scaler.update()

            # Zero gradients for the next iteration
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = 100.0 * correct_train / total_train

        fold_train_losses.append(epoch_train_loss)
        fold_train_accuracies.append(epoch_train_accuracy)

        # Validation Phase
        net.eval()
        val_running_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = net(inputs)  # Adjust depending on your model's output
                loss = loss_function(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = 100.0 * correct_val / total_val

        fold_val_losses.append(epoch_val_loss)
        fold_val_accuracies.append(epoch_val_accuracy)

        print(f"[Epoch {epoch + 1}] Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_accuracy:.2f}%, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.2f}%")

        # Check for improvement in validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0  # Reset patience counter
            # Save the best model based on validation loss
            torch.save(net.state_dict(), save_path)
            print(f"Epoch {epoch + 1}: Validation loss improved, saving model.")
        else:
            patience_counter += 1
            print(f"Epoch {epoch + 1}: No improvement in validation loss for {patience_counter} epochs.")

        # Early Stopping check
        # if patience_counter >= patience:
        #     print("Early stopping triggered. Exiting training loop.")
        #     break  # Exit the training loop

    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)
    all_train_accuracies.append(fold_train_accuracies)
    all_val_accuracies.append(fold_val_accuracies)

# After all folds, compute and print average metrics across all folds
avg_train_loss = np.mean([np.mean(fold_losses) for fold_losses in all_train_losses])
avg_val_loss = np.mean([np.mean(fold_losses) for fold_losses in all_val_losses])
avg_train_accuracy = np.mean([np.mean(fold_acc) for fold_acc in all_train_accuracies])
avg_val_accuracy = np.mean([np.mean(fold_acc) for fold_acc in all_val_accuracies])

print(f"\nAverage Training Loss: {avg_train_loss:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")
print(f"Average Training Accuracy: {avg_train_accuracy:.2f}%")
print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}%")

# Plotting training and validation loss across folds
plt.figure(figsize=(10, 5))

# Plot training loss
plt.subplot(1, 2, 1)
for i, fold_losses in enumerate(all_train_losses):
    epochs = range(1, len(fold_losses) + 1)
    plt.plot(epochs, fold_losses, label=f'Fold {i+1}')
plt.title('Training Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot validation loss
plt.subplot(1, 2, 2)
for i, fold_losses in enumerate(all_val_losses):
    epochs = range(1, len(fold_losses) + 1)
    plt.plot(epochs, fold_losses, label=f'Fold {i+1}')
plt.title('Validation Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('cross_validation_loss.png')
plt.show()

# Plotting training and validation accuracy across folds
plt.figure(figsize=(10, 5))

# Plot training accuracy
plt.subplot(1, 2, 1)
for i, fold_accuracies in enumerate(all_train_accuracies):
    epochs = range(1, len(fold_accuracies) + 1)
    plt.plot(epochs, fold_accuracies, label=f'Fold {i+1}')
plt.title('Training Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
for i, fold_accuracies in enumerate(all_val_accuracies):
    epochs = range(1, len(fold_accuracies) + 1)
    plt.plot(epochs, fold_accuracies, label=f'Fold {i+1}')
plt.title('Validation Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('cross_validation_accuracy.png')
plt.show()


print('Finished Training')
