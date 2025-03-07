import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from model_features import AlexNet
from dataset_features import CustomImageDataset
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


def precision_score(predicted_labels, true_labels):
    # Ensure that both predicted and true labels are boolean types
    predicted_labels = predicted_labels.bool()
    true_labels = true_labels.bool()

    # Calculate the number of true positives per sample (predicted and actual positive)
    true_positives = (predicted_labels & true_labels).sum(dim=1)

    # Calculate the number of predicted positives per sample (true positives + false positives)
    predicted_positives = predicted_labels.sum(dim=1)

    # Avoid division by zero
    predicted_positives[predicted_positives == 0] = 1

    # Calculate precision for each sample
    sample_precision = true_positives / predicted_positives

    # Compute the average precision
    precision = sample_precision.mean().item()

    return precision


def recall_score(predicted_labels, true_labels):
    # Ensure that both predicted and true labels are boolean types
    predicted_labels = predicted_labels.bool()
    true_labels = true_labels.bool()

    # Calculate the number of true positives per sample (predicted and actual positive)
    true_positives = (predicted_labels & true_labels).sum(dim=1)

    # Calculate the number of actual positives per sample (true positives + false negatives)
    actual_positives = true_labels.sum(dim=1)

    # Avoid division by zero
    actual_positives[actual_positives == 0] = 1

    # Calculate recall for each sample
    sample_recall = true_positives / actual_positives

    # Compute the average recall
    recall = sample_recall.mean().item()

    return recall

# Load feature vectors and name mapping
feature_vectors = pd.read_csv('../../../CSLB/updated_feature_matrix.csv', index_col=0)
with open('../../../MAPPING/updated_namemap_verified.json') as nm_file:
    namemap = json.load(nm_file)

# Define transformations
data_transform = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

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

# Initialize the dataset
full_dataset = CustomImageDataset(
    img_dir='../../../things_data/filtered_object_images',
    feature_vectors=feature_vectors,
    namemap=namemap,
    transform=None
)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Training parameters
epochs = 40
batch_size = 128
save_path = './AlexNet_Features.pth'

# K-Fold Cross-Validation setup
num_folds = 5
# kf = KFold(n_splits=num_folds, shuffle=True)
y = [full_dataset[i][1] for i in range(len(full_dataset))]
kf = StratifiedKFold(n_splits=num_folds, random_state=None, shuffle=True)

loss_function = nn.BCEWithLogitsLoss()

# Lists to hold all fold results
all_train_losses, all_val_losses, all_train_acc, all_val_acc, all_train_rmse, all_val_rmse ,all_train_pre, all_val_pre, all_train_rec, all_val_rec= [], [], [], [], [], [], [], [], [], []

# Initialize variables for Early Stopping
best_val_loss = float('inf')  # Initial best validation loss set to infinity
patience = 5  # Allowed number of epochs without improvement
patience_counter = 0  # Tracks epochs without improvement

# Instantiate a gradient scaler for dynamically scaling the gradient to prevent gradient underflow at low precision
scaler = GradScaler()

threshold = 0.5
for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset, y)):
    print(f'Starting fold {fold + 1}/{num_folds}')

    # Create subsets for the current fold
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    # Apply transformations
    train_subset.dataset.transform = data_transform["train"]
    val_subset.dataset.transform = data_transform["val"]

    # DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize model, loss function, optimizer
    net = AlexNet(num_features=2725, init_weights=True).to(device)

    # Initialize the optimizer with L2 regularization (weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Lists to hold per fold results
    fold_train_losses, fold_val_losses, fold_train_acc, fold_val_acc, fold_train_rmse, fold_val_rmse, fold_train_pre, fold_val_pre, fold_train_rec, fold_val_rec= [], [], [], [], [], [], [], [], [], []

    for epoch in range(epochs):  # Number of epochs
        net.train()
        running_loss = 0.0
        train_concept = 0
        train_correct = 0
        tpre = 0
        trec = 0
        train_predictions, train_targets = [], []

        for i, (inputs, labels_concepts, labels_features) in enumerate(train_loader):
            inputs, labels_features = inputs.to(device), labels_features.float().to(device)
            labels_concepts = labels_concepts.to(device)
            optimizer.zero_grad()

            # Use autocast context manager to run the forward pass in mixed precision
            with torch.cuda.amp.autocast():
                outputs_features, activations = net(inputs)
                loss = loss_function(outputs_features, labels_features)

            # Scale the loss and call backward to create scaled gradients
            scaler.scale(loss).backward()

            # Unscale the gradients and step
            scaler.step(optimizer)

            # Update the scale for the next iteration
            scaler.update()

            # Zero gradients for the next iteration
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

            train_concept += labels_concepts.size(0)

            #
            predicted_features = (outputs_features > threshold).float()
            # 获取actual_features的特征数量n
            actual_features = (labels_features == 1).float()
            n = actual_features.shape[1]
            # 对predicted_features的每一行进行排序，获取最大的n个值的索引
            _, top_indices = torch.topk(predicted_features, n, dim=1)
            # 生成一个新的预测特征张量，只包含每个样本中前n个最大值的位置设为1，其他设为0
            top_predicted_features = torch.zeros_like(actual_features)
            for i in range(top_predicted_features.shape[0]):
                top_predicted_features[i, top_indices[i]] = 1
            # 比较top_predicted_features与actual_features，计算正确预测的数量
            correct_predictions = (top_predicted_features == actual_features).float().sum()
            train_correct += correct_predictions.sum().item()

            train_predictions.extend(outputs_features.detach().cpu().numpy())
            train_targets.extend(labels_features.detach().cpu().numpy())
            trainpre = precision_score(predicted_features,labels_features)
            trainrec = recall_score(predicted_features,labels_features)

            tpre += trainpre
            trec += trainrec
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / (train_concept * labels_features.shape[1])
        epoch_train_rmse = sqrt(mean_squared_error(train_targets, train_predictions))
        epoch_train_pre = tpre / len(train_loader)
        epoch_train_rec = trec / len(train_loader)
        fold_train_losses.append(epoch_train_loss)
        fold_train_acc.append(epoch_train_acc)
        fold_train_rmse.append(epoch_train_rmse)
        fold_train_pre.append(epoch_train_pre)
        fold_train_rec.append(epoch_train_rec)
        # Validation loop
        net.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_concept = 0
        val_predictions, val_targets = [], []
        pre = 0
        rec = 0
        with torch.no_grad():
            for val_inputs, val_labels_concepts, val_labels_features in val_loader:
                val_inputs, val_labels_features = val_inputs.to(device), val_labels_features.float().to(device)
                val_labels_concepts = val_labels_concepts.to(device)
                val_outputs_features, _ = net(val_inputs)
                val_loss = loss_function(val_outputs_features, val_labels_features)
                val_running_loss += val_loss.item()

                val_concept += val_labels_concepts.size(0)

                # Compute accuracy for predictions
                predicted_features = (val_outputs_features > threshold).float()
                # Get the number of features n in actual_features
                actual_features = (val_labels_features == 1).float()
                n = actual_features.shape[1]
                # Sort each row in predicted_features and retrieve the indices of the top n values
                _, top_indices = torch.topk(predicted_features, n, dim=1)
                # Create a new predicted features tensor that only contains the top n values set to 1, others set to 0
                top_predicted_features = torch.zeros_like(actual_features)
                for i in range(top_predicted_features.shape[0]):
                    top_predicted_features[i, top_indices[i]] = 1
                # Compare top_predicted_features with actual_features to compute the number of correct predictions
                correct_predictions = (top_predicted_features == actual_features).float().sum()
                val_correct += correct_predictions.sum().item()

                # Collect predictions and actual values for calculating RMSE
                val_predictions.extend(val_outputs_features.detach().cpu().numpy())
                val_targets.extend(val_labels_features.cpu().numpy())

                valpre = precision_score(predicted_features,val_labels_features)
                valrec = recall_score(predicted_features,val_labels_features)

                pre += valpre
                rec += valrec

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / (val_concept * val_labels_features.shape[1])
        epoch_val_rmse = sqrt(mean_squared_error(val_targets, val_predictions))
        epoch_val_pre = pre / len(val_loader)
        epoch_val_rec = rec / len(val_loader)
        fold_val_losses.append(epoch_val_loss)
        fold_val_acc.append(epoch_val_acc)
        fold_val_rmse.append(epoch_val_rmse)
        fold_val_pre.append(epoch_val_pre)
        fold_val_rec.append(epoch_val_rec)
        print(
            f'[Epoch {epoch + 1}] Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_acc:.2f}%, Training RMSE: {epoch_train_rmse:.4f}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.2f}%, Validation RMSE: {epoch_val_rmse:.4f}')
        print('epoch_val_pre')
        print(epoch_val_pre)
        print('epoch_val_rec')
        print(epoch_val_rec)
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

        # scheduler.step()

    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)

    all_train_acc.append(fold_train_acc)
    all_val_acc.append(fold_val_acc)

    all_train_rmse.append(fold_train_rmse)
    all_val_rmse.append(fold_val_rmse)

    all_train_pre.append(fold_train_pre)
    all_val_pre.append(fold_val_pre)

    all_train_rec.append(fold_train_rec)
    all_val_rec.append(fold_val_rec)

# Calculate and print average metrics across all folds
avg_train_loss = np.mean([np.mean(f) for f in all_train_losses])
avg_val_loss = np.mean([np.mean(f) for f in all_val_losses])
avg_train_acc = np.mean([np.mean(f) for f in all_train_acc])
avg_val_acc = np.mean([np.mean(f) for f in all_val_acc])
avg_train_rmse = np.mean([np.mean(f) for f in all_train_rmse])
avg_val_rmse = np.mean([np.mean(f) for f in all_val_rmse])
avg_train_pre = np.mean([np.mean(f) for f in all_train_pre])
avg_val_pre = np.mean([np.mean(f) for f in all_val_pre])
avg_train_rec = np.mean([np.mean(f) for f in all_train_rec])
avg_val_rec = np.mean([np.mean(f) for f in all_val_rec])
print(f'\nAverage Training Loss: {avg_train_loss:.4f}')
print(f'Average Validation Loss: {avg_val_loss:.4f}')

print(f'Average Training Loss: {avg_train_acc:.4f}')
print(f'Average Validation Loss: {avg_val_acc:.4f}')

print(f'Average Training RMSE: {avg_train_rmse:.4f}')
print(f'Average Validation RMSE: {avg_val_rmse:.4f}')

print(f'Average Training Precision: {avg_train_pre:.4f}')
print(f'Average Validation Precision: {avg_val_pre:.4f}')

print(f'Average Training Recall: {avg_train_rec:.4f}')
print(f'Average Validation Recall: {avg_val_rec:.4f}')
# Plotting training and validation loss across folds

# Plot training loss
plt.figure(figsize=(10, 5))
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
for i, losses in enumerate(all_val_losses):
    plt.plot(losses, label=f'Fold {i+1}')
plt.title('Validation Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

plt.savefig('cross_validation_loss.png')
plt.show()

# Plotting training and validation accuracy across folds

# Plot training loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, fold_acc in enumerate(all_train_acc):
    epochs = range(1, len(fold_acc) + 1)
    plt.plot(epochs, fold_acc, label=f'Fold {i + 1}')
plt.title('Training Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Plot validation loss
plt.subplot(1, 2, 2)
for i, fold_acc in enumerate(all_val_acc):
    plt.plot(fold_acc, label=f'Fold {i + 1}')
plt.title('Validation Accuracy Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()

plt.savefig('cross_validation_accuracy.png')
plt.show()

# Plotting training and validation RMSE across folds

# Plot training RMSE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, rmse in enumerate(all_train_rmse):
    plt.plot(rmse, label=f'Fold {i+1} Training RMSE')
plt.title('Training RMSE Across Folds')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()

# Plot validation RMSE
plt.subplot(1, 2, 2)
for i, rmse in enumerate(all_val_rmse):
    plt.plot(rmse, label=f'Fold {i+1} Validation RMSE')
plt.title('Validation RMSE Across Folds')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.tight_layout()

plt.savefig('cross_validation_RMSE.png')
plt.show()

print('Finished Training')

# Plotting training and validation RMSE across folds

# Plot training RMSE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, rmse in enumerate(all_train_rmse):
    plt.plot(rmse, label=f'Fold {i+1} Training RMSE')
plt.title('Training RMSE Across Folds')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()

# Plot validation RMSE
plt.subplot(1, 2, 2)
for i, rmse in enumerate(all_val_rmse):
    plt.plot(rmse, label=f'Fold {i+1} Validation RMSE')
plt.title('Validation RMSE Across Folds')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.tight_layout()

plt.savefig('cross_validation_RMSE.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, pre in enumerate(all_train_pre):
    plt.plot(pre, label=f'Fold {i+1} Training Precision')
plt.title('Training Precision Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

# Plot validation pre
plt.subplot(1, 2, 2)
for i, pre in enumerate(all_val_pre):
    plt.plot(pre, label=f'Fold {i+1} Validation Precision')
plt.title('Validation Precision Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()

plt.savefig('cross_validation_Precision.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, rec in enumerate(all_train_rec):
    plt.plot(pre, label=f'Fold {i+1} Training Recall')
plt.title('Training Recall Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

# Plot validation Recall
plt.subplot(1, 2, 2)
for i, rec in enumerate(all_val_rec):
    plt.plot(rec, label=f'Fold {i+1} Validation Recall')
plt.title('Validation Recall Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.tight_layout()

plt.savefig('cross_validation_Recall.png')
plt.show()
print('Finished Training')