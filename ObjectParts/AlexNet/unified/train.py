from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from model import AlexNet
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from dataset import CustomImageDataset
import pandas as pd
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def precision_score(predicted_labels, true_labels):
    predicted_labels = predicted_labels.bool()
    true_labels = true_labels.bool()
    true_positives = (predicted_labels & true_labels).sum(dim=1)
    predicted_positives = predicted_labels.sum(dim=1)
    predicted_positives[predicted_positives == 0] = 1
    sample_precision = true_positives / predicted_positives
    precision = sample_precision.mean().item()
    return precision

def recall_score(predicted_labels, true_labels):
    predicted_labels = predicted_labels.bool()
    true_labels = true_labels.bool()    
    true_positives = (predicted_labels & true_labels).sum(dim=1)    
    actual_positives = true_labels.sum(dim=1)
    actual_positives[actual_positives == 0] = 1
    sample_recall = true_positives / actual_positives   
    recall = sample_recall.mean().item()
    return recall

# Load feature vectors and namemap
feature_vectors = pd.read_csv('../../../CSLB/updated_feature_matrix.csv', index_col=0)
with open('../../../MAPPING/updated_namemap_verified.json') as nm_file:
    namemap = json.load(nm_file)

# Define transformations
# data_transform = {
#     "train": transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
#     "val": transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

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

# Initialize datasets
full_dataset = CustomImageDataset(
    img_dir='../../../things_data/filtered_object_images',
    feature_vectors=feature_vectors,
    namemap=namemap,
    transform=data_transform["train"]
)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the number of folds for cross-validation
num_folds = 5
# kf = KFold(n_splits=num_folds, shuffle=True)
y = [full_dataset[i][1] for i in range(len(full_dataset))]
kf = StratifiedKFold(n_splits=num_folds, random_state=None, shuffle=True)

# Loss function and optimizer
loss_function_concepts = nn.CrossEntropyLoss()
loss_function_features = nn.BCEWithLogitsLoss()
# loss_function_features = nn.BCEWithLogitsLoss()

# Training parameters
epochs = 40
batch_size = 128
save_path = './AlexNet.pth'

# Lists for recording training and validation metrics across folds
all_train_losses = []
all_train_losses_concepts = []
all_train_losses_features = []
all_train_accuracies_concepts = []
all_train_accuracies_features = []
all_train_rmses = []

all_val_losses = []
all_val_losses_concepts = []
all_val_losses_features = []
all_val_accuracies_concepts = []
all_val_accuracies_features = []
all_val_rmses = []
all_train_pre, all_val_pre, all_train_rec, all_val_rec= [], [], [], []

# Initialize the best model and best accuracy
best_acc = 0.0

# Initialize variables for Early Stopping
best_val_loss = float('inf')  # Initial best validation loss set to infinity
patience = 5  # Allowed number of epochs without improvement
patience_counter = 0  # Tracks epochs without improvement

# Instantiate a gradient scaler for dynamically scaling the gradient to prevent gradient underflow at low precision
scaler = GradScaler()

# For feature classification
threshold = 0.5

# Cross-validation loop
for fold, (train_index, val_index) in enumerate(kf.split(full_dataset, y)):
    print(f'Starting fold {fold + 1}/{num_folds}')

    # Create subsets for the current fold
    train_subset = Subset(full_dataset, train_index)
    val_subset = Subset(full_dataset, val_index)

    # Apply transformations
    train_subset.dataset.transform = data_transform["train"]
    val_subset.dataset.transform = data_transform["val"]

    # DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validate_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize the AlexNet model
    net = AlexNet(num_concepts=473, num_features=2725, init_weights=True).to(device)

    # Initialize the optimizer with L2 regularization (weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

    fold_train_losses = []
    fold_train_losses_concepts = []
    fold_train_losses_features = []
    fold_train_accuracies_concept = []
    fold_train_accuracies_feature = []
    fold_train_rmses = []

    fold_val_losses = []
    fold_val_losses_concepts = []
    fold_val_losses_features = []
    fold_val_accuracies_concept = []
    fold_val_accuracies_feature = []
    fold_val_rmses = []
    fold_train_pre, fold_val_pre, fold_train_rec, fold_val_rec= [], [], [], []
    # Before starting the epoch, create a mapping from index back to the concept labels
    index_to_concept = {idx: concept for concept, idx in full_dataset.concept_to_index.items()}

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        running_loss_concepts = 0.0
        running_loss_features = 0.0
        train_correct_concept = 0
        train_correct_feature = 0
        train_concept = 0
        train_predictions = []
        train_targets = []
        correct = 0
        total = 0
        tpre = 0
        trec = 0
        for i, (inputs, labels_concepts, labels_features) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels_concepts = labels_concepts.to(device)
            labels_features = labels_features.float().to(device)

            optimizer.zero_grad()

            # Use autocast context manager to run the forward pass in mixed precision
            with torch.cuda.amp.autocast():
                outputs_concepts, outputs_features, activations = net(inputs)
                weight_concepts = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
                weight_features = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
                loss_concepts = loss_function_concepts(outputs_concepts, labels_concepts)
                loss_features = loss_function_features(outputs_features, labels_features)
                total_loss = weight_concepts * loss_concepts + weight_features * loss_features

            # Scale the loss and call backward to create scaled gradients
            scaler.scale(total_loss).backward()

            # Unscale the gradients and step
            scaler.step(optimizer)

            # Update the scale for the next iteration
            scaler.update()

            # Zero gradients for the next iteration
            optimizer.zero_grad(set_to_none=True)

            running_loss += total_loss.item()
            running_loss_concepts += loss_concepts.item()
            running_loss_features += loss_features.item()

            # Compute the index of the max log-probability which corresponds to the predicted class
            _, predicted = torch.max(outputs_concepts, 1)
            # Count the total number of concept labels processed
            train_concept += labels_concepts.size(0)
            # Count the number of correctly predicted concepts
            train_correct_concept += (predicted == labels_concepts).sum().item()

            # Compute predictions for features by applying a threshold
            predicted_features = (outputs_features > threshold).float()
            # Get the number of actual positive features n in labels_features
            actual_features = (labels_features == 1).float()
            n = actual_features.shape[1]
            # Sort each row in predicted_features and get the indices of the top n values
            _, top_indices = torch.topk(predicted_features, n, dim=1)
            # Create a new tensor for predicted features where only the top n values are set to 1, rest are set to 0
            top_predicted_features = torch.zeros_like(actual_features)
            for i in range(top_predicted_features.shape[0]):
                top_predicted_features[i, top_indices[i]] = 1
            # Compare the new tensor of predicted features with actual features to calculate the number of correct predictions
            correct_predictions = (top_predicted_features == actual_features).float().sum()
            # Accumulate the count of correct predictions
            train_correct_feature += correct_predictions.sum().item()

            train_predictions.extend(outputs_features.detach().cpu().numpy())
            train_targets.extend(labels_features.detach().cpu().numpy())
            trainpre = precision_score(predicted_features,labels_features)
            trainrec = recall_score(predicted_features,labels_features)

            tpre += trainpre 
            trec += trainrec

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_loss_concepts = running_loss_concepts / len(train_loader)
        epoch_train_loss_features = running_loss_features / len(train_loader)
        epoch_train_accuracy_concept = 100 * train_correct_concept / train_concept
        epoch_train_accuracy_feature = 100 * train_correct_feature / (train_concept * labels_features.shape[1])
        epoch_train_rmses = sqrt(mean_squared_error(train_targets, train_predictions))
        epoch_train_pre = tpre / len(train_loader)
        epoch_train_rec = trec / len(train_loader)
        fold_train_losses.append(epoch_train_loss)
        fold_train_losses_concepts.append(epoch_train_loss_concepts)
        fold_train_losses_features.append(epoch_train_loss_features)
        fold_train_accuracies_concept.append(epoch_train_accuracy_concept)
        fold_train_accuracies_feature.append(epoch_train_accuracy_feature)
        fold_train_rmses.append(epoch_train_rmses)
        fold_train_pre.append(epoch_train_pre)
        fold_train_rec.append(epoch_train_rec)
        net.eval()
        val_correct_concept = 0
        val_correct_feature = 0
        val_total = 0
        val_loss = 0.0
        val_loss_concepts = 0.0
        val_loss_features = 0.0
        val_predictions = []
        val_targets = []
        pre = 0
        rec = 0
        with torch.no_grad():
            for val_images, val_concepts, val_features in validate_loader:
                val_images = val_images.to(device)
                val_concepts = val_concepts.to(device)
                val_features = val_features.float().to(device)

                outputs_concepts, outputs_features, _ = net(val_images)
                loss_concepts = loss_function_concepts(outputs_concepts, val_concepts)
                loss_features = loss_function_features(outputs_features, val_features)
                total_loss = loss_concepts + loss_features
                val_loss += total_loss.item()
                val_loss_concepts += loss_concepts.item()
                val_loss_features += loss_features.item()


                # Compute accuracy for concept predictions
                _, predicted = torch.max(outputs_concepts, 1)
                val_total += val_concepts.size(0)
                val_correct_concept += (predicted == val_concepts).sum().item()

                # Compute accuracy for concept predictions
                predicted_features = (outputs_features > threshold).float()
                # 获取actual_features的特征数量n
                actual_features = (val_features == 1).float()
                n = actual_features.shape[1]
                # 对predicted_features的每一行进行排序，获取最大的n个值的索引
                _, top_indices = torch.topk(predicted_features, n, dim=1)
                # 生成一个新的预测特征张量，只包含每个样本中前n个最大值的位置设为1，其他设为0
                top_predicted_features = torch.zeros_like(actual_features)
                for i in range(top_predicted_features.shape[0]):
                    top_predicted_features[i, top_indices[i]] = 1
                # 比较top_predicted_features与actual_features，计算正确预测的数量
                correct_predictions = (top_predicted_features == actual_features).float().sum()
                val_correct_feature += correct_predictions.sum().item()
                # Collect features predictions and actual values for calculating RMSE
                val_predictions.extend(outputs_features.detach().cpu().numpy())
                val_targets.extend(val_features.cpu().numpy())
                valpre = precision_score(predicted_features,val_features)
                valrec = recall_score(predicted_features,val_features)
                pre += valpre+precision_score(predicted_features,val_features)
                rec += valrec+recall_score(predicted_features,val_features)
        # Calculate average loss and accuracy
        epoch_val_loss = val_loss / len(validate_loader)
        epoch_val_loss_concepts = val_loss_concepts / len(validate_loader)
        epoch_val_loss_features = val_loss_features / len(validate_loader)
        epoch_val_accuracy_concept = 100 * val_correct_concept / val_total
        epoch_val_accuracy_feature = 100 * val_correct_feature / (val_total * val_features.shape[1])
        epoch_val_rmse = sqrt(mean_squared_error(val_targets, val_predictions))
        epoch_val_pre = pre / len(validate_loader)
        epoch_val_rec = rec / len(validate_loader)
        # Append results to lists for tracking over epochs
        fold_val_losses.append(epoch_val_loss)
        fold_val_losses_concepts.append(epoch_val_loss_concepts)
        fold_val_losses_features.append(epoch_val_loss_features)
        fold_val_accuracies_concept.append(epoch_val_accuracy_concept)
        fold_val_accuracies_feature.append(epoch_val_accuracy_feature)
        fold_val_rmses.append(epoch_val_rmse)
        fold_val_pre.append(epoch_val_pre)
        fold_val_rec.append(epoch_val_rec)
        print(
            f"[Epoch {epoch + 1}] Total Training Loss: {epoch_train_loss:.4f}, Concepts Training Loss: {epoch_train_loss_concepts:.4f}, Features Training Loss: {epoch_train_loss_features:.4f}, Concepts Training Accuracy: {epoch_train_accuracy_concept:.2f}%, Features Training Accuracy: {epoch_train_accuracy_feature:.2f}%, Features Training RMSE: {epoch_train_rmses:.4f}, "
            f"Total Validation Loss: {epoch_val_loss:.4f}, Concepts Validation Loss: {epoch_val_loss_concepts:.4f}, Features Validation Loss: {epoch_val_loss_features:.4f}, Concepts Validation Accuracy: {epoch_val_accuracy_concept:.2f}%, Features Validation Accuracy: {epoch_val_accuracy_feature:.2f}%, Features Validation RMSE: {epoch_val_rmse:.4f}")
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

    # Record metrics for this fold
    all_train_losses.append(fold_train_losses)
    all_train_losses_concepts.append(fold_train_losses_concepts)
    all_train_losses_features.append(fold_train_losses_features)
    all_train_accuracies_concepts.append(fold_train_accuracies_concept)
    all_train_accuracies_features.append(fold_train_accuracies_feature)
    all_train_rmses.append(fold_train_rmses)

    all_val_losses.append(fold_val_losses)
    all_val_losses_concepts.append(fold_val_losses)
    all_val_losses_features.append(fold_val_losses)
    all_val_accuracies_concepts.append(fold_val_accuracies_concept)
    all_val_accuracies_features.append(fold_val_accuracies_feature)
    all_val_rmses.append(fold_val_rmses)
    all_train_pre.append(fold_train_pre)
    all_val_pre.append(fold_val_pre)

    all_train_rec.append(fold_train_rec)
    all_val_rec.append(fold_val_rec)
    
# Print average performance metrics across folds
avg_train_loss = np.mean([np.mean(f) for f in all_train_losses])
avg_train_loss_concepts = np.mean([np.mean(f) for f in all_train_losses_concepts])
avg_train_loss_features = np.mean([np.mean(f) for f in all_train_losses_features])
avg_val_loss = np.mean([np.mean(f) for f in all_val_losses])
avg_val_loss_concepts = np.mean([np.mean(f) for f in all_val_losses_concepts])
avg_val_loss_features = np.mean([np.mean(f) for f in all_val_losses_features])

avg_train_accuracy_concept = np.mean([np.mean(f) for f in all_train_accuracies_concepts])
avg_val_accuracy_concept = np.mean([np.mean(f) for f in all_val_accuracies_concepts])
avg_train_accuracy_feature = np.mean([np.mean(f) for f in all_train_accuracies_features])
avg_val_accuracy_feature = np.mean([np.mean(f) for f in all_val_accuracies_features])

avg_train_rmse = np.mean([np.mean(f) for f in all_train_rmses])
avg_val_rmse = np.mean([np.mean(f) for f in all_val_rmses])
avg_train_pre = np.mean([np.mean(f) for f in all_train_pre])
avg_val_pre = np.mean([np.mean(f) for f in all_val_pre])
avg_train_rec = np.mean([np.mean(f) for f in all_train_rec])
avg_val_rec = np.mean([np.mean(f) for f in all_val_rec])
print(f'\nAverage Total Training Loss: {avg_train_loss:.4f}')
print(f'Average Total Validation Loss: {avg_val_loss:.4f}')

print(f'\nAverage Concepts Training Loss: {avg_train_loss_concepts:.4f}')
print(f'Average Concepts Validation Loss: {avg_val_loss_concepts:.4f}')

print(f'\nAverage Features Training Loss: {avg_train_loss_features:.4f}')
print(f'Average Features Validation Loss: {avg_val_loss_features:.4f}')

print(f'Average Concepts Training Accuracy: {avg_train_accuracy_concept:.2f}%')
print(f'Average Concepts Validation Accuracy: {avg_val_accuracy_concept:.2f}%')

print(f"Average Features Training Accuracy: {avg_train_accuracy_feature:.2f}%")
print(f"Average Features Validation Accuracy: {avg_val_accuracy_feature:.2f}%")

print(f'Average Training RMSE: {avg_train_rmse:.4f}')
print(f'Average Validation RMSE: {avg_val_rmse:.4f}')

print(f'Average Training feature Precision: {avg_train_pre:.4f}')
print(f'Average Validation feature Precision: {avg_val_pre:.4f}')

print(f'Average Training feature Recall: {avg_train_rec:.4f}')
print(f'Average Validation feature Recall: {avg_val_rec:.4f}')
# Plotting training and validation loss across folds
plt.figure(figsize=(10, 5))

# Plot total training loss
plt.subplot(1, 2, 1)
for i in range(num_folds):
    plt.plot(all_train_losses[i], label=f'Fold {i + 1}')
plt.title('Total Training Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot total validation loss
plt.subplot(1, 2, 2)
for i in range(num_folds):
    plt.plot(all_val_losses[i], label=f'Fold {i + 1}')
plt.title('Total Validation Loss across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('total_cross_validation_loss.png')
plt.show()

# Plotting concepts training and validation loss across folds
plt.figure(figsize=(10, 5))

# Plot concepts training loss
plt.subplot(1, 2, 1)
for i in range(num_folds):
    plt.plot(all_train_losses_concepts[i], label=f'Fold {i + 1}')
plt.title('Concepts Training Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot concepts validation loss
plt.subplot(1, 2, 2)
for i in range(num_folds):
    plt.plot(all_val_losses_concepts[i], label=f'Fold {i + 1}')
plt.title('Concepts Validation Loss across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('concepts_cross_validation_loss.png')
plt.show()

# Plotting features training and validation loss across folds
plt.figure(figsize=(10, 5))

# Plot features training loss
plt.subplot(1, 2, 1)
for i in range(num_folds):
    plt.plot(all_train_losses_features[i], label=f'Fold {i + 1}')
plt.title('Features Training Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot features validation loss
plt.subplot(1, 2, 2)
for i in range(num_folds):
    plt.plot(all_val_losses_features[i], label=f'Fold {i + 1}')
plt.title('Features Validation Loss across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('features_cross_validation_loss.png')
plt.show()

# Plotting concepts training and validation accuracy across folds
plt.figure(figsize=(10, 5))

# Plot concepts training accuracy
plt.subplot(1, 2, 1)
for i in range(num_folds):
    plt.plot(all_train_accuracies_concepts[i], label=f'Fold {i + 1}')
plt.title('Concepts Training Accuracy across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Plot concepts validation accuracy
plt.subplot(1, 2, 2)
for i in range(num_folds):
    plt.plot(all_val_accuracies_concepts[i], label=f'Fold {i + 1}')
plt.title('Concepts Validation Accuracy across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('concepts_cross_validation_accuracy.png')
plt.show()

# Plotting features training and validation accuracy across folds
plt.figure(figsize=(10, 5))

# Plot features training accuracy
plt.subplot(1, 2, 1)
for i in range(num_folds):
    plt.plot(all_train_accuracies_features[i], label=f'Fold {i + 1}')
plt.title('Features Training Accuracy across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Plot features validation accuracy
plt.subplot(1, 2, 2)
for i in range(num_folds):
    plt.plot(all_val_accuracies_features[i], label=f'Fold {i + 1}')
plt.title('Features Validation Accuracy across Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('features_cross_validation_accuracy.png')
plt.show()

# Plotting training and validation RMSE across folds
plt.figure(figsize=(10, 5))

# Plot training RMSE
plt.subplot(1, 2, 1)
for i in range(num_folds):
    plt.plot(all_train_rmses[i], label=f'Fold {i + 1}')
plt.title('Features Training RMSE across Folds')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()

# Plot validation RMSE
plt.subplot(1, 2, 2)
for i in range(num_folds):
    plt.plot(all_val_rmses[i], label=f'Fold {i+1}')
plt.title('Features Validation RMSE across Folds')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()

plt.tight_layout()
plt.savefig('features_cross_validation_rmse.png')
plt.show()

print('Finished Training')

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