import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Define the paths where the RSM files are saved for each model
paths = {
    "unified": "unified/rsm/",
    "concepts": "concepts/rsm/",
    "features": "features/rsm/"
}

# Function to load RSMs from files
def load_rsms_from_directory(directory):
    return [np.load(os.path.join(directory, file)) for file in sorted(os.listdir(directory)) if file.endswith('.npy')]

# Load all RSMs for all models
all_rsms = {}
for model_name, model_path in paths.items():
    all_rsms[model_name] = load_rsms_from_directory(model_path)

# Create labels for the axes that indicate the model and layer
axis_labels = []
for model_name, rsms in all_rsms.items():
    for i in range(1, len(rsms) + 1):
        axis_labels.append(f"{model_name}_Layer_{i}")

# Initialize the second-order RSM
num_rsms = len(axis_labels)
second_order_rsm = np.zeros((num_rsms, num_rsms))

# Calculate the second-order RSM
current_index = 0
for model_name, rsms in all_rsms.items():
    for rsm in rsms:
        compare_index = 0
        for other_model_name, other_rsms in all_rsms.items():
            for other_rsm in other_rsms:
                if current_index == compare_index:
                    # Set the diagonal to the highest similarity score, e.g., 1.0
                    second_order_rsm[current_index, compare_index] = 1.0
                else:
                    similarity = cosine_similarity(
                        [rsm[np.triu_indices_from(rsm, k=1)]],
                        [other_rsm[np.triu_indices_from(other_rsm, k=1)]]
                    )[0, 0]
                    second_order_rsm[current_index, compare_index] = similarity
                compare_index += 1
        current_index += 1

# Set the style for the seaborn plot
sns.set(style="white")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap
sns.heatmap(second_order_rsm, cmap='viridis', square=True, cbar_kws={"shrink": .5}, ax=ax, xticklabels=axis_labels, yticklabels=axis_labels)

# Set the title and the axes labels
plt.title('Second-Order RSM Heatmap')
plt.xlabel('Model Layer Combinations')
plt.ylabel('Model Layer Combinations')

# Optional: Rotate the axis labels for better readability
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# Use a tight layout
plt.tight_layout()

# Save the heatmap to a file
plt.savefig('second_order_rsm_heatmap.png', dpi=300)

# Display the heatmap
plt.show()
