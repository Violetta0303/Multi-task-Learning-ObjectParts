import pandas as pd

# Load the original and updated DataFrames
original_df = pd.read_csv('feature_matrix.csv', index_col=0)  # Assuming the first column is the index
updated_df = pd.read_csv('updated_feature_matrix.csv', index_col=0)  # Assuming the first column is the index

# Compare the two DataFrames
comparison = original_df.compare(updated_df)

# Check if the comparison DataFrame is empty
if comparison.empty:
    print("No changes were found between the two DataFrames.")
else:
    print("Differences were found:")
    print(comparison)


