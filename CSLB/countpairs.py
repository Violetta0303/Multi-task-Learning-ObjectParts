import json

# Load the JSON file
file_path = 'processed_pairs.json'

# Read the file and count the number of (concept, feature) pairs
with open(file_path, 'r') as file:
    data = json.load(file)

# Count the number of pairs
num_pairs = len(data)

print(num_pairs)