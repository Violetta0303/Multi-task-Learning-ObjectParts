import json

# Load the original namemap file
with open('updated_namemap_verfied.json', 'r') as file:
    namemap = json.load(file)

# Sort the namemap by keys
sorted_namemap = dict(sorted(namemap.items(), key=lambda item: item[0]))

# Save the sorted namemap to a new file
with open('sorted_updated_namemap_verfied.json', 'w') as sorted_file:
    json.dump(sorted_namemap, sorted_file, indent=4)
