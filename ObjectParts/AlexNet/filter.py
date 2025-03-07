import os
import shutil
import json

# Path to the namemap mapping file
namemap_file = '..\\..\\MAPPING\\updated_namemap_verified.json'

# Load the namemap mapping and create its reverse
try:
    with open(namemap_file, 'r') as file:
        namemap = json.load(file)
except FileNotFoundError:
    print("Error: The specified namemap file was not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: The namemap file contains invalid JSON.")
    exit(1)

# Paths to THINGS dataset images and the directory for filtered images
things_images_directory = "..\\..\\things_data\\object_images"
filtered_images_directory = "..\\..\\things_data\\filtered_object_images"

# Create the directory for filtered images if it does not exist
try:
    if not os.path.exists(filtered_images_directory):
        os.makedirs(filtered_images_directory)
except PermissionError:
    print("Error: Permission denied while creating the directory for filtered images.")
    exit(1)

# Iterate over each subfolder in the THINGS dataset image directory
try:
    for class_folder in os.listdir(things_images_directory):
        class_folder_path = os.path.join(things_images_directory, class_folder)
        # Check if the class_folder matches any value in namemap
        for key, value in namemap.items():
            if class_folder == value:  # Match found based on namemap value
                target_folder_name = key  # Use the key from namemap for naming
                destination = os.path.join(filtered_images_directory, target_folder_name)
                try:
                    if not os.path.exists(destination):
                        os.makedirs(destination)
                except PermissionError:
                    print(f"Error: Permission denied while creating the directory {destination}.")
                    continue

                for image_file in os.listdir(class_folder_path):
                    source = os.path.join(class_folder_path, image_file)
                    destination_file = os.path.join(destination, image_file)
                    try:
                        shutil.copy(source, destination_file)
                        print(f"Processed folder: {class_folder} -> {target_folder_name}, Image: {image_file}")
                    except shutil.Error as e:
                        print(f"Error copying {source} to {destination_file}: {e}")
                break  # Stop searching after the first match
except FileNotFoundError:
    print("Error: The specified source directory for THINGS dataset images does not exist.")
except PermissionError:
    print("Error: Permission denied when accessing the THINGS dataset images directory.")

print("Filtering completed.")





