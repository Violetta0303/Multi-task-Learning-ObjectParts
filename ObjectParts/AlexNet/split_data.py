import os
from shutil import copy, rmtree
import random

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # If the folder exists, delete the original folder and then recreate it
        rmtree(file_path)
    os.makedirs(file_path)

def main():
    # Ensure randomness is reproducible
    random.seed(0)

    # Allocate 10% of the dataset to the validation set
    split_rate = 0.1

    # Path to your extracted 'things_data' folder
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "..\\..\\things_data")
    origin_things_path = os.path.join(data_root, "filtered_object_images")
    assert os.path.exists(origin_things_path), "path '{}' does not exist.".format(origin_things_path)

    things_class = [cla for cla in os.listdir(origin_things_path)
                    if os.path.isdir(os.path.join(origin_things_path, cla))]

    # Create folder for training dataset
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in things_class:
        # Create a folder for each category
        mk_file(os.path.join(train_root, cla))

    # Create folder for validation dataset
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in things_class:
        # Create a folder for each category
        mk_file(os.path.join(val_root, cla))

    for cla in things_class:
        cla_path = os.path.join(origin_things_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # Randomly sample indices for validation set
        eval_index = random.sample(images, k=int(num * split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # Copy files allocated to the validation set to the appropriate directory
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # Copy files allocated to the training set to the appropriate directory
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # Processing bar
        print()

    print("Processing done!")

if __name__ == '__main__':
    main()
