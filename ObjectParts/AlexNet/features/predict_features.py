import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from model_features import AlexNet
import json

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # data_transform = transforms.Compose([
    #     transforms.Resize((800, 800)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load image
    # img_path = "../test_images/boot_01b.jpg"
    # img_path = "../test_images/cat_01b.jpg"
    # img_path = "../test_images/coffeemaker_01b.jpg"
    img_path = "../test_images/truck_05s.jpg"

    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # # Load updated_namemap_verified.json
    # with open('updated_namemap_verified.json', 'r') as f:
    #     namemap = json.load(f)

    # The concept_name is the key in updated_namemap_verified.json for the selected image
    # concept_name = "bicycle"  # This should be dynamically determined based on the image or provided externally
    concept_name = "truck"

    # Load feature matrix and select row corresponding to concept_name
    feature_names = pd.read_csv('../../../CSLB/updated_feature_matrix.csv', nrows=0).columns.tolist()[1:]

    # Create model and load weights
    model = AlexNet(num_features=2725).to(device)
    weights_path = "./AlexNet_Features.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Predict features
    model.eval()
    with torch.no_grad():
        features_output, _ = model(img.to(device))
        # # Apply sigmoid function to the output features to convert outputs to a range between 0 and 1
        # features_output = torch.sigmoid(features_output)
        features_output = torch.squeeze(features_output).cpu()

        # Prepare list of features and their names
        features_list = [(feature_name, feature_value.item()) for feature_name, feature_value in
                         zip(feature_names, features_output)]

        # Filter features greater than 0.5 and sort by value descending
        filtered_features = [feat for feat in features_list if feat[1] > 0.5]
        sorted_features = sorted(filtered_features, key=lambda x: x[1], reverse=True)

        # Display sorted features
        print(f"Features for '{concept_name}' with values > 0.5:")
        for feature_name, feature_value in sorted_features:
            print(f"{feature_name}: {feature_value:.4f}")

if __name__ == '__main__':
    main()

