import os
import json
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AlexNet

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
    img_path = "../test_images/boot_01b.jpg"
    # img_path = "../test_images/cat_01b.jpg"
    # img_path = "../test_images/coffeemaker_01b.jpg"
    # img_path = "../test_images/truck_05s.jpg"

    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Load concept_to_index dictionary and feature names
    with open('../../../MAPPING/updated_namemap_verified.json', 'r') as f:
        concept_names = list(json.load(f).keys())
    feature_names = pd.read_csv('../../../CSLB/updated_feature_matrix.csv', nrows=0).columns.tolist()[1:]

    # Create model and load weights
    model = AlexNet(num_concepts=len(concept_names), num_features=2725).to(device)
    weights_path = "./AlexNet.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Predict class and features
    model.eval()
    with torch.no_grad():
        concepts_output, features_output, _ = model(img.to(device))
        concepts_output = torch.squeeze(concepts_output).cpu()
        # # Apply sigmoid function to the output features to convert outputs to a range between 0 and 1
        # features_output = torch.sigmoid(features_output)
        features_output = torch.squeeze(features_output).cpu()

        # Get the top 5 predicted concepts with their probabilities
        top_probs, top_classes = torch.topk(torch.softmax(concepts_output, dim=0), 5)
        print("Top 5 Predicted Concepts:")
        for i in range(top_probs.size(0)):
            print(f"{concept_names[top_classes[i]]}: {top_probs[i].item():.4f}")

        # Filter and sort features with values greater than 0.5
        features_list = [(feature_name, feature_value.item()) for feature_name, feature_value in
                         zip(feature_names, features_output)]
        filtered_features = [feat for feat in features_list if feat[1] > 0.3]
        sorted_features = sorted(filtered_features, key=lambda x: x[1], reverse=True)

        # Display sorted and filtered features
        print(f"Features for '{concept_names[top_classes[0]]}' with values > 0.3:")
        for feature_name, feature_value in sorted_features:
            print(f"{feature_name}: {feature_value:.4f}")

    plt.show()

if __name__ == '__main__':
    main()
