import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model_concepts import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # data_transform = transforms.Compose([
    #     transforms.Resize((800, 800)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # Load image
    img_path = "../test_images/boot_01b.jpg"
    # img_path = "../test_images/cat_01b.jpg"
    # img_path = "../test_images/coffeemaker_01b.jpg"
    # img_path = "../test_images/truck_05s.jpg"

    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Load concept dictionary
    with open('../../../MAPPING/updated_namemap_verified.json', 'r') as f:
        concept_names = list(json.load(f).keys())

    # Create model and load weights
    model = AlexNet(num_concepts=len(concept_names), init_weights=True).to(device)
    weights_path = "./AlexNet_Concepts.pth"  # Update this with your model path
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Predict concept
    model.eval()
    with torch.no_grad():
        concepts_output, _ = model(img.to(device))
        concepts_output = torch.squeeze(concepts_output).cpu()

        # concept_predict = torch.softmax(concepts_output, dim=0)
        # concept_predict_cla = torch.argmax(concept_predict).numpy()
        # concept_name = concept_names[concept_predict_cla]
        # print("Predicted Concept: {}   Prob: {:.3f}".format(concept_name, concept_predict[concept_predict_cla].numpy()))

        # Get the top 5 predicted concepts with their probabilities
        top_probs, top_classes = torch.topk(torch.softmax(concepts_output, dim=0), 5)
        print("Top 5 Predicted Concepts:")
        for i in range(top_probs.size(0)):
            print(f"{concept_names[top_classes[i]]}: {top_probs[i].item():.4f}")

    plt.show()

if __name__ == '__main__':
    main()
