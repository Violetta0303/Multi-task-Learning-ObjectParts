import unittest

import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_concepts=473, num_features=2725, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27] kernel_num=none padding=0
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27] stride=1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            # Dropout: Random inactivation of a fraction of neurons during forward propagation
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.classifier_concepts = nn.Linear(2048, num_concepts)
        self.classifier_features = nn.Linear(2048, num_features)

        if init_weights:
            self._initialize_weights()

    # def forward(self, x):
    #     x = self.features(x)
    #     x = torch.flatten(x, start_dim=1)
    #     x = self.classifier(x)
    #     concepts = self.classifier_concepts(x)
    #     features = self.classifier_features(x)
    #     return concepts, features

    def forward(self, x):
        activations = []
        for layer in self.features:
            x = layer(x)
            activations.append(x)

        # # Apply Attention Module
        # x = self.attention(x)

        # x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        concepts = self.classifier_concepts(x)
        features = self.classifier_features(x)
        return concepts, features, activations

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
class TestAlexNetArchitecture(unittest.TestCase):
    def test_model_initialization(self):
        # Initialize model
        model = AlexNet()
        # Check if the model has the right number of outputs for concepts and features
        self.assertEqual(model.classifier_concepts.out_features, 473, "The output features of the final layer should match the number of concept classes.")
        self.assertEqual(model.classifier_features.out_features, 2725, "The output features of the final layer should match the number of feature labels.")
        # Print the model architecture
        print(model)
