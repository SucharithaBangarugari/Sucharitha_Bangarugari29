from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# Define the ResNet-9 class
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.conv5 = conv_block(512, 1028, pool=True)
        self.res3 = nn.Sequential(conv_block(1028, 1028), conv_block(1028, 1028))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(1028, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out


# Initialize Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "resnet9_cifar100_full.pth"
model = torch.load(model_path, map_location=device)
model.eval()

# Define CIFAR-100 normalization
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.486, 0.441), (0.267, 0.256, 0.276)),
    ]
)

# Define label classes (CIFAR-100 class names)
classes = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]



@app.route("/")
def home():
    return "Welcome to the ResNet-9 Prediction Service! Use the /predict endpoint to submit an image and get predictions."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure an image file is provided
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        # Read the image
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Preprocess the image
        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, dim=1)

        # Return the predicted class
        predicted_label = classes[predicted_class.item()]
        return jsonify({"predicted_label": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)

import requests
response = requests.get('https://drive.google.com/uc?id=17Jbd4X4Bx2cNV33Oa1GY80V50Ff2lfNH', allow_redirects=True)
with open('resnet9_cifar100_full.pth', 'wb') as f:
    f.write(response.content)

import os
import requests

# Direct download link
model_url = "https://drive.google.com/uc?id=17Jbd4X4Bx2cNV33Oa1GY80V50Ff2lfNH&export=download"
model_path = "resnet9_cifar100_full.pth"

if not os.path.exists(model_path):
    print("Downloading model...")
    response = requests.get(model_url, stream=True)
    if response.status_code == 200:
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Model downloaded successfully.")
    else:
        print("Failed to download the model. Status code:", response.status_code)
