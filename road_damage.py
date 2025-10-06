import torch  # used for deep learning framework tasks like loading models and making predictions.
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define full class names
CLASS_NAMES = {
    "D00": "Linear crack",
    "D10": "Alligator crack",
    "D20": "Pothole",
    "D40": "Rutting",
    "D43": "Crosswalk blur",
    "D44": "White line blur",
    "D50": "Patchwork"
}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("C:/Users/Administrator/Downloads/efficientnet_road_damage.pth", map_location=device))
model.to(device)
model.eval()

# Define transformations
transform = A.Compose([
    A.Resize(224, 224),#pixels 
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Function to predict an image
def predict(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform the image
    transformed = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed)
        _, predicted = output.max(1)

    predicted_label = list(CLASS_NAMES.keys())[predicted.item()]
    predicted_class = CLASS_NAMES[predicted_label]

    # Display results
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class} ({predicted_label})")
    plt.axis("off")
    plt.show()

# Test on a local image
image_path = "C:/Users/Administrator/Pictures/road_damage_dataset/train/damaged/Screenshot 2025-03-20 220314.png"
predict(image_path)