from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "efficientnet_b0_good_bad.pth"
)

DEVICE = torch.device("cpu")


# ============================================================
# CHECK INPUT IMAGE
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print(r'python Src\predict.py "path_to_image.jpg"')
    sys.exit(1)

IMAGE_PATH = Path(sys.argv[1])

if not IMAGE_PATH.exists():
    print(f"Image not found: {IMAGE_PATH}")
    sys.exit(1)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

class_to_idx = checkpoint["class_to_idx"]

print("Class mapping:", class_to_idx)

# Reverse mapping:
# 0 -> bad
# 1 -> good

idx_to_class = {
    index: class_name
    for class_name, index in class_to_idx.items()
}


# ============================================================
# CREATE EFFICIENTNET-B0
# ============================================================

# We do NOT need pretrained weights here because
# the trained weights are loaded from our checkpoint.

model = efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

# Must match validation/test preprocessing used during training.

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = transform(image)

# Add batch dimension:
# [3,224,224] -> [1,3,224,224]

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()


predicted_class = idx_to_class[predicted_index]

confidence = (
    probabilities[0][predicted_index].item()
    * 100
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

print(f"Image      : {IMAGE_PATH.name}")
print(f"Prediction : {predicted_class.upper()}")
print(f"Confidence : {confidence:.2f}%")

print("\nClass probabilities:")

for index in range(len(idx_to_class)):

    class_name = idx_to_class[index]

    probability = (
        probabilities[0][index].item()
        * 100
    )

    print(
        f"{class_name.upper():5s}: "
        f"{probability:.2f}%"
    )

print("=" * 50)