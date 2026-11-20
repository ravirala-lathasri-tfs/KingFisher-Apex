from pathlib import Path

from PIL import Image
import timm
import torch
from torchvision.transforms.functional import to_pil_image

# Your input image path.
input_path = r"C:/Users/lathasri.ravirala/OneDrive - Thermo Fisher Scientific/KF-Barcode\Dataset_3_classes/average/05102009103.jpg"

# Open the image and print its original size.
img = Image.open(input_path).convert("RGB")
print("Input image size (width, height):", img.size)

# Create the model from your original code.
model = timm.create_model("tf_efficientnet_lite0.in1k", pretrained=True)
model.eval()

# Create the same model-specific transformation.
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

print("\nModel transforms:")
print(transforms)

# Apply the transformation.
transformed_tensor = transforms(img)

# Print the transformed dimensions.
height = transformed_tensor.shape[1]
width = transformed_tensor.shape[2]

print("\nTransformed tensor shape:", tuple(transformed_tensor.shape))
print(f"Transformed image size: {width} × {height} pixels")

# Undo normalization to make the transformed image viewable.
mean = torch.tensor(data_config["mean"]).view(-1, 1, 1)
std = torch.tensor(data_config["std"]).view(-1, 1, 1)

viewable_tensor = (transformed_tensor * std + mean).clamp(0, 1)
output_img = to_pil_image(viewable_tensor)

# Save beside the input image.
output_path = Path(input_path).parent / "transformed_11.png"
output_img.save(output_path)

print("\nSaved transformed image:", output_path)