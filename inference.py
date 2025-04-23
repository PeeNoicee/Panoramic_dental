import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from segmentation_models_pytorch import UnetPlusPlus
import os

# --- Config ---
data_dir = 'DENTEX/disease/input'
output_dir = 'DENTEX/disease/results1'
checkpoint_path = 'fined_tuned_models/impacted_fined_tuned_efficientnet_b1.pth'  # Ensure this is the correct B1 checkpoint
image_size = (512, 512)
impact_class_id = 1  # Impacted class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Setup ---
model = UnetPlusPlus(
    encoder_name='efficientnet-b1',   # Change this to 'efficientnet-b1'
    encoder_weights=None,             # Set to None if you're not using pre-trained weights
    in_channels=3,
    classes=5,
    activation=None
)

# Load the checkpoint for EfficientNet-B1
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get('model_state_dict', checkpoint)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard EfficientNet B1 normalization
])

# --- Helpers ---
def get_bounding_boxes_and_centroids(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, centroids = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # Only consider boxes larger than 5x5 pixels
            boxes.append((x, y, w, h))
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2
            centroids.append((cx, cy))
    return boxes, centroids, contours

def estimate_tooth_number(x_center):
    region = int(x_center * 8 / image_size[0])
    return 11 + region


# --- Main Inference ---
img_path = 'DENTEX/training_data/unlabelled/xrays/train_4.png'  # Replace with your image path
original = Image.open(img_path).convert('RGB')
input_tensor = preprocess(original).unsqueeze(0).to(device)

# Inference
output = model(input_tensor)  # [1, 5, H, W]
output_prob = torch.softmax(output, dim=1)  # Get probabilities

# Debugging: Print class probabilities for "Impacted"
impacted_prob = output_prob[0, impact_class_id].detach().cpu().numpy()
print("Class Probabilities for the Impacted Class:")
print(f"Class {impact_class_id} (Impacted): {np.max(impacted_prob)}")

# Visualize the raw probability map for the "Impacted" class
impacted_prob_map = output_prob[0, impact_class_id].detach().cpu().numpy()

plt.imshow(impacted_prob_map, cmap='jet')
plt.title("Probability Map for Impacted Class")
plt.colorbar()
plt.show()

# --- Apply Threshold to Impacted Class ---
threshold = 0.05  # Adjust threshold as needed
impacted_mask = (output_prob[0, impact_class_id] > threshold).float().cpu().numpy()

print("Class Probabilities: ")
for i in range(5):  # Assuming 5 classes
    class_prob = output_prob[0, i].detach().cpu().numpy()
    print(f"Class {i}: {np.max(class_prob)}")

# Resize to match original image size
impacted_mask = (impacted_mask * 255).astype(np.uint8)
impacted_mask = cv2.resize(impacted_mask, image_size)

# --- Apply Mask for Impacted Class ---
colored_impacted_mask = np.zeros((impacted_mask.shape[0], impacted_mask.shape[1], 3), dtype=np.uint8)
colored_impacted_mask[impacted_mask == 255] = [255, 0, 0]  # Red for impacted areas

# Combine with the Original Image
original_resized = np.array(original.resize(image_size))
overlayed_image = cv2.addWeighted(original_resized, 1, colored_impacted_mask, 0.8, 0)  # Apply impacted mask overlay

# --- Annotate with Bounding Boxes and Labels ---
boxes, centroids, contours = get_bounding_boxes_and_centroids(impacted_mask)

# Debugging: Print boxes and centroids
print(f"Boxes: {boxes}")
print(f"Centroids: {centroids}")

annotated = Image.fromarray(overlayed_image)
draw = ImageDraw.Draw(annotated)

# Draw bounding boxes and labels for impacted class
for (x, y, w, h), (cx, cy) in zip(boxes, centroids):
    print(f"Drawing box: {(x, y, w, h)} and centroid: {(cx, cy)}")

    label = f"N: {estimate_tooth_number(cx)}  D: Impacted"
    draw.rectangle([x, y, x + w, y + h], outline='red', width=2)
    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill='blue')

    label_font = ImageFont.truetype("arial.ttf", 16)  # Define the label_font
    draw.text((x, max(y - 20, 0)), label, fill='red', font=label_font)

# --- Save and Show Results ---
overlayed_image = Image.fromarray(overlayed_image)
overlayed_image.save(f"{output_dir}/overlayed_unlabelled.png")

# Show final image with overlay
plt.imshow(annotated)
plt.title("Original Image with Segmented Mask Overlay")
plt.show()

print(f"Impacted Mask shape: {impacted_mask.shape}")
print(f"Impacted Mask sum: {np.sum(impacted_mask)}")
