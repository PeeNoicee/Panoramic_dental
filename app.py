from flask import Flask, render_template, request, send_file
import os
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import cv2
import io
from torchvision import transforms
from model import get_model
from utils import get_bounding_boxes_and_centroids

app = Flask(__name__)
UPLOAD_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model()
checkpoint_path = os.path.join('fined_tuned_models', 'impacted_fined_tuned_efficientnet_b1.pth')
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

preprocess = transforms.Compose([
    #transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    original_size = img.size  # (W, H)

    # Resize image to 512x512 for model input
    img_resized = img.resize((512, 512))
    input_tensor = preprocess(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        impacted_mask = (probs[0, 1] > 0.05).float().cpu().numpy()

        # Create overlay mask in 512x512
        impacted_mask = (impacted_mask * 255).astype(np.uint8)
        impacted_mask_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        impacted_mask_rgb[impacted_mask == 255] = [255, 0, 0]

        original_resized_np = np.array(img_resized)
        overlay = cv2.addWeighted(original_resized_np, 1, impacted_mask_rgb, 0.6, 0)

        # Annotate boxes on 512x512 overlay
        boxes, centroids, _ = get_bounding_boxes_and_centroids(impacted_mask)
        annotated = Image.fromarray(overlay)
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()
        for (x, y, w, h), (cx, cy) in zip(boxes, centroids):
            label = f"N: {11 + int(cx * 8 / 512)} D: Impacted"
            draw.rectangle([x, y, x + w, y + h], outline='red', width=2)
            draw.text((x, y - 10), label, fill='red', font=font)

        # Resize final annotated image to half the original size
        half_original_size = (original_size[0] // 2, original_size[1] // 2)
        annotated_half = annotated.resize(half_original_size)

        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
        annotated_half.save(result_path)
        return send_file(result_path, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)
