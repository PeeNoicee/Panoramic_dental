import torch
from PIL import Image, ImageDraw
import numpy as np
import json
import os
from torch.utils.data import Dataset


class ImpactedToothDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the X-ray images.
            json_dir (str): Directory containing JSON annotations for each X-ray image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.json_dir = json_dir  # Directory with individual JSON files
        self.transform = transform

        # List of all JSON files (each corresponding to one X-ray image)
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        # List of all image metadata (assuming JSON filenames correspond to image filenames)
        self.images = []
        self.image_ids = []
        self.annotations = []

        # Load all JSON files and associate them with the corresponding images
        for json_file in self.json_files:
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_info = data.get('image', {})
            image_id = image_info.get('id', None)
            file_name = image_info.get('file_name', None)
            if image_id and file_name:
                self.images.append(image_info)
                self.image_ids.append(image_id)
                self.annotations.extend(data.get('annotations', []))

        # Extract disease classes from categories_3
        categories = data.get('categories_3', [])
        self.original_classes = {cat['id']: cat['name'] for cat in categories}

        # Build mapping: background = 0, disease classes = 1, 2, 3...
        self.class_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(sorted(self.original_classes.keys()))}
        self.class_mapping_reverse = {v: k for k, v in self.class_mapping.items()}

        # Class names list (e.g., for visualization or evaluation)
        self.class_names = ['Background'] + [self.original_classes[k] for k in sorted(self.original_classes.keys())]

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_id = img_info['id']
        image_name = img_info['file_name']
        image_path = os.path.join(self.image_dir, image_name)

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Get the original image size dynamically (based on the current image)
        original_size = image.size  # (width, height)

        # Resize image to 512x512 for consistency
        image = image.resize((512, 512), Image.LANCZOS)

        # Generate mask for the current image
        mask = self.generate_mask(image_id, original_size)

        # Apply transformations (if any)
        if self.transform:
            mask_pil = Image.fromarray(mask.numpy().astype('uint8'), mode='L')
            transformed = self.transform(image=image, mask=mask_pil)
            if isinstance(transformed, tuple):
                image, mask = transformed
            else:
                image = transformed['image']
                mask = transformed['mask']
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(np.array(mask))

        return image, mask

    def generate_mask(self, image_id, original_size):
        # Initialize the mask with background (0)
        mask = torch.zeros((512, 512), dtype=torch.long)

        # Retrieve all annotations for this image ID
        anns = [a for a in self.annotations if a['image_id'] == image_id]
       # print(f"Processing {len(anns)} annotations for image ID {image_id}")

        for ann in anns:
            # Get category_id_3 (disease class)
            category_id_3 = ann.get('category_id_3')  # Disease type (Impacted, Caries, etc.)

            # Skip if category_id_3 is None (no disease category)
            if category_id_3 is None:
                continue

            # Disease classes: Map category_id_3 to class numbers
            disease_class = category_id_3 + 1  # This will map 0 -> 1, 1 -> 2, etc.

            # Get the segmentation (polygon or RLE)
            seg = ann.get('segmentation', [])
            if not seg or not seg[0]:
                continue

            # Process polygon segmentation
            if isinstance(seg[0], list):
                for polygon in seg:
                    mask = self.add_polygon_to_mask(mask, polygon, disease_class, original_size)

            # Process RLE segmentation
            else:
                height = ann.get('height', 256)
                width = ann.get('width', 256)
                rle = coco_mask.frPyObjects(seg, height, width)
                decoded = coco_mask.decode(rle)

                if len(decoded.shape) == 3:  # Multiple objects in RLE
                    for m in decoded:
                        mask[m == 1] = disease_class  # Assign disease class to mask
                else:
                    mask[decoded == 1] = disease_class  # Assign disease class to mask

        return mask

    def scale_coordinates(self, polygon, original_size, new_size=(512, 512)):
        # Calculate scaling factors for x and y dimensions based on actual image resolution
        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]

        # Scale the coordinates directly from the original points
        scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in polygon]

        # Debug: Print the scaling factors and scaled points to verify correctness
       # print(f"Original size: {original_size}, New size: {new_size}")
       # print(f"Scaling factors: {scale_x}, {scale_y}")
       # print(f"Scaled points: {scaled_points}")

        return scaled_points

    def compute_class_frequencies(self):
        """
        Dynamically compute class frequencies from the annotations.
        """
        frequencies = {class_id: 0 for class_id in self.class_mapping.values()}

        for ann in self.annotations:
            # Get class ID for the annotation
            class_id = ann.get('category_id_3')
            if class_id is not None:
                mapped_class_id = self.class_mapping.get(class_id, 0)  # 0 is background
                if mapped_class_id != 0:
                    frequencies[mapped_class_id] += 1

        return frequencies

    def add_polygon_to_mask(self, mask, polygon, class_id, original_size, new_size=(512, 512)):
        mask_pil = Image.fromarray(mask.numpy().astype('uint8'), mode='L')
        draw = ImageDraw.Draw(mask_pil)

        # Ensure we get a correct list of coordinates
        coords = polygon[0] if isinstance(polygon[0], list) else polygon
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

        # Use the scale_coordinates method to scale the points correctly
        scaled_points = self.scale_coordinates(points, original_size=original_size, new_size=new_size)

        # Apply clamping to ensure points are within the image bounds (0 to 511 for both x and y)
        clamped_points = [
            (
                max(0, min(x, new_size[0] - 1)),  # x-coordinates clamped between 0 and 511
                max(0, min(y, new_size[1] - 1))  # y-coordinates clamped between 0 and 511
            ) for x, y in scaled_points
        ]

        # Draw the polygon on the mask
        draw.polygon(clamped_points, fill=class_id)

        return torch.from_numpy(np.array(mask_pil))
