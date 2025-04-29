import numpy as np
from scipy import ndimage
import random
from torchvision import transforms
from PIL import Image  # Import Image class from PIL
import torch
import torchvision.transforms.functional as F


class SegmentationTransforms:
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def elastic_transform(self, image, mask, alpha=34, sigma=4):
        """
        Apply elastic deformation to both image and mask
        """
        # Convert images to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        random_state = np.random.RandomState(None)

        # Get the shape of the image (height, width)
        height, width = image.shape[:2]

        # Random displacement fields (dx, dy) within the image size
        dx = random_state.uniform(-1, 1, size=(height, width)) * alpha
        dy = random_state.uniform(-1, 1, size=(height, width)) * alpha

        # Apply Gaussian smoothing to the displacement fields
        dx = ndimage.gaussian_filter(dx, sigma, mode="constant", cval=0)
        dy = ndimage.gaussian_filter(dy, sigma, mode="constant", cval=0)

        # Create a grid of coordinates (meshgrid) for the image
        x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")

        # Calculate displaced coordinates
        coords = np.array([y + dy, x + dx])  # Shape: (2, height, width)

        # Apply the elastic transformation to the image
        if image.ndim == 3:  # If the image is RGB/multi-channel
            elastic_image = np.stack(
                [ndimage.map_coordinates(image[..., c], coords, order=1, mode='nearest')
                 for c in range(image.shape[2])],
                axis=-1
            )
        else:  # Grayscale image
            elastic_image = ndimage.map_coordinates(image, coords, order=1, mode='nearest')

        # Apply the elastic transformation to the mask
        elastic_mask = ndimage.map_coordinates(mask, coords, order=1, mode='nearest')

        return elastic_image, elastic_mask

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Convert images and masks to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        # Geometric transforms (same for image & mask)
        if random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        if random.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        # Random rotation
        angle = random.uniform(-30, 30)
        image = ndimage.rotate(image, angle, reshape=False, order=1)
        mask = ndimage.rotate(mask, angle, reshape=False, order=0)  # Nearest-neighbor for masks

        # Apply elastic transformation
        image, mask = self.elastic_transform(image, mask)

        # Random resized crop
        image_pil = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8))

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image_pil, scale=(0.8, 1.0), ratio=(1.0, 1.0)
        )
        image = F.resized_crop(image_pil, i, j, h, w, size=(224, 224), interpolation=Image.BILINEAR)
        mask = F.resized_crop(mask_pil, i, j, h, w, size=(224, 224), interpolation=Image.NEAREST)

        # Color jitter on image only
        image = self.color_jitter(image)

        # Convert to tensor and normalize the image
        image = F.to_tensor(image)
        image = self.normalize(image)
        # Convert to tensor mask (integer labels)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask
