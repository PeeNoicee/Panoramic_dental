import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex as IoU
from tqdm import tqdm
import logging
from model import get_model
from augmentation import SegmentationTransforms
from utils import save_checkpoint
from dataset import ImpactedToothDataset

# 1. Reproducibility: set random seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 2. Set up logging
logging.basicConfig(filename='training_log3txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Log the start of the training
logging.info('Training started')

# Device configuration (CPU-only)
device = torch.device("cpu")

# Hyperparameters
batch_size = 8
num_epochs = 200
learning_rate = 1e-2
patience_early_stop = 25  # Early stopping patience
patience_scheduler = 0

# Data augmentation / preprocessing
transform = SegmentationTransforms()

# Datasets and loaders
train_dataset = ImpactedToothDataset(
    image_dir="DENTEX/training_data/quadrant-enumeration-disease/xrays",
    json_dir="DENTEX/training_data/quadrant-enumeration-disease/split_jsons",
    transform=transform
)
val_dataset = ImpactedToothDataset(
    image_dir="DENTEX/validation_data/quadrant_enumeration_disease/xrays",
    json_dir="DENTEX/validation_data/quadrant_enumeration_disease/split_jsons",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Track loss and IoU metrics for plotting
train_losses = []
train_ious = []
val_losses = []
val_ious = []

# Model initialization
model = get_model().to(device)

# Load pre-trained model for fine-tuning (if available)
checkpoint = torch.load('trained_models/impacted_trained_efficientnet-b1.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Load the best IoU value from the checkpoint
best_val_IoU = checkpoint.get('best_val_IoU', float('-inf'))

# Optionally freeze/unfreeze layers for fine-tuning
for param in model.parameters():
    param.requires_grad = False

for param in model.decoder.parameters():
    param.requires_grad = True

for param in model.segmentation_head.parameters():
    param.requires_grad = True


# Initialize CrossEntropyLoss with static class weights
class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1, reduction='mean'):
        super(CrossEntropyDiceLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def dice_loss(self, pred, target, smooth=1.0):
        num_classes = pred.size(1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        pred = pred.reshape(-1)
        target_one_hot = target_one_hot.reshape(-1)

        intersection = torch.sum(pred * target_one_hot)
        dice_score = (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target_one_hot) + smooth)
        return 1 - dice_score

    def forward(self, outputs, targets):
        ce_loss = self.cross_entropy(outputs, targets)
        outputs_softmax = torch.softmax(outputs, dim=1)
        dice_loss_val = self.dice_loss(outputs_softmax, targets)
        return ce_loss + dice_loss_val


# Instantiate the combined loss function
criterion = CrossEntropyDiceLoss()

# Optimizer & Scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-9, verbose=True)

# Metric: multi-class IoU
iou_metric = IoU(num_classes=5)

# Early stopping vars
best_val_IoU = float('-inf')
best_val_loss = float('inf')
patience_counter = 0
patience_threshold = 5

# Record epoch times
epoch_times = []

# Fine-tuning loop
for epoch in range(1, num_epochs + 1):
    start = time.time()
    model.train()
    train_loss = 0.0
    train_iou = 0.0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{num_epochs}", ncols=100)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # [B, 5, H, W]

        # Static class weights setup
        static_class_weights = torch.tensor([1.0, 5.0, 1.0, 1.0, 1.0]).to(device)
        criterion = CrossEntropyDiceLoss(weight=static_class_weights)

        # Compute loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        iou = iou_metric(outputs.softmax(dim=1), masks)
        train_iou += iou.item()

        pbar.set_postfix({'loss': f'{train_loss / (pbar.n + 1):.4f}', 'iou': f'{train_iou / (pbar.n + 1):.4f}'})

    avg_train_loss = train_loss / len(train_loader)
    avg_train_iou = train_iou / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validate", ncols=100)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            iou = iou_metric(outputs.softmax(dim=1), masks)
            val_iou += iou.item()
            pbar.set_postfix({'loss': f'{val_loss / (pbar.n + 1):.4f}', 'iou': f'{val_iou / (pbar.n + 1):.4f}'})

    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")
    print(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

    # Save metrics for plotting
    val_losses.append(avg_val_loss)
    val_ious.append(avg_val_iou)

    # Log metrics after each epoch
    logging.info(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")
    logging.info(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

    # Save the checkpoint with the best IoU when it improves
    if avg_val_iou > best_val_IoU:
        best_val_IoU = avg_val_iou
        patience_counter = 0  # Reset patience if IoU improves
        save_checkpoint(model, optimizer, best_val_IoU, 'fined_tuned_models/impacted_fined_tuned_efficientnet_b1.pth')
        print("Validation IoU improved, model saved.")
        patience_scheduler = 0  # Reset patience_scheduler after Validation IoU improved
        logging.info("Validation IoU improved, model saved.")

    # If neither IoU nor loss improved, increment patience counter
    else:
        patience_counter += 1
        patience_scheduler += 1

    learning_rate_before = optimizer.param_groups[0]['lr']
    print(f"Before scheduler step - Learning rate: {learning_rate_before:.9f}")
    logging.info(f"Before scheduler step - Learning rate: {learning_rate_before:.9f}")
    scheduler.step(avg_val_iou)  # Reduce learning rate if it is triggered n times based on the patience value


    # After IoU stagnation, the learning rate will be reduced
    if patience_scheduler >= patience_threshold:
            print("Learning rate reduced due to stagnation.")
            # Log the updated learning rate after reduction

            # Unfreeze more layers for further fine-tuning
            print("Unfreezing more layers for further fine-tuning.")
            for param in model.encoder.parameters():  # Unfreeze the encoder
                param.requires_grad = True
            for param in model.segmentation_head.parameters():  # Unfreeze the segmentation head
                param.requires_grad = True

            learning_rate_after = optimizer.param_groups[0]['lr']
            print(f"After scheduler step - Updated learning rate: {learning_rate_after:.9f}")
            logging.info(f"After scheduler step - Updated learning rate: {learning_rate_after:.9f}")

            patience_scheduler = 0  # Reset patience_scheduler after taking action

    # Early stopping condition: stop training if patience exceeds limit
    if patience_counter >= patience_early_stop:
        print("Early stopping activated.")
        logging.info("Early stopping activated.")
        break  # Stop the training loop if no improvement for 'patience' epochs

    # Epoch timing
    epoch_time = time.time() - start
    epoch_times.append(epoch_time)
    avg_time = sum(epoch_times) / len(epoch_times)
    print(f"Average Epoch Time: {avg_time:.2f}s")

# Log the end of training
logging.info("Training finished")

# Log the highest validation IoU after training finishes
print(f"Highest Validation IoU: {best_val_IoU:.4f}")
logging.info(f"Highest Validation IoU: {best_val_IoU:.4f}")

# At the end of each epoch, append the average training loss
train_losses.append(avg_train_loss)  # Add this line to track train losses

# Validation loss and IoU should also be tracked
val_losses.append(avg_val_loss)  # Add this line to track validation losses
val_ious.append(avg_val_iou)     # Add this line to track validation IoU

# After the training loop, plot the losses and IoU values
plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Plot training and validation IoU
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_ious, label='Training IoU', color='blue')
plt.plot(range(1, num_epochs + 1), val_ious, label='Validation IoU', color='red')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.title('IoU Curve')
plt.legend()

plt.tight_layout()
plt.show()