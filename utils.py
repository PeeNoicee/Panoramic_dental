# utils.py
import torch

def save_checkpoint(model, optimizer, best_val_IoU, filename):
    """Save model + optimizer state along with best validation IoU"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_IoU': best_val_IoU,  # Add the best IoU to the checkpoint
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer, lr=None):
    """
    Load model + optimizer state and the best validation IoU.
    Optionally reset learning rate to `lr`.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load best_val_IoU from the checkpoint
    best_val_IoU = checkpoint.get('best_val_IoU', float('-inf'))  # Default to -inf if not found

    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return model, optimizer, best_val_IoU

def per_class_iou(logits, targets, class_idx):
    """
    Compute IoU for a single class in multi-class segmentation.
    logits: Tensor[B, C, H, W] (raw scores)
    targets:Tensor[B, H, W] (0..C-1)
    class_idx: int
    """
    preds = torch.argmax(logits, dim=1)
    pred_mask = (preds == class_idx)
    true_mask = (targets == class_idx)
    intersection = (pred_mask & true_mask).sum().float()
    union = (pred_mask | true_mask).sum().float()
    if union == 0:
        return float('nan')
    return (intersection / union).item()
