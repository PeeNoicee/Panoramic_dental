# model.py
import segmentation_models_pytorch as smp

def get_model():
    """
    Returns a U-Net++ segmentation model with:
      - 5 output channels (Background, Impacted, Caries, Periapical Lesion, Deep Caries)
      - No built‑in activation (we’ll apply softmax or logits directly in training)
    """
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b1',
        encoder_weights='imagenet',
        in_channels=3,
        classes=5,        # 5 classes (Background, Impacted, Caries, Periapical Lesion, Deep Caries)
        activation=None,  # logits for CrossEntropyLoss
    )
    return model


