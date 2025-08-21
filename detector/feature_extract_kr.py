import torch
import torch.nn as nn
from torchvision.models.video import mc3_18

class MC3_FeatureExtractor(nn.Module):
    """
    MC3-18-based feature extractor for video.
    Outputs 512-dim features from final conv layer.
    """

    def __init__(self, pretrained=True):
        super(MC3_FeatureExtractor, self).__init__()
        model = mc3_18(pretrained=pretrained)
        model.fc = nn.Identity()  # Remove final classifier head
        self.backbone = model  # Outputs shape [B, 512]

    def forward(self, x):
        return self.backbone(x)  # [B, 512]

@torch.no_grad()
def get_mc3_feature_extractor(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    model = MC3_FeatureExtractor()
    model.eval()
    model.to(device)
    return model