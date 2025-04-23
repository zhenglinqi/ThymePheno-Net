import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class GrowthStageAttention(nn.Module):
    def __init__(self, in_channels):
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_weights = self.spatial_attention(x)
        channel_weights = self.channel_attention(x)
        return x * spatial_weights * channel_weights


class ImprovedThymeModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.features = nn.Sequential(*list(backbone.children())[:-2])

        self.attention1 = GrowthStageAttention(256) 
        self.attention2 = GrowthStageAttention(512)
        self.attention3 = GrowthStageAttention(1024)  
        self.attention4 = GrowthStageAttention(2048) 

        self.conv1x1_1 = nn.Conv2d(256, 256, 1)
        self.conv1x1_2 = nn.Conv2d(512, 256, 1)
        self.conv1x1_3 = nn.Conv2d(1024, 256, 1)
        self.conv1x1_4 = nn.Conv2d(2048, 256, 1)

        self.fine_grained = nn.Sequential(
            nn.Conv2d(256 * 4, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [4, 5, 6, 7]:
                features.append(x)

        f1 = self.attention1(features[0])
        f2 = self.attention2(features[1])
        f3 = self.attention3(features[2])
        f4 = self.attention4(features[3])

        f1 = self.conv1x1_1(f1)
        f2 = self.conv1x1_2(f2)
        f3 = self.conv1x1_3(f3)
        f4 = self.conv1x1_4(f4)

        f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)

        fused_features = torch.cat([f1, f2, f3, f4], dim=1)

        fine_features = self.fine_grained(fused_features)

        out = self.classifier(fine_features)
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
