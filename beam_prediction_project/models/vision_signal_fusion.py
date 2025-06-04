import torch
import torch.nn as nn
import torchvision.models as models

class VisionSignalFusionNet(nn.Module):
    def __init__(self, num_classes=64):
        super(VisionSignalFusionNet, self).__init__()

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # Son katmanı çıkar, 512-d feature elde ederiz

        self.signal_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, power):
        img_feat = self.cnn(image)          
        power_feat = self.signal_mlp(power) 
        fusion = torch.cat((img_feat, power_feat), dim=1)  # (batch, 640)
        output = self.classifier(fusion)
        return output
