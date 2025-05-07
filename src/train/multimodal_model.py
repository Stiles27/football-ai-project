import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MultimodalCNN, self).__init__()

        # === BRANCHE IMAGE VIDEO (ResNet18) ===
        self.video_cnn = models.resnet18(pretrained=True)
        self.video_cnn.fc = nn.Identity()  # Supprimer la dernière couche

        # === BRANCHE AUDIO (ResNet18 sur spectrogrammes) ===
        self.audio_cnn = models.resnet18(pretrained=True)
        self.audio_cnn.fc = nn.Identity()

        # === COMBINAISON + CLASSEUR FINAL ===
        self.fc = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # 2 classes : "action", "non-action"
        )

    def forward(self, video_img, audio_img):
        v_feat = self.video_cnn(video_img)
        a_feat = self.audio_cnn(audio_img)
        combined = torch.cat((v_feat, a_feat), dim=1)
        output = self.fc(combined)
        return output
