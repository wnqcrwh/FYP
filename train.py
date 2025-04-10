import data
import mobilefacenet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim as optim
from torch.vision import models, transforms
import torch.nn.functional as F



class MultiModalModel(nn.Module):
    def __init__(self, num_classes=7):
        super(MultiModalModel, self).__init__()
        self.video_model = models.resnet18(pretrained=True)
        self.video_model.fc = nn.Linear(self.video_model.fc.in_features, 512)

        self.audio_model = models.resnet18(pretrained=True)
        self.audio_model.fc = nn.Linear(self.audio_model.fc.in_features, 512)

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, video_input, audio_input):
        video_output = self.video_model(video_input)
        audio_output = self.audio_model(audio_input)
        combined_output = torch.cat((video_output, audio_output), dim=1)
        output = self.fc(combined_output)
        return output