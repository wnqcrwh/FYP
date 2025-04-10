import data
import mobilefacenet
import audio_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim as optim
from torch.vision import models, transforms
import torch.nn.functional as F

class MultiModalModel(nn.Module):
    def __init__(self, num_classes=(7,3)):
        super().__init__()
        self.video_model = mobilefacenet.MobileFaceNet() #(B, T, 128)
        self.audio_model = audio_model.AudioCNN() #(B, T, 128)
        self.fc = nn.Linear(256 + 512, num_classes)  # Combine video and audio features

    def forward(self, video, audio):
        video_features = self.video_model(video)
        audio_features = self.audio_model(audio)
        combined_features = torch.cat((video_features, audio_features), dim=1)
        output = self.fc(combined_features)
        return output




