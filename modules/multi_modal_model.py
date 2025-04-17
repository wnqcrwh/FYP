from modules import mobilefacenet
from modules import audio_model
from modules.face_detector import FaceDetector
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class MultiModalModel(nn.Module):
    def __init__(self, num_classes=7, image_size=(112, 112), lstm_layers=2, dropout=0.3, device=None):
        super().__init__()
        self.dropout = dropout
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        self.image_size = image_size       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not device else device
        
        self.face_extractor = FaceDetector(device=self.device)
        
        self.face_model = mobilefacenet.MobileFaceNet(input_size=128) #(B, T, 128)
        self.video_model = mobilefacenet.MobileFaceNet(input_size=128) #(B, T, 128)
        self.audio_model = audio_model.AudioCNN(dropout=self.dropout) #(B, T, 128)

        self.face_lstm = nn.LSTM(input_size = 128, hidden_size = 128, num_layers = self.lstm_layers, batch_first=True, bidirectional=False, dropout=self.dropout)
        self.video_lstm = nn.LSTM(input_size = 128, hidden_size = 128, num_layers = self.lstm_layers, batch_first=True, bidirectional=False, dropout=self.dropout)
        self.audio_lstm = nn.LSTM(input_size = 128, hidden_size = 128, num_layers = self.lstm_layers, batch_first=True, bidirectional=False, dropout=self.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_classes)
        )

    def forward(self,original_frames, video, audio):
        face_frames = self.face_extractor(original_frames.to(device=self.device))
        face_features = self.face_model(face_frames.to(device=self.device))
        video_features = self.video_model(video.to(device=self.device))
        audio_features = self.audio_model(audio.to(device=self.device))

        face_features, _ = self.face_lstm(face_features) 
        video_features, _ = self.video_lstm(video_features)
        audio_features, _ = self.audio_lstm(audio_features)

        face_features = face_features.mean(dim=1)
        video_features = video_features.mean(dim=1)
        audio_features = audio_features.mean(dim=1)
        features = torch.cat((face_features, video_features, audio_features), dim=1) # (B, 384)

        logits = self.classifier(features)
        return logits
    

       




