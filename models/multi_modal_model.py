import FYP.models.mobilefacenet as mobilefacenet
import FYP.models.audio_model as audio_model
import torch
import torch.nn as nn
from mctnn import MCTNN
import torchvision.transforms as T

class MultiModalModel(nn.Module):
    def __init__(self, num_classes=7, image_size=(112, 112), lstm_layers=2, dropout=0.3):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_extractor = MCTNN(image_size=112, margin=0, keep_all=False, device=self.device)
        self.image_size = image_size
        self.augment = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.face_model = mobilefacenet.MobileFaceNet() #(B, T, 128)
        self.video_model = mobilefacenet.MobileFaceNet() #(B, T, 128)
        self.audio_model = audio_model.AudioCNN() #(B, T, 128)

        self.face_lstm = nn.LSTM(input_size = 128, hidden_size = 128, lstm_layers = lstm_layers, batch_first=True, bidirectional=False, dropout=dropout)
        self.video_lstm = nn.LSTM(input_size = 128, hidden_size = 128, lstm_layers = lstm_layers, batch_first=True, bidirectional=False, dropout=dropout)
        self.audio_lstm = nn.LSTM(input_size = 128, hidden_size = 128, lstm_layers = lstm_layers, batch_first=True, bidirectional=False, dropout=dropout)

        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_classes)
        )
    
    def extract_face_frames(self, original_frames):
        B, T, C, H, W = original_frames.shape
        original_frames = original_frames.view(B * T, C, H, W)
        original_frames = [T.ToPILImage()(img.cpu()) for img in original_frames]
        
        face_frames = self.face_extractor(original_frames)
        face_frames = [
            self.augment(face) if (self.training and face is not None and hasattr(self, 'augment')) 
            else face if face is not None 
            else torch.zeros((3, 112, 112), dtype=torch.float32)
            for face in face_frames
        ]
        face_frames = torch.stack(face_frames).view(B, T, 3, 112, 112).to(self.device)
        return face_frames

    def forward(self,original_frames, video, audio):
        face_frames = self.extract_face_frames(original_frames)
        face_features = self.face_model(face_frames, dropout=self.dropout)
        video_features = self.video_model(video, dropout=self.dropout)
        audio_features = self.audio_model(audio, dropout=self.dropout)

        face_features, _ = self.face_lstm(face_features) 
        video_features, _ = self.video_lstm(video_features)
        audio_features, _ = self.audio_lstm(audio_features)

        face_features = face_features.mean(dim=1)
        video_features = video_features.mean(dim=1)
        audio_features = audio_features.mean(dim=1)
        features = torch.cat((face_features, video_features, audio_features), dim=1) # (B, 384)

        logits = self.classifier(features)
        return logits
    

       




