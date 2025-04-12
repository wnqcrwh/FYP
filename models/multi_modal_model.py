from models import mobilefacenet
from models import audio_model
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms

class MultiModalModel(nn.Module):
    def __init__(self, num_classes=7, image_size=(112, 112), lstm_layers=2, dropout=0.3, device=None):
        super().__init__()
        self.dropout = dropout
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        self.image_size = image_size       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not device else device
        
        self.face_extractor = MTCNN(image_size=112, margin=0, keep_all=False, device=self.device)
        for param in self.face_extractor.parameters():
            param.requires_grad = True
        
        self.train_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.eval_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.face_model = mobilefacenet.MobileFaceNet() #(B, T, 128)
        self.video_model = mobilefacenet.MobileFaceNet() #(B, T, 128)
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
    
    def extract_face_frames(self, original_frames):
        B, T, C, H, W = original_frames.shape
        original_frames = original_frames.view(B * T, C, H, W)
        original_frames = [transforms.ToPILImage()(img.cpu()).resize((640, 360)) for img in original_frames]
        # face_frames = self.face_extractor(original_frames)
        face_frames = []
        augment = self.train_augment if self.training else self.eval_augment         
        for img in original_frames:
            face = self.face_extractor(img)
            if face is not None:
                face = augment(face)
            else:
                face = torch.zeros((3, *self.image_size), dtype=torch.float32)
            face_frames.append(face)
        
        face_frames = torch.stack(face_frames).view(B, T, 3, *self.image_size).to(self.device)
        return face_frames

    def forward(self,original_frames, video, audio):
        face_frames = self.extract_face_frames(original_frames)
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
    

       




