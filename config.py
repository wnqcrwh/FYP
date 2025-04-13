import torch
import os
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training
        self.batch_size = 16
        self.epochs = 20
        self.lr = 1e-4
        self.weight_decay = 1e-2
        self.num_workers = 8
        
        # Model
        self.num_classes = 7
        self.image_size = (112, 112)
        self.lstm_layers = 2
        self.dropout = 0.3

        # Data
        self.train_csv_path = "MELD/train_cleaned.csv"
        self.train_video_dir = "MELD/train"
        self.dev_csv_path = "MELD/dev_cleaned.csv"
        self.dev_video_dir = "MELD/dev"
        self.test_csv_path = "MELD/test_cleaned.csv"
        self.test_video_dir = "MELD/test"
        self.image_size = (112, 112)
        self.num_frames = 16
        self.sr = 16000
        self.image_augment = True
        self.audio_augment = True
        self.feature_type = 'log_mel'
        self.mode = 'train'

        # Logging
        self.log_dir = "logs/"
        self.model_save_path = "model/"
        self.best_model_path = os.path.join(self.model_save_path, "best_model.pth")

