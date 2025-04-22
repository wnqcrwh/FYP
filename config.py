import torch
import os
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training
        self.batch_size = 32
        self.epochs = 20
        self.lr = 3e-4
        self.weight_decay = 1e-4
        self.num_workers = 8
        self.freeze_epoch = 3
        
        #calculate class weights
        train_counts = torch.tensor([1109, 271, 268, 1743, 4710, 683, 1205], dtype=torch.float32)
        total = torch.sum(train_counts)
        class_weights = total / (train_counts * len(train_counts))
        self.class_weights = class_weights

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
        self.image_size = (128, 128)
        self.frame_rate = 15
        self.sr = 16000
        self.image_augment = True
        self.audio_augment = True
        self.feature_type = 'log_mel'
        self.mode = 'train'
        self.label_names = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']

        # Logging
        self.log_dir = "logs/"
        self.model_save_path = "model/"
        self.best_model_path = os.path.join(self.model_save_path, "best_model.pth")

