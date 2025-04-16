import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import torchvision.transforms as T
import numpy as np
from moviepy.editor import VideoFileClip
import cv2
import tempfile
import subprocess
import random
import librosa

class MELD_Dataset(Dataset):
    def __init__(self, csv_path, video_dir, image_size=(112, 112), num_frames=16, sr=16000,
                 image_augment=True, audio_augment=True, feature_type='log_mel', mode='train'):
        self.df = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.image_size = image_size
        self.num_frames = num_frames
        self.sr = sr
        self.image_augment = image_augment
        self.audio_augment = audio_augment
        self.feature_type = feature_type
        self.mode = mode

        if self.image_augment and self.mode == 'train':
            self.image_transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)


    # Extract video frames from video file
    # This function extracts frames from the video and resizes them to the specified image size.
    # It returns a list of original frames and a tensor of shape (num_frames, C, H, W) where C is the number of channels,
    def extract_video_frames(self, video_path):
        clip = VideoFileClip(video_path)
        duration = clip.duration
        frame_list = []
        origin_frame_list = []

        if duration ==0:
            print(f"Warning: Video {video_path} has zero duration.")
            return []
        
        for i in range(self.num_frames):
            t = i * (duration / self.num_frames)
            frame = clip.get_frame(t)
            original = frame.copy()
            original = cv2.resize(original, (224,224))         
            origin_frame_list.append(torch.from_numpy(original).permute(2, 0, 1))
            frame = cv2.resize(frame, self.image_size)
            frame = self.image_transform(frame)
            frame_list.append(frame)
        
        return torch.stack(origin_frame_list), torch.stack(frame_list)
    
    # Extract audio from video file
    # This function uses ffmpeg to extract audio from the video file and convert it to a specified format.
    # It returns a tensor of the audio waveform or a feature representation (MFCC or log-mel spectrogram).
    # The audio is augmented if specified and the mode is 'train'.
    # The audio is also padded or truncated to a specified length.
    # The audio is returned as a tensor of shape (num_chunks, 128) for log-mel spectrogram.
    # The audio is returned as a tensor of shape (sr*2) for waveform.
    # The audio is returned as a tensor of shape (num_chunks, 20) for MFCC.
    def extract_audio(self, video_path):

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            # 提取音频为 wav 格式
            command = [
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1",              # 单声道
                "-ar", str(self.sr),     # 采样率
                "-loglevel", "error",
                tmpfile.name
            ]
            subprocess.run(command, check=True)

            # 加载音频 waveform
            audio, _ = librosa.load(tmpfile.name, sr=self.sr)
            if audio is None or len(audio) == 0 or audio.shape[0] == 0:
                print(f"Warning: Audio extraction failed for {video_path}.")
                return torch.zeros((6, 1, 128, 128), dtype=torch.float32)
            if self.audio_augment and self.mode == 'train':
                audio = self.augment_waveform(audio)      
            if self.feature_type == 'waveform':
                return torch.tensor(audio[:self.sr*2]).float()
            elif self.feature_type == 'mfcc':
                feature = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=20)
                feature = self.pad_or_truncate(feature, 128)
                return torch.tensor(feature).float()
            elif self.feature_type == 'log_mel':
                hop_length = 80
                num_chunks = 6
                target_len = num_chunks * 128
                mel = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128, hop_length=hop_length)
                feature = librosa.power_to_db(mel, ref=np.max)
                feature = self.pad_or_truncate(feature, target_len)
                chunks = [feature[:, i*128:(i+1)*128][np.newaxis, :, :] for i in range(num_chunks)]
                return torch.tensor(np.stack(chunks)).float()
            else:
                raise ValueError("Unsupported feature type. Choose 'waveform', 'mfcc', or 'log_mel'.")

    def pad_or_truncate(self, feature, target_len):
        if feature.shape[1] < target_len:
            pad_width = target_len - feature.shape[1]
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
        else:
            feature = feature[:, :target_len]
        return feature

    def augment_waveform(self, audio):
        audio = audio + np.random.normal(0, 0.005, audio.shape)
        audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.9, 1.1))
        return audio
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        video_path = os.path.join(self.video_dir, filename)

        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} does not exist.")
            return None
        
        # Extract video frames and audio
        origin_frames, frames = self.extract_video_frames(video_path)
        if len(frames) == 0:
            print(f"Warning: No frames extracted from video {video_path}.")
            return None
        audio = self.extract_audio(video_path)
        if audio is None:
            print(f"Warning: No audio extracted from video {video_path}.")
            return None
        
        # Get labels
        emotion_label = torch.tensor(row["Emotion"]).long()

        return (origin_frames, frames, audio), emotion_label


        
