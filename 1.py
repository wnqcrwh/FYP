import data
import torch

dataset = data.MELD_Dataset(
    csv_path="MELD/train.csv",
    video_dir="MELD/train",
)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        
for batch_idx, ((original_frames, frames, audio), emotion_label) in enumerate(loader):
    print(f"Batch {batch_idx + 1}")
    print(f"Original frames shape: {original_frames.shape}")  # [B, num_frames, C, H, W]
    print(f"Frames shape:      {frames.shape}")         # [B, num_frames, C, H, W]
    print(f"Audio shape:      {audio.shape}")          # [B, n_mfcc, T] (e.g. [8, 20, 256])
    print(f"Emotion labels:    {emotion_label}")
   
    break