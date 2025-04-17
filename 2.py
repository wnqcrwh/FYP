from data.meld_data import MELD_Dataset
from torch.utils.data import DataLoader
from data.utils import collate_fn
from config import Config

C=Config()
dataset= MELD_Dataset(
    csv_path=C.train_csv_path,
    video_dir=C.train_video_dir,
    image_size=C.image_size,
    frame_rate=C.frame_rate,
    sr=C.sr,
    image_augment=C.image_augment,
    audio_augment=C.audio_augment,
    feature_type=C.feature_type,
    mode='train',
)

dataloader = DataLoader(
    dataset,
    batch_size=C.batch_size,
    shuffle=True,
    num_workers=C.num_workers,
    collate_fn=collate_fn
)

count = 0
for (origin, video, audio), labels in dataloader:
    print("origin shape:", origin.shape)
    print("video shape:", video.shape)
    print("audio shape:", audio.shape)
    print("labels shape:", labels.shape)
    if count>3:
        break
    count += 1
    


