import torch
import torch.nn as nn
from config import Config
from utils import evaluate
from modules.multi_modal_model import MultiModalModel
import os
from config import Config
from data.meld_data import MELD_Dataset
from data.utils import collate_fn
from torch.utils.data import DataLoader
from torchvision import transforms

C=Config()

dev_dataset = MELD_Dataset(
    csv_path=C.dev_csv_path,
    video_dir=C.dev_video_dir,
    image_size=C.image_size,
    num_frames=C.num_frames,
    sr=C.sr,
    image_augment=C.image_augment,
    audio_augment=C.audio_augment,
    feature_type=C.feature_type,
    collate_fn=collate_fn,
    mode='dev'
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=C.batch_size,
    shuffle=False,
    num_workers=C.num_workers
)

model = MultiModalModel(
    num_classes=C.num_classes,
    image_size=C.image_size,
    lstm_layers=C.lstm_layers,
    dropout=C.dropout,
    device=C.device
)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(C.device)

if torch.cuda.device_count() > 1:
    model.module.face_extractor.adjust_threshold(5, 20)
else:
    model.face_extractor.adjust_threshold(5, 20)

checkpoint_path = os.path.join(C.model_save_path, "best_model.pth")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=C.device)
    print(checkpoint.keys())
    print(checkpoint.values())
    exit()
    model.load_state_dict(checkpoint['model_state_dict'])


def save_batch_images(tensor_batch, save_dir="test", prefix="face", denormalize=False):
    os.makedirs(save_dir, exist_ok=True)
    
    to_pil = transforms.ToPILImage()
    if tensor_batch.dim() == 5:
        tensor_batch = tensor_batch.view(-1, *tensor_batch.shape[2:])  # (B*T, C, H, W)

    for i in range(tensor_batch.size(0)):
        print(tensor_batch[i].mean(), tensor_batch[i].std())
    
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor_batch.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor_batch.device)
        tensor_batch = tensor_batch * std + mean
        tensor_batch = torch.clamp(tensor_batch, 0, 1)
    
    tensor_batch = (tensor_batch * 255).clamp(0, 255).byte()
    
    for i, img_tensor in enumerate(tensor_batch):
        img = to_pil(img_tensor.cpu())
        img.save(os.path.join(save_dir, f"{prefix}_{i}.png"))

    print(f"✅ Saved {len(tensor_batch)} images to {save_dir}/ with prefix '{prefix}_'")

model.eval()
with torch.no_grad():
    for batch in dev_loader:
        if torch.cuda.device_count() > 1:
            face_input = batch[0][0].to(C.device)
            output = model.module.face_extractor(face_input)
        else:
            face_input = batch[0][0].to(C.device)
            output = model.face_extractor(face_input)

        print("Face extractor output shape:", output.shape)
        save_batch_images(output, save_dir="test", prefix="face", denormalize=True)
        break 

