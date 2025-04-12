import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from config import Config
from data import MELD_Dataset
from torch.utils.data import DataLoader
from models.multi_modal_model import MultiModalModel
from utils import evaluate


# Initialize Config
C = Config()

# Initialize Data
train_dataset = MELD_Dataset(
    csv_path=C.train_csv_path,
    video_dir=C.train_video_dir,
    image_size=C.image_size,
    num_frames=C.num_frames,
    sr=C.sr,
    image_augment=C.image_augment,
    audio_augment=C.audio_augment,
    feature_type=C.feature_type,
    mode='train'
)
train_loader = DataLoader(
    train_dataset,
    batch_size=C.batch_size,
    shuffle=True,
    # num_workers=C.num_workers
)
dev_dataset = MELD_Dataset(
    csv_path=C.dev_csv_path,
    video_dir=C.dev_video_dir,
    image_size=C.image_size,
    num_frames=C.num_frames,
    sr=C.sr,
    image_augment=C.image_augment,
    audio_augment=C.audio_augment,
    feature_type=C.feature_type,
    mode='dev'
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=C.batch_size,
    shuffle=False,
    # num_workers=C.num_workers
)

# Initialize Model
model = MultiModalModel(
    num_classes=C.num_classes,
    image_size=C.image_size,
    lstm_layers=C.lstm_layers,
    dropout=C.dropout,
    device=C.device
).to(C.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=C.lr,
    weight_decay=C.weight_decay
)

# Optionally load a pre-trained model
# model.load_state_dict(torch.load(C.pretrained_model_path))
# Optionally load a checkpoint
# model.load_state_dict(torch.load(C.checkpoint_path))
# Optionally load a best model
# model.load_state_dict(torch.load(C.best_model_path))


# Train the model
# Ensure that the optimizer covers all model parameters
assert sum(p.numel() for p in model.parameters()) == \
    sum(p.numel() for g in optimizer.param_groups for p in g['params']), \
    "Optimizer does not cover all model parameters!"

best_val_acc = 0.0
for epoch in range(C.epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Initialize the progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{C.epochs}")

    for batch_idx, batch in enumerate(pbar):
        (original_frames, video_frames, audio_frames), labels = batch
        original_frames = original_frames.to(C.device)
        video_frames = video_frames.to(C.device)
        audio_frames = audio_frames.to(C.device)
        labels = labels.to(C.device)

        optimizer.zero_grad()
        logits = model(original_frames, video_frames, audio_frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": loss.item(), "acc": correct / total})
        print(f"Epoch {epoch+1}/{C.epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {correct / total:.4f}")

    train_loss = total_loss / total
    train_acc = correct / total

    val_loss, val_acc = evaluate(model, dev_loader, criterion, C.device)
    # Log the results
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    log_path = os.path.join(C.log_dir, "train.log")
    os.makedirs(C.log_dir, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%\n")
    # Save the best model
    if val_acc > best_val_acc:
        os.makedirs(C.model_save_path, exist_ok=True)
        best_val_acc = val_acc
        best_model = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        torch.save(best_model, C.best_model_path)
    # Save checkpoints every 5 epochs
    if (epoch + 1) % 5 == 0:
        os.makedirs(C.model_save_path, exist_ok=True)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        torch.save(checkpoint, os.path.join(C.model_save_path, f"checkpoint_epoch_{epoch+1}.pth"))








