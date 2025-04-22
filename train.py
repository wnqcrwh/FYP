import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from config import Config
from data.meld_data import MELD_Dataset
from data.utils import collate_fn
from torch.utils.data import DataLoader
from modules.multi_modal_model import MultiModalModel
from utils import evaluate


# Initialize Config
C = Config()

# Initialize Data
train_dataset = MELD_Dataset(
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
train_loader = DataLoader(
    train_dataset,
    batch_size=C.batch_size,
    shuffle=True,
    num_workers=C.num_workers,
    collate_fn=collate_fn
)
dev_dataset = MELD_Dataset(
    csv_path=C.dev_csv_path,
    video_dir=C.dev_video_dir,
    image_size=C.image_size,
    frame_rate=C.frame_rate,
    sr=C.sr,
    image_augment=C.image_augment,
    audio_augment=C.audio_augment,
    feature_type=C.feature_type,
    mode='dev',
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=C.batch_size,
    shuffle=False,
    num_workers=C.num_workers,
    collate_fn=collate_fn
)

# Initialize Model
model = MultiModalModel(
    num_classes=C.num_classes,
    image_size=C.image_size,
    lstm_layers=C.lstm_layers,
    dropout=C.dropout,
    device=C.device
)
if torch.cuda.device_count() > 1:
    model=nn.DataParallel(model)
    model.module.face_extractor.freeze()
else:
    model.face_extractor.freeze()
model.to(C.device)


criterion = nn.CrossEntropyLoss(weight=C.class_weights.to(C.device))
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

def load_checkpoint(model, optimizer, checkpoint_path, device):
    print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loaded_epoch = checkpoint.get('epoch', 0)
    loaded_batch_idx = checkpoint.get('batch_idx', 0)
    total_loss = checkpoint.get('total_loss', 0.0)
    correct = checkpoint.get('correct', 0)
    total = checkpoint.get('total', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    log = checkpoint.get('log', None)

    train_loss = total_loss/total if total != 0 else 0.0
    train_acc = correct/total if total != 0 else 0.0
    print(f"[INFO] Loaded checkpoint: Epoch {loaded_epoch+1}, Batch {loaded_batch_idx+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")
    if log:
        print(f"[INFO] Log: {log}")

    return model, optimizer, loaded_epoch, loaded_batch_idx, total_loss, correct, total, best_val_acc


# Train the model
# Ensure that the optimizer covers all model parameters
assert sum(p.numel() for p in model.parameters()) == \
    sum(p.numel() for g in optimizer.param_groups for p in g['params']), \
    "Optimizer does not cover all model parameters!"

os.makedirs(C.log_dir, exist_ok=True)
os.makedirs(C.model_save_path, exist_ok=True)

checkpoint_path = os.path.join(C.model_save_path, "latest_checkpoint_1.pth")
if os.path.exists(checkpoint_path):
    # 加载检查点
    model, optimizer, loaded_epoch, loaded_batch_idx, total_loss, correct, total, best_val_acc  = load_checkpoint(model, optimizer, checkpoint_path, C.device)

    print(f"[INFO] Reloaded model from Epoch {loaded_epoch+1}, Batch {loaded_batch_idx+1}")
else:
    loaded_epoch, loaded_batch_idx, total_loss, correct, total, best_val_acc = 0, 0, 0.0, 0, 0, 0.0
    print("[INFO] No checkpoint found. Starting training from scratch.")


for epoch in range(loaded_epoch, C.epochs):
    model.train()
    if torch.cuda.device_count() > 1:
        model.module.face_extractor.adjust_threshold(epoch, C.epochs)
    else:
        model.face_extractor.adjust_threshold(epoch, C.epochs)
    if epoch > C.freeze_epoch:
        if torch.cuda.device_count() > 1:
            model.module.face_extractor.unfreeze()
        else:
            model.face_extractor.unfreeze()
        print("[INFO] face_detector unfrozen for fine-tuning.")
    
    if epoch == loaded_epoch:
        print(f"[INFO] Resuming training from Epoch {epoch+1}, Batch {loaded_batch_idx+1}")
        print(f"total_loss: {total_loss:.4f}, correct: {correct}, total: {total}")
    else:
        print(f"[INFO] Starting Epoch {epoch+1}")
        total_loss = 0.0
        correct = 0
        total = 0

    # Initialize the progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{C.epochs}")

    for batch_idx, batch in enumerate(pbar):
        if epoch==loaded_epoch and batch_idx <= loaded_batch_idx:
            continue

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
        if (batch_idx+1) % 100 == 0:
            latest_ckpt = {
                'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'batch_idx': batch_idx,
                'total_loss': total_loss,
                'correct': correct,
                'total': total,
                'best_val_acc': best_val_acc,
                'log': f"Epoch {epoch+1}/{C.epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {total_loss/total:.4f}, Acc: {correct / total:.4f}"
            }
            torch.save(latest_ckpt, os.path.join(C.model_save_path, f"latest_checkpoint_1.pth"))

    train_loss = total_loss / total
    train_acc = correct / total

    val_loss, val_acc, _, _ = evaluate(model, dev_loader, criterion, C.device)
    # Log the results
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    log_path = os.path.join(C.log_dir, "train.log")
    with open(log_path, "a") as f:
        f.write(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%\n")
    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = {
            'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch+1,
            'best_val_acc': best_val_acc,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'log': f"Best Model: Epoch {epoch+1}, Val Acc: {val_acc*100:.2f}%"
        }
        torch.save(best_model, C.best_model_path)
    # Save checkpoints for every epoch
    checkpoint = {
        'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch+1,
        'best_val_acc': best_val_acc,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'log': f"Checkpoint: Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%"
    }
    torch.save(checkpoint, os.path.join(C.model_save_path, f"checkpoint_epoch_{epoch+1}.pth"))


