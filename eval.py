from config import Config
from data.meld_data import MELD_Dataset
from data.utils import collate_fn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from modules.multi_modal_model import MultiModalModel
from utils import evaluate
import os


C=Config()
test_dataset = MELD_Dataset(
    csv_path=C.test_csv_path,
    video_dir=C.test_video_dir,
    image_size=C.image_size,
    frame_rate=C.frame_rate,
    sr=C.sr,
    image_augment=C.image_augment,
    audio_augment=C.audio_augment,
    feature_type=C.feature_type,
    mode='test'
)
test_loader = DataLoader(
    test_dataset,
    batch_size=C.batch_size,
    shuffle=False,
    num_workers=C.num_workers,
    collate_fn=collate_fn
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
    model.module.face_extractor.adjust_threshold(20, 20)
else:
    model.face_extractor.adjust_threshold(20, 20)

checkpoint_path = os.path.join(C.model_save_path, "best_model.pth")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=C.device)
    model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss(weight=C.class_weights.to(C.device))

model.eval()
val_loss, val_acc, true_labels, pred_labels = evaluate(model, test_loader, criterion, C.device)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
# Save the evaluation results
os.makedirs(C.model_save_path, exist_ok=True)
output_path = os.path.join(C.model_save_path, "evaluation_results.txt")
with open(output_path, 'w') as f:
    f.write(f"Validation Loss: {val_loss:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n")

os.makedirs(C.model_save_path, exist_ok=True)
# Save the classification report and confusion matrix
report_dict = classification_report(true_labels, pred_labels, target_names=C.label_names, output_dict=True)
pd.DataFrame(report_dict).transpose().to_csv(os.path.join(C.model_save_path, "classification_report.csv"))

conf_mat = confusion_matrix(true_labels, pred_labels)
pd.DataFrame(conf_mat, index=C.label_names, columns=C.label_names).to_csv(os.path.join(C.model_save_path, "confusion_matrix.csv"))

f1_macro = f1_score(true_labels, pred_labels, average='macro')
f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
f1_micro = f1_score(true_labels, pred_labels, average='micro')
print(f"\nF1 Macro: {f1_macro:.4f}, F1 Weighted: {f1_weighted:.4f}, F1 Micro: {f1_micro:.4f}")
with open(os.path.join(C.model_save_path, "evaluation_results.txt"), 'a') as f:
    f.write(f"F1 Macro: {f1_macro:.4f}\n")
    f.write(f"F1 Weighted: {f1_weighted:.4f}\n")
    f.write(f"F1 Micro: {f1_micro:.4f}\n")