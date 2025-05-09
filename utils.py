import torch

#Evaluate Function
def evaluate(model, dataloader, criterion, device):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            (original_frames, video_frames, audio_frames), labels = batch
            original_frames = original_frames.to(device)
            video_frames = video_frames.to(device)
            audio_frames = audio_frames.to(device)
            labels = labels.to(device)

            logits = model(original_frames, video_frames, audio_frames)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            print(f"Batch Loss: {loss.item()}, Test Acc: {correct / total:.4f}")

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc, all_preds, all_labels