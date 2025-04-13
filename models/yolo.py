import os
import torch
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F

class YOLOFaceDetector(nn.Module):
    def __init__(self, device='cuda', model_path='../model/yolov5s.pt'):
        super(YOLOFaceDetector, self).__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.model_path = os.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist.")
        
        try:
            self.model=torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
        self.model.to(self.device)

        self.image_size = (112, 112)
        self.train_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize(self.image_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.eval_augment = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])



    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, original_frames):
        B, T, C, H, W = original_frames.shape
        original_frames = original_frames.view(B * T, C, H, W)
        device= original_frames.device
        detections_list = self.model(original_frames).pandas().xyxy

        face_frames = []
        augment = self.train_augment if self.training else self.eval_augment         
        for i in range(B*T):
            image = original_frames[i]
            detections = detections_list[i]
            if len(detections) > 0:
                best = detections.loc[detections['confidence'].idxmax()]
                x1, y1, x2, y2 = map(int, [best['xmin'], best['ymin'], best['xmax'], best['ymax']])
                face = image[:, y1:y2, x1:x2]
                face = F.interpolate(face.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False).squeeze(0)
            else:
                face = torch.zeros((3, *self.image_size), device=device, dtype=torch.float32)
            face_frames.append(face)

        face_frames = augment(torch.stack(face_frames))
       
        return  face_frames.view(B, T, 3, *self.image_size).to(device)
    