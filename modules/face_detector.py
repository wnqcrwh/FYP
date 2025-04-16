import os
import torch
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import nms
from modules.retinaface import RetinaFace

class PriorBox:
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16], [64], [256]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [
            [int(image_size[0] / step), int(image_size[1] / step)]
            for step in self.steps
        ]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i in range(f[0]):
                for j in range(f[1]):
                    for min_size in min_sizes:
                        s_kx = min_size / self.image_size[1]
                        s_ky = min_size / self.image_size[0]
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.tensor(anchors, dtype=torch.float32).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    ), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

class FaceDetector(nn.Module):
    def __init__(self, device='cuda', backbone='mobilenet0.25'):
        super(FaceDetector, self).__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = RetinaFace(backbone=backbone).to(self.device)
        priorbox = PriorBox(image_size=(224, 224))
        self.priors = priorbox.forward().to(self.device)

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
        for name, param in self.model.named_parameters():
            if name.startswith(('fpn', 'ssh', 'ClassHead', 'BboxHead', 'LandmarkHead')):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def adjust_threshold(self, epoch, max_epoch):
        self.conf_thresh = 0.05 + 0.4 * (epoch / max_epoch)
        self.nms_thresh = 0.6 - 0.2 * (epoch / max_epoch)


    def detect(self, images, conf_thresh=0.5, nms_thresh=0.4):
        """
        Args:
            images: (B, 3, H, W) tensor on GPU
        Returns:
            detections: list of length B, each is a list of {bbox, conf}
        """
        images = images.float() / 255.0
        B = images.size(0)
        detections = []
        priors = self.priors.to(images.device)

        locs, confs, _ = self.model(images)

        variances = [0.1, 0.2]
        for b in range(B):
            loc = locs[b]
            conf = F.softmax(confs[b], dim=-1)
            scores = conf[:, 1] 

            inds = torch.where(scores > conf_thresh)[0]
            if inds.numel() == 0:
                detections.append([])
                continue
           
            boxes = decode(loc[inds], priors[inds], variances)
            scores = scores[inds]

            keep = nms(boxes, scores, nms_thresh)
            boxes = boxes[keep]
            scores = scores[keep]

            final = torch.cat([boxes, scores.unsqueeze(1)], dim=1)
            detections.append(final)

        return detections

    def forward(self, original_frames):
        B, T, C, H, W = original_frames.shape
        original_frames = original_frames.view(B * T, C, H, W)
        device= original_frames.device
        
        face_frames = []
        augment = self.train_augment if self.training else self.eval_augment         
        detections_list = self.detect(original_frames, conf_thresh=self.conf_thresh, nms_thresh=self.nms_thresh)
        for i in range(B*T):
            image = original_frames[i]
            detections = detections_list[i]
            if len(detections) > 0:
                best = max(detections, key=lambda det: det[4]) 
                x1, y1, x2, y2 = best[:4].int().tolist()    
                x1 = max(0, min(x1, image.shape[2] - 1))
                x2 = max(0, min(x2, image.shape[2] - 1))
                y1 = max(0, min(y1, image.shape[1] - 1))
                y2 = max(0, min(y2, image.shape[1] - 1))
                if x2 > x1 and y2 > y1:
                    face = image[:, y1:y2, x1:x2]
                    face = face.float() / 255.0
                    face = F.interpolate(face.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False).squeeze(0)
                else:
                    face = torch.zeros((3, *self.image_size), device=device, dtype=torch.float32)
            else:
                face = torch.zeros((3, *self.image_size), device=device, dtype=torch.float32)
            face_frames.append(face)

        face_frames = augment(torch.stack(face_frames).to(device))
       
        return  face_frames.view(B, T, 3, *self.image_size)


    