import torch
from torch.utils.data import Dataset
from torchvision import datasets

class HotDogDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        super(HotDogDataset, self).__init__()
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.coco = datasets.CocoDetection(root=root, annFile=annFile)

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, target = self.coco[idx]

        # Apply transforms to the image
        if self.transform:
            img = self.transform(img)

        # Process target to handle bounding boxes
        boxes = []
        labels = []
        for obj in target:
            if 'bbox' in obj and len(obj['bbox']) == 4:
                x_min, y_min, width, height = obj['bbox']
                if width > 0 and height > 0:
                    x_max = x_min + width
                    y_max = y_min + height
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(obj['category_id'])

       # Convert lists to tensors
       # If no boxes were found, create an empty tensor with appropriate shape
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        processed_target = {
            'boxes': boxes,
            'labels': labels,
        }

        return img, processed_target
