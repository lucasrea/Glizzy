import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

class GlizzyNet:
    def __init__(self, num_classes=2, freeze_backbone=True, freeze_rpn=True, freeze_fpn=True):
        self.num_classes = num_classes
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self._initialize_model(freeze_backbone, freeze_rpn, freeze_fpn)

    def _initialize_model(self, freeze_backbone, freeze_rpn, freeze_fpn):
        # Load pre-trained Faster R-CNN model with ResNet50 FPN backbone
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        # Modify the number of output classes (including background)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # Freeze parts of the model as specified
        if freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False

        if freeze_rpn:
            for param in model.rpn.parameters():
                param.requires_grad = False

        if freeze_fpn:
            for param in model.backbone.fpn.parameters():
                param.requires_grad = False

        # Freeze parts of the ROI heads, except the final classifier layers
        for name, param in model.roi_heads.named_parameters():
            if "box_predictor" not in name:
                param.requires_grad = False

        return model.to(self.device)

    def get_model(self):
        return self.model
