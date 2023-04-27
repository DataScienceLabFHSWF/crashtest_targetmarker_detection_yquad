import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from ultralytics import YOLO
import os

# parameter_size: n,s,m,l,x
def create_ultralytics_yolov5(num_classes, name=None, checkpoint_path=None, device=None):
    if checkpoint_path is None and name is not None:
        parameters_size = name.split("5")[-1]
        if parameters_size in ["n","s","m","l","x","x6"]:
            if device is None:
                return torch.hub.load('ultralytics/yolov5', f'yolov5{parameters_size}', classes=num_classes)
            else:
                return torch.hub.load('ultralytics/yolov5', f'yolov5{parameters_size}', classes=num_classes, device=device)
        else:
            print(f"Parameter_size ({parameters_size}) for YOLOv5 not valid. Valid parameters_sizes are: 'n','s','m','l','x','x6'")
            return None
    else:
        if device is None:
            return torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(checkpoint_path), force_reload=True)
        else:
            return torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(checkpoint_path), device=device, force_reload=True)
    
# parameter_size: n,s,m,l,x
def create_ultralytics_yolov8(num_classes, name=None, checkpoint_path=None):
    if checkpoint_path is None and name is not None:
        parameters_size = name.split("8")[-1]
        if parameters_size in ["n","s","m","l","x","x6"]:
            return YOLO(f"yolov8{parameters_size}.pt")
        else:
            print(f"Parameter_size ({parameters_size}) for YOLOv8 not valid. Valid parameters_sizes are: 'n','s','m','l','x','x6'")
            return None
    else:        
        return YOLO(os.path.join(checkpoint_path))

def create_faster_rcnn_resnet50_fpn_custom_anchors(num_classes, anchorsizes = [32, 64, 128, 256, 512], scalesizes =[0.5, 1.0, 2.0]):
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', True)
    backbone.out_channels = 256

    # https://stackoverflow.com/questions/56962533/cant-change-the-anchors-in-faster-rcnn#57058964
    anchor_sizes = tuple((e,) for e in anchorsizes)
    aspect_ratios = (tuple(e for e in scalesizes),) * len(anchor_sizes)

    rpn_anchor_generator = AnchorGenerator(
                    anchor_sizes, aspect_ratios
                    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=rpn_anchor_generator,
                    box_roi_pool=roi_pooler)

    return model

def create_faster_rcnn_resnet50_fpn(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def create_faster_rcnn_resnet50_fpn_v2(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def create_faster_rcnn_mobilenet_v3_large_fpn(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_custom_rpn_savestate(savepoint_loc,device):
    savestate = torch.load(savepoint_loc, map_location=device)

    # save file broken due to customization of model
    # Missing key(s) in state_dict: "backbone.fpn.inner_blocks.0.weight", "backbone.fpn.inner_blocks.0.bias", ... etc
    # Unexpected key(s) in state_dict: "backbone.fpn.inner_blocks.0.0.weight", ... etc
    # Missing key(s) in state_dict: "rpn.head.conv.weight", "rpn.head.conv.bias".
    # Unexpected key(s) in state_dict: "rpn.head.conv.0.0.weight", "rpn.head.conv.0.0.bias".
    # in order to fix:

    savestate = {".".join(k.split(".")[:-2])  + "." + k.split(".")[-1] if k.startswith("backbone.fpn.") else k: v for k,v in savestate.items()}
    savestate = {".".join(k.split(".")[:-3])  + "." + k.split(".")[-1] if k.startswith("rpn.head.conv.0.0.") else k: v for k,v in savestate.items()}

    return savestate