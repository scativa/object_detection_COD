import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes, pretrained = True):
    
    # load Mobilenet_v3 pre-trained model
    print(f"{__name__}::torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained={pretrained})")
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)

    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model
    