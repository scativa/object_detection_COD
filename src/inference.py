# Custom Object Detection using PyTorch Faster RCNN
# https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
import numpy as np
import cv2
import torch
import glob as glob
import time
from os import path, makedirs
from model_mobilenet_v3 import create_model

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
OUTIMGS_PATH = path.join('..','test_predictions')
MODELS_PATH = path.join('..','outputs')


print(f"Device: {device}")
# load the model and the trained weights
# model_fn = '220105_GE_model26.pth'
model_fn = '220105_VW_model_62.pth'
print(f"Model: {model_fn}")
model = create_model(num_classes=4).to(device)
model.load_state_dict(torch.load(
    path.join(MODELS_PATH,
    # 'model100.pth', map_location=device
    # 'model16-128.pth'), map_location=device
    # 'model18-512.pth'), map_location=device
    # '220105_GE_model18.pth'), map_location=device
    model_fn), map_location=device
))
model.eval()

# directory where all the images are present
# DIR_TEST = '../test_data'
# DIR_TEST = 'E:/Sellos-Trainning/211124/TrainningLat/110mm 2021.11.24.R2 cam_012 - FULL/Test/101RC'
# DIR_TEST = 'E:/Sellos-Trainning/211217/cam_012'
DIR_TEST = 'E:/Sellos-Trainning/211213/cam_012'
# DIR_TEST = 'E:/Sellos-Trainning/110mm 11.24-12.01 DS/Train/101'
# DIR_TEST = '../selllos_detection/train'


test_images = glob.glob(path.join(DIR_TEST,"*.tiff"))
print(f"Test instances: {len(test_images)}")

# classes: 0 index is reserved for background
CLASSES = [ 'background', 'colada', 'junta', 'protuberancia']
color = {
    'colada':           (0,0,255), 
    'junta':            (0,255,0), 
    'protuberancia':    (255,0,0),
}

sizes = {
    'colada':           (130, 130), 
    'junta':            (50, 130), 
    'protuberancia':    (55, 130),
}

for i, c in enumerate(CLASSES):
    sub_fldr = c if i > 0 else ''
    if not path.exists(path.join(OUTIMGS_PATH,sub_fldr)):
        makedirs(path.join(OUTIMGS_PATH,sub_fldr))


def resize_box(box, clase):
    s = sizes[clase]

    difx = s[0] - (box[2] - box[0])
    newx = box[0] - difx / 2.

    dify = s[1] - (box[3] - box[1])
    newy = box[1] - dify / 2.

    return [
        box[0], newy, ## No toca el tamaño horizontal, si el vertical
        box[2], newy + s[1]
        # newx, newy,
        # newx + s[0], newy + s[1]
    ]



# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.8

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0].split('\\')[-1]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda() # if device == "cuda" else torch.tensor(image, dtype=torch.float).cpu()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        now = round(time.time() * 1000)
        outputs = model(image)
        print(f"model(image) {round(time.time() * 1000) - now}ms; ", end="")
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        contador = {'colada': 0, 'junta': 0, 'protuberancia': 0 }        
        for j, box in enumerate(draw_boxes):
            pred_class = pred_classes[j]
            r_box = resize_box(box,pred_class)
            img_obj = orig_image[ int(r_box[1]):int(r_box[3]), int(r_box[0]):int(r_box[2]), :].copy()

            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color[pred_classes[j]], 2)
            cv2.putText(orig_image, pred_class, 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)

            # cv2.imshow('Prediction', orig_image)
            # cv2.waitKey(1)
            if not img_obj is None and img_obj.shape[0] > 0 and img_obj.shape[1] > 0:
                cv2.imwrite(path.join(OUTIMGS_PATH, f"{image_name}-{pred_class}-{contador[pred_class]:04}.tiff"), img_obj,)
                # cv2.imwrite(path.join(OUTIMGS_PATH, pred_class, f"{image_name}-{pred_class}-{contador[pred_class]:04}.tiff"), img_obj,)
                contador[pred_class] += 1
        outfn = path.join(OUTIMGS_PATH, f"{image_name}.jpg")
        res = cv2.imwrite(outfn, orig_image,)
        # res = True
    # print(f"Image {i+1:4d} done {res}... ({outfn})",end="\r")
    print(f"Image {i+1:4d} {contador} ({outfn})")
    
print('-'*50)

print('TEST PREDICTIONS COMPLETE')
# cv2.destroyAllWindows()