# Custom Object Detection using PyTorch Faster RCNN
# https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
import numpy as np
import cv2
import torch
import time

# from torch.nn.modules.container import ModuleList
from mod_PARCHfromCJP import compose_PARCH_from_CJP

CLASSES = ['background', 'colada', 'junta', 'protuberancia']

def load_model(model_full_path, model_class, device):
    if model_class == "fasterrcnn_resnet50_fpn":
        from model_resnet50 import create_model
    elif model_class == "fasterrcnn_mobilenet_v3_large_fpn":
        from model_mobilenet_v3 import create_model
    else:
        print(f"Error: No existe el modelo clase {model_class}")
        return None
    model = create_model(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_full_path, map_location=device))
    model.eval()

    return model

# classes: 0 index is reserved for background
sizes = {
    'colada':           (130, 130), 
    'junta':            (50, 130), 
    'protuberancia':    (55, 130),
}

def resize_box(box, clase):
    s = sizes[clase]

    difx = s[0] - (box[2] - box[0])
    newx = box[0] - difx / 2.

    dify = s[1] - (box[3] - box[1])
    newy = box[1] - dify / 2.

    return [
        ## No toca el tamaño horizontal, si el vertical
        # box[0], newy,
        # box[2], newy + s[1]

        # Modifica tanto el tamaño vertical como el horizontal
        newx, newy,
        newx + s[0], newy + s[1]
    ]



# define the detection threshold...
# ... any detection having score below this will be discarded

def extract_CJP(image, model, detection_threshold):
    # BGR to RGB
    torch_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    torch_img /= 255.0
    # bring color channels to front
    torch_img = np.transpose(torch_img, (2, 0, 1)).astype(np.float)
    # convert to tensor
    torch_img = torch.tensor(torch_img, dtype=torch.float).cuda() # if device == "cuda" else torch.tensor(image, dtype=torch.float).cpu()
    # add batch dimension
    torch_img = torch.unsqueeze(torch_img, 0)
    with torch.no_grad():
        now = round(time.time() * 1000)
        outputs = model(torch_img)
        print(f"forward {round(time.time() * 1000) - now:4d}ms; ", end="")

    CJP =  {'colada': [], 'junta': [], 'protuberancia': [] }
    
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
        # contador = {'colada': 0, 'junta': 0, 'protuberancia': 0 }
        for j, box in enumerate(draw_boxes):
            pred_class = pred_classes[j]
            r_box = resize_box(box,pred_class)
            img_obj = image[ int(r_box[1]):int(r_box[3]), int(r_box[0]):int(r_box[2]), :].copy()
            if not (img_obj.shape[1], img_obj.shape[0]) == sizes[pred_class]:
                print(f"Error: resize_box fuera de medida '{pred_class}' {img_obj.shape}", end="")
                img_obj = None

            if not img_obj is None and img_obj.shape[0] > 0 and img_obj.shape[1] > 0:
                CJP[pred_class].append((img_obj, len(CJP[pred_class])))
                # cv2.imwrite(path.join(OUTIMGS_PATH, f"{image_name}-{pred_class}-{contador[pred_class]:04}.tiff"), img_obj,)
                # cv2.imwrite(path.join(OUTIMGS_PATH, pred_class, f"{image_name}-{pred_class}-{contador[pred_class]:04}.tiff"), img_obj,)
                # contador[pred_class] += 1
        print(f"{[len(c) for c in CJP.values()]}; ", end="")
        # print(f"Image {i+1:4d}; {list(CJP.keys())}; {[len(c) for c in CJP.values()]}", end="")
        return CJP
###
### VER: "detect_CJP_batch.py"
###
# if __name__ == "__main__":
#     import string
#     import random
#     import glob as glob
#     from os import path, makedirs

    
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     MODELS_PATH = "\\Sellos-Trainning\\obj_detect\\COD_Train\\models"

#     model_class = 'fasterrcnn_resnet50_fpn'
#     model_fn = 'model18-512.pth'
#     # model_fn = '220105_GE_model26.pth'
#     # model_fn, model_class = '220105_GE_model26.pth', 'fasterrcnn_mobilenet_v3_large_fpn'

#     print(f"Device: {device}")
#     print(f"Model: {model_fn} ({model_class})")
#     # load the model and the trained weights
#     model = load_model(path.join(MODELS_PATH, model_fn), model_class, device)



#     # directory where all the images are present
#     OUTIMGS_PATH = 'E:/Sellos-Trainning/211213/obj_detect'
#     INIMG_PATH = 'E:/Sellos-Trainning/211213/cam_012'


#     images = glob.glob(path.join(INIMG_PATH,"*.tiff"))
#     print(f"Cantidad de imagenes: {len(images)}")


#     serie = f"{time.strftime('%Y%m%d')[2:]}_{''.join(random.choice(string.ascii_uppercase) for i in range(2))}" # Fecha + Random


#     for sub_fldr in ['',f'{serie}_PARCH',f'{serie}_CJP']:
#         if not path.exists(path.join(OUTIMGS_PATH,sub_fldr)):
#             makedirs(path.join(OUTIMGS_PATH,sub_fldr))

#     contador_gral = {'colada': 0, 'junta': 0, 'protuberancia': 0 }
#     PARCH_height, PARCH_width = sizes['colada'][1], sizes['colada'][0] * 3


#     for i in range(len(images)):
#         # get the image file name for saving output later on
#         image_name = images[i].split('/')[-1].split('.')[0].split('\\')[-1]
#         image = cv2.imread(images[i])

#         print(f"{serie}; {i+1:4d}/{len(images):4d}; {image_name}; ", end="")
#         CJP = extract_CJP(image, detection_threshold = 0.8)
#         CJ = CJP.copy()
#         CJ['protuberancia'] = list() # Saco las protuberancias
#         img_parch = compose_PARCH_from_CJP(CJ, PARCH_height, PARCH_width)
#         if not img_parch is None:
#             cv2.imwrite(path.join(OUTIMGS_PATH,f'{serie}_PARCH', f"{image_name}-PARCH.tiff"), img_parch,)

#         for i, pred_class in enumerate(CLASSES):
#             if i > 0: # la primera es background
#                 for img_obj, contador in CJP[pred_class]:
#                     fn = path.join(OUTIMGS_PATH,f'{serie}_CJP', f"{image_name}-{pred_class}-{contador:04}.tiff")
#                     if not cv2.imwrite(fn, img_obj,):
#                         print(f"Error: No se puede grabar el archivo {fn}")
#                     contador_gral[pred_class] += 1

#         print(f"{contador_gral}; ")
        
        
#     print('-'*50)

#     print(f'Detección CJP finalizada. Serie: {serie}. Salida: {OUTIMGS_PATH}')
