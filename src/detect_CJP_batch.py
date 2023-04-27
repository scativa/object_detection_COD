from mod_CJP import *
import string
import random
import glob as glob
from os import path, makedirs


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODELS_PATH = "\\Sellos-Trainning\\obj_detect\\COD_Train\\models"

model_class = 'fasterrcnn_resnet50_fpn'
model_fn = 'model18-512.pth'
# model_fn = '220105_GE_model26.pth'
# model_fn, model_class = '220105_GE_model26.pth', 'fasterrcnn_mobilenet_v3_large_fpn'

print(f"Device: {device}")
print(f"Model: {model_fn} ({model_class})")
# load the model and the trained weights
model = load_model(path.join(MODELS_PATH, model_fn), model_class, device)



# directory where all the images are present
INIMG_PATH = '/Sellos-Trainning/220117/cam_012'
OUTIMGS_PATH = '/Sellos-Trainning/220117/obj_detect'

lote="???" # "???"
images = glob.glob(path.join(INIMG_PATH,f"*-{lote}-C012.tiff"))
print(f"Cantidad de imagenes: {len(images)}")


serie = f"{time.strftime('%Y%m%d')[2:]}_{''.join(random.choice(string.ascii_uppercase) for i in range(2))}" # Fecha + Random


for sub_fldr in ['',f'{serie}_PARCH',f'{serie}_CJP']:
    if not path.exists(path.join(OUTIMGS_PATH,sub_fldr)):
        makedirs(path.join(OUTIMGS_PATH,sub_fldr))

contador_gral = {'colada': 0, 'junta': 0, 'protuberancia': 0 }
PARCH_height, PARCH_width = sizes['colada'][1], sizes['colada'][0] * 3


for i in range(len(images)):
    # get the image file name for saving output later on
    image_name = images[i].split('/')[-1].split('.')[0].split('\\')[-1]
    image = cv2.imread(images[i])

    print(f"{serie}; {i+1:4d}/{len(images):4d}; {image_name}; ", end="")
    CJP = extract_CJP(image, model, detection_threshold = 0.8)
    CJ = CJP.copy()
    CJ['protuberancia'] = list() # Saco las protuberancias
    img_parch = compose_PARCH_from_CJP(CJ, PARCH_height, PARCH_width)
    if not img_parch is None:
        cv2.imwrite(path.join(OUTIMGS_PATH,f'{serie}_PARCH', f"{image_name}-PARCH.tiff"), img_parch,)
    else:
        print("<--- Sin CJ; ", end="")

    for i, pred_class in enumerate(CLASSES):
        if i > 0: # la primera es background
            for img_obj, contador in CJP[pred_class]:
                fn = path.join(OUTIMGS_PATH,f'{serie}_CJP', f"{image_name}-{pred_class}-{contador:04}.tiff")
                if not cv2.imwrite(fn, img_obj,):
                    print(f"Error: No se puede grabar el archivo {fn}")
                contador_gral[pred_class] += 1

    print(f"{contador_gral}; ")
    
    
print('-'*50)

print(f'Detección CJP finalizada. Serie: {serie}. Salida: {OUTIMGS_PATH}')
