# Modulo para generar la imagen PARCH a partir de una lista de CJP
import cv2
import numpy as np

# import sys
# sys.path.append('E:/Proyectos/sellos-QA/src/seguimiento_piezas')

def  compose_PARCH_from_CJP(CJP, height, width, width_step = None):
        if sum([len(cl) for cl in CJP.values()]) == 0: # La lista no tiene parches
            return None

        global count, descarte
        # ancho = 3*130
        # alto  = 130
        # height, width = 130, 130 * 3

        # join_image = Image.new('RGB', (ancho,alto), (0, 0, 0))
        join_image = np.zeros((height,width,3), np.uint8)
        
        # if descartar and (len(res) > 1):
        base_col = 0 

        for pred_class in ['colada','junta']:
            for img_obj, _ in CJP[pred_class]:
                join_image[:,
                base_col:base_col + img_obj.shape[1],
                :] = img_obj
                base_col += img_obj.shape[1] if width_step is None else width_step # Si usa el mismo para todas los objetos o no

        return join_image

if __name__ == "__main__":
    import os
    for lote in [""]: #, "101","201"]:
        print()
        print(lote)
        address_from = "E:\\Sellos-Trainning\\211213\\obj_detect\\CJP\\" + lote
        address_to = "E:\\Sellos-Trainning\\211213\\obj_detect\\BORRAR\\" + lote
        # address_from = "E:\\Sellos-Trainning\\obj_detect\\COD_Inf\\PARCH 21.12.13 AnchoLibre 220105_GE_model26_pth\\CyJ\\" + lote
        if not os.path.exists(address_to):
            os.makedirs(address_to)
        address_to = address_to + lote
        if not os.path.exists(address_to):
            os.makedirs(address_to)

        img_list = os.listdir(address_from)
        img_list.sort()

        #me quedo con la raiz del nombre en comun
        img_idx = [idx[0:26] for idx in img_list if True]
        img_idx = list(set(img_idx))
        img_idx.sort()

        print(len(img_list))
        print(len(img_idx))
        count, descarte = 0, 0

    PARCH_height, PARCH_width = 130, 130 * 3

    for files in img_idx:
        print(f"{count:5}/{len(img_idx)}",end="\r")
        address = [os.path.join(address_from, idx) for idx in img_list if idx.startswith(files)]
        CJP =  {'colada': [], 'junta': [], 'protuberancia': [] }

        for img_name in address:
            img_obj = cv2.imread(img_name,1)

            fn_ext  = os.path.splitext(img_name)[1]
            fn_head = os.path.split(img_name)[1][:-len(fn_ext)]

            pred_class = None
            for c in ['colada','junta']:
                if c in fn_head:
                    pred_class = c 
                    break

            if not pred_class is None:
                CJP[pred_class].append((img_obj, len(CJP[pred_class])))

        img_parch = compose_PARCH_from_CJP(CJP,PARCH_height, PARCH_width)
        if not img_parch is None:
            cv2.imwrite(os.path.join(address_to, f"{fn_head[:27]}-PARCH.tiff"), img_parch,)
        count += 1

    print(f"Finalizado. Descarte {descarte}/{len(img_idx)}")