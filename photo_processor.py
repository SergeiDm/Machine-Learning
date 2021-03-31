# Consists of helper functions for
# background_changer.py

# Import dependencies
from io import BytesIO
from math import exp
from os.path import join
from random import randint
import base64

from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import torch

from _config import PHOTO_PROC_SETT


if PHOTO_PROC_SETT['IS_QUALITY_HIGH']:
    QUALITY = PHOTO_PROC_SETT['QUALITY_HIGH']
else:
    QUALITY = PHOTO_PROC_SETT['QUALITY_NORM']


def get_head_region(ini_height, ini_width, face_box, 
                    coef_height_expansion=[0.55, 1.2], 
                    ratio_width_to_height=2/3,
                    margin=[25, 80]):
    '''
    Calculates actual head region and head region with margin.
    
    Parameters
    ----------
    ini_height: Height of the input image.
    ini_width: Width of the input image.
    face_box: Tuple with coordinates of the box localized detected face.
    coef_height_expansion: Part of height of face_box in which height 
        will be increased.
    ratio_width_to_height: Ratio of the result photo.
    margin: Number of pixels for increasing borders to get cleaner result 
        from neural network.
    
    Returns
    -------
    actual_head_region: Cordinates (x_left, y_upper, x_right, y_bottom) 
        of head region.
    margin_head_region: Cordinates (x_left_margin, y_upper_margin, 
        x_right_margin, y_bottom_margin) of head region with margins.
    '''
    x, y, x2, y2 = face_box
    height = (y2 - y)
    y_upper = int(y - height * coef_height_expansion[0])
    if y_upper < 0:
        y_upper = 0     
    y_bottom = int(y2 + height * coef_height_expansion[1])
    if y_bottom > ini_height:
        y_bottom = ini_height

    width_expansion = (y_bottom - y_upper) * ratio_width_to_height

    x_center = (x + x2) / 2
    x_left = int(x_center - width_expansion / 2)
    if x_left < 0:
        x_left = 0
    x_right = int(x_left + width_expansion) + 1
    if x_right > ini_width:
        x_right = ini_width
    
    # Crop image with front object
    margin_y = int((y_bottom - y_upper) * 0.04)
    if margin_y < margin[0]:
        margin_y = margin[0]
    elif margin_y > margin[1]:
        margin_y = margin[1]
    y_upper_margin = y_upper - int(margin_y / 3)  # default: margin_y
    if y_upper_margin < 0:
        y_upper_margin = 0

    y_bottom_margin = y_bottom + margin_y
    if y_bottom_margin > ini_height:
        y_bottom_margin = ini_height

    margin_x = int((x_right - x_left) * 0.05)
    if margin_x < margin[0]:
        margin_x = margin[0]
    elif margin_x > margin[1]:
        margin_x = margin[1]
    x_left_margin = x_left - margin_x
    if x_left_margin < 0:
        x_left_margin = 0

    x_right_margin = x_right + margin_x
    if x_right_margin > ini_width:
        x_right_margin = ini_width
 
    actual_head_region = (x_left, y_upper, x_right, y_bottom)
    margin_head_region = (
        x_left_margin, y_upper_margin, x_right_margin, y_bottom_margin
    )
    return actual_head_region, margin_head_region


def process_photo(front_object_ini, background_ini,
                  model_face_detector, model_segmentator,
                  device):
    '''
    Change background for the image into given press-wall.

    Parameters
    ----------
    front_object_ini: The image with front object.
    background_ini: The image with background.
    model_face_detector: NN model for face detection.
    model_segmentator: NN model for person segmentation.
    device: 'CUDA' or 'CPU' device.
    
    Returns
    -------
    result_base64: The image where front object is moved to new background. 
        Encoded into Base64.
    error: Error message if there is a problem.
    '''
    photo_size = front_object_ini.shape[0] * front_object_ini.shape[1]
    if photo_size > QUALITY['PHOTO_SIZE_LIMIT']:
        scale_factor = QUALITY['PHOTO_SIZE_LIMIT'] / photo_size
        front_object_ini = cv2.resize(
            front_object_ini,
            dsize=None,
            fx=scale_factor, fy=scale_factor,
            interpolation=cv2.INTER_AREA
        )

    # Get boxes for faces
    try:
        face_boxes_all, _  = model_face_detector.detect(
            front_object_ini, landmarks=False
        )
    except RuntimeError as e:
        return None, e
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if face_boxes_all is not None:
        face_boxes = []
        for box in face_boxes_all:
            if (box[3] - box[1]) > PHOTO_PROC_SETT['SQUARE_THRESH'] \
                and (box[2] - box[0]) > PHOTO_PROC_SETT['SQUARE_THRESH']:
                face_boxes.append(box)
    else:
        return None, 'Не удалось определить лицо на фотографии. \
            Пожалуйста, смените фотографию.'

    l_face_boxes = len(face_boxes)
    if (l_face_boxes > 0) and (l_face_boxes <= PHOTO_PROC_SETT['NUM_PERSON']):
        ini_height = front_object_ini.shape[0]
        ini_width = front_object_ini.shape[1]
        x, y, x2, y2 = face_boxes[0]
        actual_head_region, margin_head_region = get_head_region(
            ini_height=ini_height, 
            ini_width=ini_width,
            face_box=face_boxes[0], 
            coef_height_expansion=PHOTO_PROC_SETT['COEF_HEIGHT_EXPANSION'],
            ratio_width_to_height=PHOTO_PROC_SETT['RATIO_WIDTH_TO_HEIGHT'],
            margin=QUALITY['MARGIN']
        )
        
        x_left, y_upper, x_right, y_bottom = actual_head_region
        x_left_margin, y_upper_margin, x_right_margin, y_bottom_margin \
            = margin_head_region

        # Crop front object according to margin_head_region
        front_object = front_object_ini[
            y_upper_margin:y_bottom_margin, x_left_margin:x_right_margin, :
        ].copy()

        # Limit sie of the object otherwise there may be message 
        # 'RuntimeError: CUDA out of memory. Tried to allocate…'
        #if (device.type == 'cuda') \
        # and (PHOTO_PROC_SETT['SIZE_OF_CUDA_MEMORY'] == 2):
        head_region_size = front_object.shape[0] * front_object.shape[1]
        scale_factor = 1
        if head_region_size > QUALITY['HEAD_REGION_SIZE_LIMIT']:
            scale_factor = QUALITY['HEAD_REGION_SIZE_LIMIT'] / head_region_size
            front_object = cv2.resize(
                front_object, 
                dsize=None, 
                fx=scale_factor,fy=scale_factor, 
                interpolation=cv2.INTER_AREA
            )
        
        # Transform front object for better result neural network
        f_object = front_object.copy()

        # Sharpness correction
        blur = cv2.medianBlur(f_object, 5)
        f_object = cv2.addWeighted(f_object, 2, blur, -1, 0)

        # Contrast stretching
        hsv = cv2.cvtColor(f_object, cv2.COLOR_RGB2HSV)
        min_v = np.min(hsv[:, :, 2])
        max_v = np.max(hsv[:, :, 2])
        s0 = max_v - min_v
        if (s0 < 252) and (s0 != 0):
            s1 = 255 / s0
            s2 = 255 * min_v / s0
            hsv[:, :, 2] = s1 * hsv[:, :, 2] - s2
            f_object = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Preprocess front object image for neural network model
        preprocess_front_object = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        input_tensor = preprocess_front_object(f_object)
        input_batch = input_tensor.unsqueeze(0).to(device)  # create mini-batch

        # Get result from neural network model
        try:
            with torch.no_grad():
                output = model_segmentator.forward(input_batch)
            predictions = torch.argmax(output['out'][0], dim=0) \
                               .byte().cpu().numpy()
        except RuntimeError as e:
            return None, e
        else:
            del input_batch
            del output
        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Build image with a person segment
        r = np.zeros_like(predictions).astype(np.uint8)
        g = np.zeros_like(predictions).astype(np.uint8)
        b = np.zeros_like(predictions).astype(np.uint8)
        idx = predictions == PHOTO_PROC_SETT['PERSON_VALUE']
        r[idx] = 128  # Here and below, 128 is arbitrary value
        g[idx] = 128
        b[idx] = 128
        person_segment = np.stack([r, g, b], axis=2)
  
        # Delete margin from front_object and person_segment images
        fh = front_object.shape[0]
        fw = front_object.shape[1]
        fy1 = int((y_upper-y_upper_margin) * scale_factor)
        fy2 = int(fh - (y_bottom_margin - y_bottom) * scale_factor)
        fx1 = int((x_left-x_left_margin) * scale_factor)
        fx2 = int(fw - (x_right_margin - x_right) * scale_factor)
        front_object = front_object[fy1:fy2, fx1:fx2, :]
        person_segment = person_segment[fy1:fy2, fx1:fx2, :]

        # Transform background
        # Add some randomness
        y_rand = randint(0, int(background_ini.shape[0] * 0.05))
        x_rand = randint(0, int(background_ini.shape[1] * 0.05))
        background_ini = background_ini[y_rand:, x_rand:, :]
        # If size of background is more than front object, shrink background
        # If size of background is less than front object, stretch background
        scale_factor = max(front_object.shape[0] / background_ini.shape[0],
                           front_object.shape[1] / background_ini.shape[1])
        background = cv2.resize(
            background_ini, 
            dsize=None, 
            fx=scale_factor, fy=scale_factor, 
            interpolation=cv2.INTER_AREA
        )
        y_bg_mid = background.shape[0] / 2
        x_bg_mid = background.shape[1] / 2
        y_bg_upper = int(y_bg_mid - front_object.shape[0] / 2)
        x_bg_left = int(x_bg_mid - front_object.shape[1] / 2)
        background = background[y_bg_upper:y_bg_upper + front_object.shape[0], 
                                x_bg_left:x_bg_left + front_object.shape[1], :]

        # Create a binary mask
        # If pixel has a value > 0 its value set to 255
        _, mask = cv2.threshold(person_segment, 0, 255, cv2.THRESH_BINARY)

        # Apply erosion (removing) 
        # and smoothing (blurring) pixels from front_object boundaries
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (7, 7), sigmaX=0, sigmaY=0,
                                borderType=cv2.BORDER_DEFAULT)

        # Normalize mask to 0-1
        mask = mask.astype(np.float32) / 255
                
        # Apply mask to front_object image to select a person
        front_object = cv2.multiply(mask, front_object.astype(np.float32))        

        # Apply anti-mask to background
        background = cv2.multiply(1.0 - mask, background.astype(np.float32))
        
        # Add front_object with background
        result = cv2.add(front_object, background).astype(np.uint8)
        
        # Send result
        result = Image.fromarray(result)
        binary_result = BytesIO()
        result.save(binary_result, format='JPEG', quality=95)
        binary_result.seek(0)
        # Encode result to make the size of the response smaller
        result_base64 = base64.b64encode(binary_result.read())

        del background
        del binary_result
        del front_object
        del mask
        del person_segment
        del result

        return result_base64, None

    elif l_face_boxes > PHOTO_PROC_SETT['NUM_PERSON']:
        return None, 'На фотографии обнаружено несколько человек. \
            Пожалуйста, смените фотографию.'
