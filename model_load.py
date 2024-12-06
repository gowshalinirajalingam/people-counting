import sys
import os
import subprocess
import shlex

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants




#PEOPLE DETECTION MODEL LOAD (COCO)
def people_model_load():
    weights = '/interplay_v2/public/private/people_counting/checkpoints/yolov4-416'
    framework = 'tf'
    
    saved_model_loaded = ''
    infer = ''
    interpreter = ''
    input_details = ''
    output_details = ''
    
    if framework == 'tflite':
                    interpreter = tf.lite.Interpreter(model_path=weights)
                    interpreter.allocate_tensors()
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    print(input_details)
                    print(output_details)
    # otherwise load standard tensorflow saved model
    else:
                    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
                    infer = saved_model_loaded.signatures['serving_default']        
                    
    people_params = {"infer":infer ,"interpreter":interpreter,  "input_details":input_details, "output_details":output_details}
    return people_params



###HEAD DETECTION IMPORTS
def head_model_load():
    framework='tf'
    #weights_weapon= '/interplay_v2/public/private/weaponresource/checkpoints_weapon/yolo_weapon1711_best_416' 
    input_size= 416
    
    
    config_data1 = '''
#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.CLASSES              = "/interplay_v2/object_detection_yolov4_imports/data/classes/crowdhuman.names"
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/val2017.txt"
__C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/val2017.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5

    '''
         

    with open('/interplay_v2/object_detection_yolov4_imports/core/config.py', 'w+') as f:
            f.write(config_data1)
            f.flush()
            f.close()

    
    
    def convert_darknet_weights_to_tensorflow(input_size):
        
        weights = '/interplay_v2/public/private/people_counting/yolov4-crowdhuman-416x416_best.weights'
        # checkpoint_path = '/interplay_v2/public/private/people_counting/models/yolov4-crowdhuman-416x416_best'
        checkpoint_path = '/interplay_v2/public/private/people_counting/models/head_416'

        
        cmd = '/home/miniconda/bin/python3.9 /interplay_v2/save_model.py --weights '+weights+' \
          --output '+checkpoint_path+' \
          --input_size '+str(input_size)+' \
          --model yolov4' 
          
        print('cmd',cmd)
        sys.stdout.flush()
        if not os.path.exists(checkpoint_path):
            output = subprocess.check_output(shlex.split(cmd))  #it will create the full folder structure.
            print('output',output)
            sys.stdout.flush()
            
        elif len(os.listdir(checkpoint_path)) == 0:
            output = subprocess.check_output(shlex.split(cmd)) 
            print('output',output)
            sys.stdout.flush()


    
        return checkpoint_path
    
    

    
    
    weights= convert_darknet_weights_to_tensorflow(input_size)  #checkpoint path
    
    saved_model_loaded = ''
    infer = ''
    interpreter = ''
    input_details = ''
    output_details = ''
    ########Load model#####
    print("loading model")
    if framework == 'tflite':
          interpreter = tf.lite.Interpreter(model_path=weights)
          interpreter.allocate_tensors()
          input_details = interpreter.get_input_details()
          output_details = interpreter.get_output_details()
          print(input_details)
          print(output_details)    
          print("loaded model")    
    else:
          saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
          infer = saved_model_loaded.signatures['serving_default']
          print("loaded model")
    
    head_params = {"saved_model_loaded":saved_model_loaded, "infer":infer ,"interpreter":interpreter,  "input_details":input_details, "output_details":output_details}
    return head_params


