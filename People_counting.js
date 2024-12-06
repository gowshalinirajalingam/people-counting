module.exports = function (RED) {
    var spawn = require('child_process').spawn;
    var util = require('util');

    function indentLines(fnCode, depth) {
        return fnCode.split('\n').map((line) => Array(depth).join(' ') + line).join('\n')
    }

    function spawnFn(self) {
        PYTHON_PATH = "/home/miniconda/bin/python3.9";
        //PYTHON_PATH = "/root/miniconda3/envs/rapids-22.06/bin/python";

        self.child = spawn(PYTHON_PATH, ['-uc', self.func.code], {
            stdio: ['pipe', 'pipe', 'pipe', 'ipc']
        });
        self.child.stdout.on('data', function (data) {
            self.log(data.toString());
        });
        self.child.stderr.on('data', function (data) {
            self.error(data.toString());
        });
        self.child.on('close', function (exitCode) {
            if (exitCode) {
                self.error(`Python Function process exited with code ${exitCode}`);
                if (self.func.attempts) {
                    spawnFn(self);
                    self.func.attempts--;
                } else {
                    self.error(`Function '${self.name}' has failed more than 10 times. Fix it and deploy again`)
                    self.status({
                        fill: 'red',
                        shape: 'dot',
                        text: 'Stopped, see debug panel'
                    });
                }
            }
        });
        self.child.on('message', function (response) {
            switch (response.ctx) {
                case 'send':
                    sendResults(self, response.msgid, response.value);
                    break;
                case 'log':
                case 'warn':
                case 'error':
                case 'status':
                    self[response.ctx].apply(self, response.value);
                    break;
                default:
                    throw new Error(`Don't know what to do with ${response.ctx}`);
            }
        });
        self.log(`Python function '${self.name}' running on PID ${self.child.pid}`);
        self.status({
            fill: 'green',
            shape: 'dot',
            text: 'Running'
        });
    }

    function sendResults(self, _msgid, msgs) {
        if (msgs == null) {
            return;
        } else if (!util.isArray(msgs)) {
            msgs = [msgs];
        }
        var msgCount = 0;
        for (var m = 0; m < msgs.length; m++) {
            if (msgs[m]) {
                if (util.isArray(msgs[m])) {
                    for (var n = 0; n < msgs[m].length; n++) {
                        msgs[m][n]._msgid = _msgid;
                        msgCount++;
                    }
                } else {
                    msgs[m]._msgid = _msgid;
                    msgCount++;
                }
            }
        }
        if (msgCount > 0) {
            if (self.req !== undefined){
            msgs[0].req = self.req;
          }
          // Restore RES object if it exists.
          if (self.res !== undefined){
            msgs[0].res = self.res;
          }
            self.send(msgs);
        }
    }

    function PythonFunction(config) {

        var self = this;
        RED.nodes.createNode(self, config);

        //console.log("This is it");
        //console.log(config);
        self.func = {
            code: `
# Age detection, threading, video reading writing, rtsp processing are added            
print("Python code starts")
from platform import python_version
import sys
print('python version:',sys.version)
import os
import json
import re
import pandas as pd
import threading
import queue
import functools
from multiprocessing import Process



#deepsort imports
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from datetime import datetime
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print('physical_devices:',physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import GenderDetectionImports.core.utils as utils
from GenderDetectionImports.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from GenderDetectionImports.core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from GenderDetectionImports.deep_sort import preprocessing, nn_matching
from GenderDetectionImports.deep_sort.detection import Detection
from GenderDetectionImports.deep_sort.tracker import Tracker
from GenderDetectionImports.tools import generate_detections as gdet

from tensorflow.keras.models import load_model


import glob
import shutil
import os
from datetime import datetime
from PIL import Image

import base64
import io

import subprocess
import shlex

channel = None
if sys.version_info[0]<3:
    channel = os.fdopen(3, "r+")
else:
    channel = os.fdopen(3, "r+b", buffering=0)

class Msg(object):
    SEND = 'send'
    LOG = 'log'
    WARN = 'warn'
    ERROR = 'error'
    STATUS = 'status'

    def __init__(self, ctx, value, msgid):
        self.ctx = ctx
        self.value = value
        self.msgid = msgid

    def dumps(self):
        return json.dumps(vars(self)) + "\\n"

    @classmethod
    def loads(cls, json_string):
        return cls(**json.load(json_string))


class Node(object):
    def __init__(self, msgid, channel):
        self.__msgid = msgid
        self.__channel = channel

    def send(self, msg):
        msg = Msg(Msg.SEND, msg, self.__msgid)
        self.send_to_node(msg)

    def log(self, *args):
        msg = Msg(Msg.LOG, args, self.__msgid)
        self.send_to_node(msg)

    def warn(self, *args):
        msg = Msg(Msg.WARN, args, self.__msgid)
        self.send_to_node(msg)

    def error(self, *args):
        msg = Msg(Msg.ERROR, args, self.__msgid)
        self.send_to_node(msg)

    def status(self, *args):
        msg = Msg(Msg.STATUS, args, self.__msgid)
        self.send_to_node(msg)

    def send_to_node(self, msg):
        m = msg.dumps()
        if sys.version_info[0]>2:
            m = m.encode("utf-8")
        self.__channel.write(m)

class RTSVideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) #internal buffer will now store only 1 frames

    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

class thread_with_trace(threading.Thread):
  def __init__(self, *args, **keywords):
    threading.Thread.__init__(self, *args, **keywords)
    self.killed = False
 
  def start(self):
    self.__run_backup = self.run
    self.run = self.__run     
    threading.Thread.start(self)
 
  def __run(self):
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup
 
  def globaltrace(self, frame, event, arg):
    if event == 'call':
      return self.localtrace
    else:
      return None
 
  def localtrace(self, frame, event, arg):
    if self.killed:
      if event == 'line':
        raise SystemExit()
    return self.localtrace
 
  def kill(self):
    self.killed = True
  



# initialize deep sort
print("tracking model loading")
framework = 'tf'
# weights = '/interplay_v2/public/private/people_counting/checkpoints/yolov4-832'
# size = 832
weights = '/interplay_v2/public/private/people_counting/checkpoints/yolov4-416'
size=416
tiny = False
model = 'yolov4'
# output_format = 'XVID' #for mp4
output_format = 'vp80' #for webm

iou=0.45
score = 0.5
dont_show = False
info = False
count = True

max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
model_filename = '/interplay_v2/public/private/people_counting/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
# tracker = Tracker(metric,max_age=20)

# load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

input_size = size

# load tflite model if flag is set
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
                


###HEAD DETECTION IMPORTS
def head_detection_load():
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

__C.YOLO.CLASSES              = "/interplay_v2/node_modules/python-bridge/object_detection_yolov4_imports/data/classes/crowdhuman.names"
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
         

    with open('/interplay_v2/modules/python-bridge/object_detection_yolov4_imports/core/config.py', 'w+') as f:
            f.write(config_data1)
            f.flush()
            f.close()

    
    
    def convert_darknet_weights_to_tensorflow(input_size):
        
        weights = '/interplay_v2/nodes/core/nodesrepositories/Functions/People_counting/yolov4-crowdhuman-416x416_best.weights'
        checkpoint_path = '/interplay_v2/nodes/core/nodesrepositories/Functions/People_counting/models/yolov4-crowdhuman-416x416_best'
        
        cmd = '/home/miniconda/bin/python3.9 /interplay_v2/node_modules/python-bridge/save_model.py --weights '+weights+' \
          --output '+checkpoint_path+' \
          --input_size '+str(input_size)+' \
          --model yolov4' 
          
        print('cmd',cmd)
        sys.stdout.flush()
        if not os.path.exists(checkpoint_path):
            output = subprocess.check_output(shlex.split(cmd))
            print('output',output)
            sys.stdout.flush()
            
        elif len(os.listdir(checkpoint_path)) == 0:
            output = subprocess.check_output(shlex.split(cmd)) 
            print('output',output)
            sys.stdout.flush()


    
        return checkpoint_path
    
    

    
    
    weights= convert_darknet_weights_to_tensorflow(input_size)  #checkpoint path

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
          print("loaded model")
    
    return saved_model_loaded

def inference_images_edited(image1,class_file_path,saved_model_loaded):
    try:
        
        print("inference_images_edited starts")
        ### Should import the below import after updating config.py as config.py uses classes.names file.
        import object_detection_yolov4_imports.core.utils as utils
        from object_detection_yolov4_imports.core.yolov4 import filter_boxes
        from object_detection_yolov4_imports.core.functions import crop_objects 
        from object_detection_yolov4_imports.core.functions import count_objects 

        tiny= False
        model= 'yolov4'
        #output_format = 'XVID'
        output_format = 'vp80' #for webm
        iou= 0.5
        score_n= 0.5
        count= True
        dont_show= False
        info= False
        crop= False
        ocr= False
        plate= False 
        

        image_data = cv2.resize(image1, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()    #detection starting time     
        # print(start_time)    #o/p:1622462456.7511091
        if framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            infer = saved_model_loaded.signatures['serving_default']

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            end_time = time.time()

            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score_n
        )
        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = image1.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]



        # read in all class names from config
        # class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        # Gowshi added

        class_names = utils.read_class_names(class_file_path)
        print('class names edited',class_names)

        
        

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        #print('allowed_classes edited',allowed_classes)


        # custom allowed classes (uncomment line below to allow detections for only people)
        allowed_classes = ['head']

        # if crop flag is enabled, crop each detection and save it as new image
        if crop:
            crop_rate = 150 # capture images every so many frames (ex. crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass          
                crop_objects(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
            else:
                pass
            
            
        '''
        # if count flag is enabled, perform counting of objects
        if count:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image2 = utils.draw_bbox(image1, pred_bbox, info, counted_classes, allowed_classes=allowed_classes, read_plate = plate)
        else:
            image2 = utils.draw_bbox(image1, pred_bbox, info, allowed_classes=allowed_classes, read_plate = plate)


        # print("test",pred_bbox, info)
        '''
        #ADDED custom code start
        out_boxes, out_scores, out_classes, num_boxes = pred_bbox
        classes = allowed_classes  #['head']
        num_classes = len(classes)  #1

        
        #Converting tensor to numpy array
        out_scores = tf.make_ndarray(tf.make_tensor_proto(out_scores) )
        out_classes = tf.make_ndarray(tf.make_tensor_proto(out_classes) )
        out_boxes = tf.make_ndarray(tf.make_tensor_proto(out_boxes) )

        print('out_boxes',out_boxes) 
        print('num_classes',num_classes) 
        print('num_boxes',num_boxes)
        print('out_scores',out_scores) 
        print('out_classes',out_classes) 

        for s,c,coor in zip(out_scores,out_classes,out_boxes):
           if (float(s)>0):
                   if (int(c)==0):
                       detection_class  = 'Head : '+str(s)   #Added score for prediction also
                       print(detection_class)
                       c1, c2 =  (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))  
                       #c1 = (x,y)
                       #c2 = (x+w,y+h)
                       x = int(coor[0])
                       y = int(coor[1])
                       x1 = int(coor[2])
                       y1 = int(coor[3])
                       #cv2.rectangle(image1, (x, y), (x1, y1), (255, 0, 0), 2)
                       roi = image1[y:y1, x:x1]
                       roi = cv2.GaussianBlur(roi, (53,53), 30)
                       image1[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
                       # cv2.imwrite("/interplay_v2/public/private/image1.jpg",image1)    
                       # cv2.imwrite("/interplay_v2/public/private/roi.jpg",roi)    
                   




          #ADDED custom code end



        print("Ended inference")    
        
    except Exception as e:
            print("Process exception occurred:",e)
            pass
        
    return (image1,start_time,end_time,scores,classes)

def python_function(msg):  #msg variable will globally change. when we assign msg['payload']=user_json; the msg value will change to assigned value. can't get video_link from that again. so we use msg_local variable
    print("start python_function")  

    saved_model_loaded_head = head_detection_load()

    # initialize tracker
    tracker = Tracker(metric,max_age=20)
    
    if msg is None:
        print("msg_local none")
    elif msg is not None:
        msgid = msg["_msgid"]
        node = Node(msgid, channel)
        ui=msg["payload"]
        print("msg_local[payload]:",ui)
        
        video_url = ui["video_link"]
        print('video URL:',video_url)

        video_ext =  ui["video_type"]
        print('video Type:',video_ext)
        
        video_friendly_name =  ui["friendly_name"]
        print('Friendly name:',video_friendly_name)

        video_original_name =  ui["file_original_name"]
        print('Original name:',video_original_name)
        
        
        if (video_ext=='rtsp') and ((video_url =="") or (video_friendly_name=="")):  #validation
            return msg
        #######BEGIN INFERENCE

        #video_name1 = video_url.split('/')[-1]   #Oxford_Street_in_London.mp4
        #video_name = video_name1.split('.')[0]   #Oxford_Street_in_London
        video_name1 = video_friendly_name  #if needed add the extension
        video_name = video_friendly_name
        video_original_name = video_original_name.split('.')[0] #without ext
        #For videos, inference and save full video. for RTSP streams save last 2 mins frames
        if (video_ext in ['mp4','webm']):
          video_path = '/interplay_v2/public/private/people_counting/videos/'+video_original_name+'_input.mp4'
        elif (video_ext in ['rtsp','mjpeg']):
          video_path = video_url

        
        print('Begin video capture now')

        if (video_ext in ['mp4','webm']):
            try:
                vid = cv2.VideoCapture(str(video_path))
            except Exception as e:
                print('Video URL is not valid:',e)
                node.warn("Video URL is not valid")
                return msg
        elif (video_ext in ['rtsp','mjpeg']):
            try:
                vid_latest_rtsp = RTSVideoCapture(str(video_path))    #Read the latest frame from rtsp stream
                vid = cv2.VideoCapture(str(video_path))  #to get height, width of the frame to save
            except Exception as e:
                print('Video URL is not valid:',e)
                node.warn("Video URL is not valid")
                return msg
            
            

        import os
        import shutil
        from PIL import Image

        if os.path.exists('/interplay_v2/public/private/people_counting/'+video_name):
                shutil.rmtree('/interplay_v2/public/private/people_counting/'+video_name)  #remove directory even it contains files inside
                print(video_name+' folder exists. So deleted it')

        if not os.path.exists('/interplay_v2/public/private/people_counting/'+video_name):
                os.makedirs('/interplay_v2/public/private/people_counting/'+video_name)
                os.makedirs('/interplay_v2/public/private/people_counting/'+video_name+'/snapshot/')
                print('Created '+video_name+' folder')
                
        


        out = None
        output = '/interplay_v2/public/private/people_counting/'+video_name+'/'+video_name+'.webm'
        output_frame_path = '/interplay_v2/public/private/people_counting/'+video_name+'/snapshot/Live1.jpg'

        if (video_ext in ['mp4','webm']):
            if output:
        
                    # by default VideoCapture returns float instead of int
                    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(vid.get(cv2.CAP_PROP_FPS))
                    codec = cv2.VideoWriter_fourcc(*output_format)
                    # out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
                    out = cv2.VideoWriter(output, codec, fps, (width, height))
        
        elif (video_ext in ['rtsp','mjpeg']):
            if output:
            
                    # by default VideoCapture returns float instead of int
                    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = 1   #fps of stream video will vary. So manually hard coding fps
                    codec = cv2.VideoWriter_fourcc(*output_format)
                    # out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
                    out = cv2.VideoWriter(output, codec, fps, (width, height))
                    
        frame_num = 0
        cnt_false = 0
        countt=0

        # while video is running
        while True:
                print('While loop starts')
                
                if (video_ext in ['mp4','webm']):
                    return_value, frame = vid.read()
                    if return_value:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                    else:
    
                        if frame_num !=0:   # if video format issue frame_num will be 0. if >0 then it is coming after inference is completed
                            #after inference completed do this
                            print("Ended video inference here")
                            json_response = {'video_friendly_name':video_friendly_name,'Input_type':video_ext, 'total_people_count':countt,'Track_id':str(track.track_id),'snapshot':image_path,'output_video_path':output,'end_status':True,'timestamp':datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'output_live_image':output_frame_path}
                      
                            msg['payload'] = json_response
                            node.send(msg)
                            #df_tracking.to_csv('/interplay_v2/public/private/people_counting/'+video_name+'/'+video_name+'_all_frames_trackids.csv',index=False)    
                        else:
                            print('Video has ended or failed, try a different video format!')
    
                        break
                        
                    #if frame_num>5:
                    #  #after inference completed do this
                    #   print("ended inference here")
                    #  df_tracking.to_csv('/interplay_v2/public/private/AgeGenderDetection/flask_video/static/frontend_display/'+video_name+'/'+video_name+'_all_frames_trackids.csv',index=False)    
                       
                    #   break
                        
                elif (video_ext in ['rtsp','mjpeg']):  
                    try:
                        frame = vid_latest_rtsp.read()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                        cnt_false = 0   # if comes inside try after going to exception make count again 0.

                    except Exception as e:
                        #rtsp stream can break due to network issue. so w8 for 1 minute and check return_value is True. Else exit.

                        cnt_false = cnt_false + 1
                        
                        if cnt_false <60:
                                print("wait for 1*60 sec")
                                #time.sleep(1)   # Delay for 1 sec (1 seconds).
                                vid_latest_rtsp = RTSVideoCapture(str(video_path))    #Read the latest frame from rtsp stream
                                continue
                        else:
                                print('Waited for 1*60 sec to get the stream.No stream. So exiting from inference')
                                break

                frame_num +=1
                print('Frame #: ', frame_num)
                
                
                
                #For videos, inference and save full video. for RTSP streams save last 5 mins frames
                if (video_ext in ['rtsp','mjpeg']):
                  if frame_num == (fps*5*60):   #total frames for 5 mins
                      print("2 mins ended and saved video stream")
                      break
                    
                print('Frame Shape #: ', frame.shape)
                frame_size = frame.shape[:2]
                image_data = cv2.resize(frame, (input_size, input_size))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                start_time = time.time()
        
                # run detections on tflite if flag is set
                if framework == 'tflite':
                    interpreter.set_tensor(input_details[0]['index'], image_data)
                    interpreter.invoke()
                    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                    # run detections using yolov3 if flag is set
                    if model == 'yolov3' and tiny == True:
                        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                        input_shape=tf.constant([input_size, input_size]))
                    else:
                        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                        input_shape=tf.constant([input_size, input_size]))
                else:
                    batch_data = tf.constant(image_data)
                    pred_bbox = infer(batch_data)
                    for key, value in pred_bbox.items():
                        boxes = value[:, :, 0:4]
                        pred_conf = value[:, :, 4:]
        
                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=iou,
                    score_threshold=score
                )
        
                # convert data to numpy arrays and slice out unused elements
                num_objects = valid_detections.numpy()[0]
                bboxes = boxes.numpy()[0]
                bboxes = bboxes[0:int(num_objects)]
                scores = scores.numpy()[0]
                scores = scores[0:int(num_objects)]
                classes = classes.numpy()[0]
                classes = classes[0:int(num_objects)]
        
                # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
                original_h, original_w, _ = frame.shape
                bboxes = utils.format_boxes(bboxes, original_h, original_w)
        
                # store all predictions in one parameter for simplicity when calling functions
                pred_bbox = [bboxes, scores, classes, num_objects]
                #print(pred_bbox)
        
                # read in all class names from config
                class_names = utils.read_class_names(cfg.YOLO.CLASSES)
                #print(class_names)
        
                # by default allow all classes in .names file
                #allowed_classes = list(class_names.values())
                allowed_classes =['person']
                # custom allowed classes (uncomment line below to customize tracker for only people)
                #allowed_classes = ['With_mask','Without_mask','Mask_weared_incorrect','Robber_mask']
        
                # loop through objects and use class index to get class name, allow only classes in allowed_classes list
                names = []
                deleted_indx = []
                for i in range(num_objects):
                    class_indx = int(classes[i])
                    class_name = class_names[class_indx]
                    if class_name not in allowed_classes:
                        deleted_indx.append(i)
                    else:
                        names.append(class_name)
                names = np.array(names)
                count = len(names)
        
        
                # if count:
                #     cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 255, 0), 1)
                #     print("Objects being tracked: {}".format(count))
        
        
                # delete detections that are not in allowed_classes
                bboxes = np.delete(bboxes, deleted_indx, axis=0)
                scores = np.delete(scores, deleted_indx, axis=0)
        
                # encode yolo detections and feed to tracker
                features = encoder(frame, bboxes)
                detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        
                #initialize color map
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        
                # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.class_name for d in detections])
                indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]       
        
                # Call the tracker
                tracker.predict()
                tracker.update(detections)
        
                # update tracks

                
                '''
                #arunn--
                for dltrack in tracker.dltracks:
                    print("DL--DL***********************************: ",tracker.trackframes)
                    
                    #Arunn delete tracks
                    try:
                       print(dltrack.track_id)
                       tracker.trackframes.pop(dltrack.track_id)
                    except:
                       print("++++++++++Not available DLLL++++++++++++++++")
                
                '''
                
                #HEAD DETECTION STARTS
                class_file_path = '/interplay_v2/node_modules/python-bridge/object_detection_yolov4_imports/data/classes/crowdhuman.names'  #class file path
                frame,start_time,end_time,scores,classes = inference_images_edited(frame,class_file_path,saved_model_loaded_head) #inference
                 
                #HEAD DETECTION ENDS
                
                result1 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                print("result1 ready")
             
                #print total mesage
                cv2.putText(frame, "Total People Count: {}".format(countt), (5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 255, 0), 1)
                #print('total people count:',countt)
                
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    print(track.track_id)
        
                    #Arunn process track details
                    try:

                       

                       
                       dd = tracker.trackframes[int(track.track_id)]
                       dd[1] +=1
                       
                       #if dd[1]>5 and dd[1]%15 !=0:   # Add skipping(which are not sent to face detection) track_id as well. other thing is if people are not coming more than 5 times those are noises. so ignore them
                       #    df_tracking = df_tracking.append({'Frame_id':frame_num,'track_id':track.track_id,'gender_prob':'NF','gender':'NF','age_range':'NF','age_prob':'NF'},ignore_index = True)

                       
                       
                       if dd[1] >2 and dd[2] != "sent":
                         wide = int(int(bbox[2])-int(bbox[0]))
                         heit = int(int(bbox[3])-int(bbox[1]))
                         
                         
                         
                         y2 = bbox[3]
        
                         if ((bbox[1]-heit*0.1) < 0):
                           y1 = bbox[1]
                         else:
                           y1 = (bbox[1]-heit*0.1)
                          
                         x1 = bbox[0]
        
                        
                         x2 = bbox[2]
                         new_image = result1[int(y1):int(y1+(y2-y1)/2), 
                            int(x1):int(x2)]
                         new_image_full = result1[int(bbox[1]):int(bbox[3]), 
                            int(bbox[0]):int(bbox[2])]
                           
                        
                         #------------------------
                             
                         # draw bbox on screen
                         countt=countt+1
                         #print(countt,"Box : ",bbox)
                         
                             
                         image_path = '/interplay_v2/public/private/people_counting/'+video_name+'/snapshot/'+str(track.track_id)+'_1.jpg'

                         try:
                            
   
                            
                            cv2.imwrite(image_path,new_image_full)

                         except Exception as e:
                            print(e)
                            continue
                        
                         print('Total People Count',countt)

                         json_response = {'video_friendly_name':video_friendly_name,'Input_type':video_ext, 'total_people_count':countt,'Track_id':str(track.track_id),'snapshot':image_path,'output_video_path':output,'end_status':False,'timestamp':datetime.now().strftime("%d/%m/%Y %H:%M:%S"),'output_live_image':output_frame_path}
                   
                         msg['payload'] = json_response
                         node.send(msg)
                        
                
                         dd[2] = "sent"


                         
                   

                         tracker.trackframes[int(track.track_id)] = dd
                         

                       
                       color = colors[int(track.track_id) % len(colors)]
                       color = [i * 255 for i in color]
                        
                       cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                       cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-20)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                       # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*25, int(bbox[1])), color, -1)
                       # cv2.putText(frame, str(track.track_id)+":"+str(dd[2])+":"+str(dd[3]),(int(bbox[0]), int(bbox[1]-6)),0, 0.5, (255,255,255),1)
                       cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1]-6)),0, 0.7, (255,255,255),2)
                       
                       # image_path1 = '/interplay_v2/public/private/people_counting/'+video_name+'/snapshot/'+str(track.track_id)+'_test.jpg'
                       # cv2.imwrite(image_path1,frame)

        
                    except Exception as e:
                       print("++++++++++Not available++++++++++++++++",e)
                       dd = {1:1,2:"NF",3:"NF"}
        
                
                
        
                    
                  

                    # if enable info flag then print details about each track
                    if info:
                        print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        
                
                
               
                            
                

                        

                
                # calculate frames per second of running detections
                fps = 1.0 / (time.time() - start_time)
                print("FPS: %.2f" % fps)
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                
                
                cv2.imwrite(output_frame_path,result)

                
                
                
                
                # if not dont_show:
                #     cv2.imshow("Output Video", result)
                
                # if output flag is set, save video file
                # if FLAGS.output:
                if output:
                    print('Write frame')
                    #cv2.imwrite("outputs/"+str(frame_num)+".jpg",result)
                    cv2.imwrite(output_frame_path,result)
                    out.write(result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cv2.destroyAllWindows()

        
        #######END INFERENCE
        
   
    # InteractiveSession.close()
    session.close()
    return msg





while True:
        print("Reading the Node's input started")
        raw_msg = channel.readline() #Node's Input
        #print('raw_msg',raw_msg)
        if not raw_msg:
            raise RuntimeError('Received EOF!')
        msg = json.loads(raw_msg.decode('utf-8'))
    
        msgid = msg["_msgid"]
        node = Node(msgid, channel)
        
        
        

       
        python_function(msg)

`,
            attempts: 10
        };
           
        spawnFn(self);
self.context().global.set("pythonturiinit",1);
//console.log("get it " +self.i_filename);
        self.on('input', function(msg) {
      // Save REQ object if it exists.
      if (msg.req !== undefined){
        self.req = msg.req;
      }
      // Save RES object if it exists.
      if (msg.res !== undefined){
        self.res = msg.res;
      }
      var cache = [];
      
    //   var t1=config.tuning_paramter1;
    //   msg.t1=parseFloat(t1);
    //   var t2=config.tuning_paramter2;
    //   msg.t2=parseFloat(t2);
    //   var t3=config.tuning_paramter3;
    //   msg.t3=parseFloat(t3);
      jsonMsg = JSON.stringify(msg, function(key, value) {
          if (typeof value === 'object' && value !== null) {
              if (cache.indexOf(value) !== -1) {
                  // Circular reference found, discard key
                  return;
              }
              // Store value in our collection
              cache.push(value);
          }
          return value;
      });
      cache = null; // Enable garbage collection
      self.child.send(JSON.parse(jsonMsg));
    });
    self.on('close', function () {
      self.child.kill();
    });
    }
    RED.nodes.registerType('People_counting', PythonFunction);
};