module.exports = function (RED) {
    var spawn = require('child_process').spawn;
    var util = require('util');

    function indentLines(fnCode, depth) {
        return fnCode.split('\n').map((line) => Array(depth).join(' ') + line).join('\n')
    }

    function spawnFn(self) {
        PYTHON_PATH = "/usr/bin/python3.8";
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
            if (self.req !== undefined) {
                msgs[0].req = self.req;
            }
            // Restore RES object if it exists.
            if (self.res !== undefined) {
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
#run only one video at a time. stop processing video when press stop and save the video which is processed until
from platform import python_version
import multiprocessing
from multiprocessing import Pool, cpu_count
import sys
import os
import json
import re
import pandas as pd
from datetime import datetime
import queue
import functools
import sys
print('python version:',sys.version)

import trace
import threading
import time

import base64
import io
from PIL import Image
import GPUtil

print("system path", sys.path)
print("Weapon library imported in the main script")
from multiprocessing import set_start_method
set_start_method("spawn", force=True)
from multiprocessing import get_context
print("multiprocessing part got imported in  main script")

# comment out below line to enable tensorflow outputs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)


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

from GenderDetectionImports.core.Multiprocess_python_people import python_function

#model load imports
from GenderDetectionImports.core.person_ReID import PRID_model_Load
from GenderDetectionImports.core.model_load import people_model_load
from GenderDetectionImports.core.model_load import head_model_load



print("Imported Multiprocess_python")

def thread_function(msg_passing,exp_process,completed_process,current_pro):
    import time
    global node
    while True:
        #print("withing while thread_function ")
        #time.sleep(1)
        if (len(msg_passing) != 0) and (node != None):
            print("withing threading msg_passing ")
            # for idx, m in enumerate(msg_passing):
            #     node.send(m)
            #     node.warn(m)
            #     msg_passing.pop(idx)
            
            while True:
                m = msg_passing.pop(0)
                node.send(m)
                node.warn(m)
                
                if len(msg_passing) == 0:
                    break
                
            # msg_from_child =  msg_passing[0]
            # node.send(msg_from_child)
            # node.warn(msg_from_child)
            # msg_passing[:] = []
            print("msg passing inside the thread:",msg_passing)

        if len(exp_process) > 0:
            common_exp = [pro for pro in current_pro.items() if pro[0] in exp_process]
            for et in common_exp:
                print("killing exception process inside threading ",et)
                et[1].kill()
                exp_process.remove(et[0])
                current_process.remove(et[0])
                del current_pro[et[0]]
                print("current pro inside the thread:",current_pro)
                print("GPU utilization after initializing threading")
                GPUtil.showUtilization()

        if len(completed_process) > 0:   #this is for video completion. not for rtsp stream. Kill completed video process
            common_completed = [pro for pro in current_pro.items() if pro[0] in completed_process]
            for ct in common_completed :
                print("killing completed thread inside threading ",ct)
                ct[1].kill()
                completed_process.remove(ct[0])
                current_process.remove(ct[0])
                del current_pro[ct[0]]
                print("current pro inside the thread:",current_pro)
                print("GPU utilization after initializing threading")
                GPUtil.showUtilization()

if __name__ == '__main__':


    from multiprocessing import Manager
    manager = Manager()


    #process_request_return ="donepythonfunction"
    current_pro = {}
    canceled_pro = []
    exp_process = manager.list([])
    current_process = manager.list([])
    completed_process = manager.list([])
    msg_passing = manager.list([])
    
    print("GPU utilization after initializing manager")
    GPUtil.showUtilization()

    t = threading.Thread(target=thread_function, args=(msg_passing,exp_process,completed_process,current_pro,))
    t.start()

    print("GPU utilization after initializing threading")
    GPUtil.showUtilization()
    #global next_request
    #global current_process
    #global exp_process
    #global completed_process
    global msg
    global msg_local
    global node
    
    #model Loads
    people_params = people_model_load()
    head_params = head_model_load()
    model_RID = PRID_model_Load()

    people_params_l = manager.list([])
    head_params_l = manager.list([])
    model_RID_l = manager.list([])
    
    people_params_l.append(people_params)
    head_params_l.append(head_params)
    model_RID_l.append(model_RID)


    


    while True:
        print("Reading the Node's input started")



        #get the message from process button click
        raw_msg = channel.readline() #Node's Input
        #print('raw_msg',raw_msg)
        if not raw_msg:
            raise RuntimeError('Received EOF!')
        msg = json.loads(raw_msg.decode('utf-8'))
        msg_local = json.loads(raw_msg.decode('utf-8'))   #msg is a global variable. the value of it will be changed after python_function ends. so copy the value at the begining to msg_local variable. this is needed for saving last 5 min streaming video

        
        #print('msg print:',msg)
        #print('msg local print:',msg_local)

        msgid = msg["_msgid"]
        node = Node(msgid, channel)

        print('current processes:',current_process)
        print('current pro:',current_pro)
        print('exception processes:',exp_process)
        print('completed_process:',completed_process)
        print('people_params_l:',people_params)
        print('head_params_l:',head_params)
        print('model_RID_l:',model_RID)

        
        #when the number of process is not sent or when stop button is clicked it should go to exception
        try:
            NoOfProcesses = msg["payload"]["NoOfProcesses"]
            print("No of processes : ",NoOfProcesses)
        except Exception as e:
            NoOfProcesses = 3
            print("No of processes when no of processes is not given : ",NoOfProcesses)



        if len(exp_process) > 0:
            common_exp = [pro for pro in current_pro.items() if pro[0] in exp_process]
            for et in common_exp:
                print("killing exception process ",et)
                et[1].kill()
                exp_process.remove(et[0])
                current_process.remove(et[0])
                del current_pro[et[0]]
            

        if len(completed_process) > 0:   #this is for video completion. not for rtsp stream
            common_completed = [pro for pro in current_pro.items() if pro[0] in completed_process]
            for ct in common_completed :
                print("killing completed thread ",ct)
                ct[1].kill()
                completed_process.remove(ct[0])
                current_process.remove(ct[0])
                del current_pro[ct[0]]

        #print('process_request_return',process_request_return)
        #print("length of the current processs : ",len(current_process))

        if len(current_process) < NoOfProcesses:  # only get the next request only if any process not running in threads.
            print("Inside Current process <",NoOfProcesses)
            #when process button clicked - no exception. when cancel button clicked - exception
            try:
                video_ext = msg["payload"]["video_type"]
                friendly_name = msg["payload"]["friendly_name"]
                
            except Exception as e:
                print("Inside process exception")
                video_ext = ""
                friendly_name = ""

            #when process button clicked - exception. when cancel button clicked - no exception
            try:
                process = msg["payload"]["process"]
                friendly_name_cancel = msg["payload"]["video"]
            except Exception as e:
                process = ""
                friendly_name_cancel = ""
                #print(friendly_name," started processing")
                pass  #do the rest

                # process = ""
                # friendly_name_cancel = ""

            
        elif len(current_process) >= NoOfProcesses:
            print("length > ",NoOfProcesses)
            try:
                video_ext = ""
                friendly_name = ""

                #here the process will sometimes have value 'canceled' if the user wants to kill running threat
                process = msg["payload"]["process"]
                friendly_name_cancel = msg["payload"]["video"]
                print(friendly_name," is going to canceled")
            except:
                print("Can't process new video. Already ",current_process," is/are running")
                #current_process[0].kill()  #temporarily added to kill process when we could't stop running process
                #current_processd = []      #temporarily added to kill process when we could't stop running process
                continue

        else:
            continue

        

        print("process", process)
        print("friendly_name_cancel", friendly_name_cancel)
        print("video_ext", video_ext)
        print("friendly_name", friendly_name)

        try:

            #msg = python_function(msg)   #create here access it anywhere like "global variable_name"

            if (video_ext in ['mjpeg','rtsp']) or (process=="canceled"):
                

                #print("process,thread",process,current_process)
                #print('Python function started. Detection for video stream')
                

                # if next_request==False and process=="canceled":
                if process=="canceled":

                        print("video "+friendly_name_cancel+" canceled ", current_process)
                        if len(current_process)>0:
                            #next_request = True
                            cancel_status = {'video_name':friendly_name_cancel,'status':'canceled'}
                            msg['payload'] = cancel_status
                            print(msg['payload'] )
                            node.send(msg)   #Node's Output
                            canceled_pro = [pro for pro in current_pro.items() if pro[0] == friendly_name_cancel]#identifies the process using the friendly names
                            canceled_pro[0][1].kill()
                            #print("current_process", current_process)
                            #print("canceled proo : ", canceled_pro[0])
                            current_process.remove(canceled_pro[0][0])
                            del current_pro[canceled_pro[0][0]]

                            # common_json_path = '/interplay_v2/public/private/weaponresource/weapon_mask_display.json'
                            # with open(common_json_path,'r') as f2:
                            #     data_json = json.load(f2)
                            #     #print('Loaded data from common json:',data_json)
                            #     f2.flush()
                            #     f2.close()

                            # data_json_new = []
                            # if len(data_json) !=0:   #if len(data_json) ==0 #when we try to loop it will give error saying trying to loop through empty array
                            #     for j in data_json:
                            #         if j['video_name'] != friendly_name_cancel:    #when python_function running more than one time again and again the summary json will append for the same stream. so 1st delete it and append again
                            #                 data_json_new.append(j)
                            #         elif j['video_name'] == friendly_name_cancel:
                            #                 data1 = {'status':'video completed','video_name':friendly_name_cancel,'video_type':j['video_type'],'datetime':j['datetime'],'Threat_status':j['Threat_status']}
                            # data_json_new.append(data1)


                            # with open(common_json_path,'w+') as fd3:
                            #         fd3.write(json.dumps(data_json_new))
                            #         #print("write canceled data_json to common json")
                            #         print("Stopped rtsp ", friendly_name_cancel)
                            #         fd3.flush()
                            #         fd3.close()

                            cancel_json = {"cancel_status":True,"friendly_name":friendly_name_cancel}
                            print(cancel_json)
                            node.send(cancel_json)

                            #process_request_return = "donepythonfunction"  #have to change this global variable to process new rtsp stream when it comes.
                        continue
                #for every 5*60*fps frames (5 mins) the python_function will break, save the video and come out. Again will save the next 5*60*fps frames as new video.....Thats y for every 5*60*fps frames we r exiting from python_function and restarting the python_function to save last 5 mins frames.
                node.warn('Detection for video stream starts...')
                
                print("before multiprocessing starts : ")
                GPUtil.showUtilization()
                

                
                #print("before multiprocessing starts")
                multiprocessing.freeze_support()
                p = multiprocessing.Process(target=python_function, args=(msg,msg_local,current_process,exp_process,completed_process,msg_passing,people_params_l,head_params_l,model_RID_l ))
                current_process.append(friendly_name)
                current_pro[friendly_name] = p
                p.start()
                print("before multiprocessing starts : ")
                GPUtil.showUtilization()
                #print("Multiprocess got started")


            elif (video_ext in ['mp4','webm']) or (process=="canceled"):
                #global threads_queue_mp4
                #print("process,thread",process,current_process)

                #print('Python function started. Detection for video')
                node.warn('Python function started. Detection for video')

                # if next_request==False and process=="canceled":
                if process=="canceled":

                        print("video "+friendly_name_cancel+" canceled ", current_process)

                        # kill thread is cancel request comes
                        if len(current_process)>0:
                            cancel_status = {'video_name':friendly_name_cancel,'status':'canceled'}
                            msg['payload'] = cancel_status
                            node.send(msg)   #Node's Output
                            canceled_pro = [pro for pro in current_pro.items() if pro[0] == friendly_name_cancel]#identifies the process using the friendly names
                            canceled_pro[0][1].kill()
                            current_process.remove(canceled_pro[0][0])
                            del current_pro[canceled_pro[0][0]]
                            print("The process of ",friendly_name_cancel," ",canceled_pro[0][0]," is canceled!")



                            # # update video completed status in common json. then only delete button will come.
                            # common_json_path = '/interplay_v2/public/private/weaponresource/weapon_mask_display.json'
                            # with open(common_json_path,'r') as f2:
                            #     data_json = json.load(f2)
                            #     #print('Loaded data from common json:',data_json)
                            #     f2.flush()
                            #     f2.close()

                            # data_json_new = []
                            # if len(data_json) !=0:   #if len(data_json) ==0 #when we try to loop it will give error saying trying to loop through empty array
                            #     for j in data_json:
                            #         if j['video_name'] != friendly_name_cancel:    #when python_function running more than one time again and again the summary json will append for the same stream. so 1st delete it and append again
                            #                 data_json_new.append(j)
                            #         elif j['video_name'] == friendly_name_cancel:
                            #                 data1 = {'status':'video completed','video_name':friendly_name_cancel,'video_type':j['video_type'],'datetime':j['datetime'],'Threat_status':j['Threat_status']}
                            # data_json_new.append(data1)


                            # with open(common_json_path,'w+') as fd3:
                            #         fd3.write(json.dumps(data_json_new))
                            #         #print("write canceled data_json to common json")
                            #         print("Stopped video ", friendly_name_cancel)
                            #         fd3.flush()
                            #         fd3.close()

                            cancel_json = {"cancel_status":True,"friendly_name":friendly_name_cancel}
                            node.send(cancel_json)
                
                        continue

                #start the process and append into current process
                #from multiprocessing import set_start_method
                #set_start_method("spawn", force=True)

                print("before multiprocessing starts : ")
                GPUtil.showUtilization()

                #from multiprocessing import get_context
                multiprocessing.freeze_support()
                p = multiprocessing.Process(target=python_function, args=(msg,msg_local,current_process,exp_process,completed_process,msg_passing,people_params_l,head_params_l,model_RID_l))
                current_process.append(friendly_name)
                current_pro[friendly_name] = p
                p.start()

                print("before multiprocessing ends: ")
                GPUtil.showUtilization()





            msgid = msg["_msgid"]
            node = Node(msgid, channel)
            node.send(msg)   #Node's Output
        except Exception as e:
            print("Passing",e,"Exception caused")
            pass
            #raise RuntimeError('Some Exception')

    `,
            attempts: 10
        };

        spawnFn(self);
        self.context().global.set("pythonturiinit", 1);
        //console.log("get it " +self.i_filename);
        self.on('input', function (msg) {
            // Save REQ object if it exists.
            if (msg.req !== undefined) {
                self.req = msg.req;
            }
            // Save RES object if it exists.
            if (msg.res !== undefined) {
                self.res = msg.res;
            }
            var cache = [];

            //   var t1=config.tuning_paramter1;
            //   msg.t1=parseFloat(t1);
            //   var t2=config.tuning_paramter2;
            //   msg.t2=parseFloat(t2);
            //   var t3=config.tuning_paramter3;
            //   msg.t3=parseFloat(t3);
            jsonMsg = JSON.stringify(msg, function (key, value) {
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
}
