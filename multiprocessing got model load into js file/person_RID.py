import torch
import os
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

import yaml

from torchvision import datasets, models, transforms
import torch

from GenderDetectionImports.Person_reID_baseline_pytorch.model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from GenderDetectionImports.Person_reID_baseline_pytorch.utils import fuse_all_conv_bn

import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import PIL

########################################

            #DEFINE VARIABLES

########################################
which_epoch = 'last'
ms = [1.0]
linear_num =512
name = 'ft_ResNet50'
base_path = '/interplay_v2/GenderDetectionImports/Person_reID_baseline_pytorch/'
batchsize = 256


config_path = os.path.join(base_path+'model',name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'

fp16 = config['fp16'] 
PCB = config['PCB']
use_dense = config['use_dense']
use_NAS = config['use_NAS']
stride = config['stride']


if 'use_swin' in config:
    use_swin = config['use_swin']
if 'use_swinv2' in config:
    use_swinv2 = config['use_swinv2']
if 'use_convnext' in config:
    use_convnext = config['use_convnext']
if 'use_efficient' in config:
    use_efficient = config['use_efficient']
if 'use_hr' in config:
    use_hr = config['use_hr']


if 'nclasses' in config: # tp compatible with old config files
    nclasses = config['nclasses']
else: 
    nclasses = 751 

if 'ibn' in config:
    ibn = config['ibn']
if 'linear_num' in config:
    linear_num = config['linear_num']


query_index = 0

if use_swin:
    h, w = 224, 224
else:
    h, w = 256, 128
if PCB:    
    h, w = 384, 192

use_gpu = torch.cuda.is_available()


########################################

            #DEFINE FUNCTIONS

########################################
def load_network(network):
    save_path = os.path.join(base_path+'model',name,'net_%s.pth'%which_epoch)
    network.load_state_dict(torch.load(save_path),strict=False)
    return network

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders, linear_num):
    #features = torch.FloatTensor()
    count = 0
    if linear_num <= 0:
        if use_swin or use_swinv2 or use_dense or use_convnext:
            linear_num = 1024
        elif use_efficient:
            linear_num = 1792
        elif use_NAS:
            linear_num = 4032
        else:
            linear_num = 2048

    for iter, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)
        ff = torch.FloatTensor(n,linear_num).zero_().cuda()

        if PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        if PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        
        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
        #features = torch.cat((features,ff.data.cpu()), 0)
        start = iter*batchsize
        end = min( (iter+1)*batchsize, len(dataloaders.dataset))
        features[ start:end, :] = ff
    return features

'''
def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels
'''
def get_id(img_path):
    labels = []
    #print('img_path',img_path)
    for path, v in img_path:
        # print('path',path)  #/content/drive/MyDrive/person_Re_Identification/Person_reID_baseline_pytorch/dataset/test_imgs_people_counting/gallery/1/10.jpg
        # print('v',v) #0

        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        # print('filename',filename) #10.jpg

        path_spt = path.split('/')
        label = path_spt[len(path_spt)-2]
        # print('label',label) #1

        labels.append(int(label))
    return labels

def PRID_model_Load():
    ########################################
    
                #LOAD TRAINED MODEL
    
    ########################################
    print('-------test-----------')
    if use_dense:
        model_structure = ft_net_dense(nclasses, stride = stride, linear_num=linear_num)
    elif use_NAS:
        model_structure = ft_net_NAS(nclasses, linear_num=linear_num)
    elif use_swin:
        model_structure = ft_net_swin(nclasses, linear_num=linear_num)
    elif use_swinv2:
        model_structure = ft_net_swinv2(nclasses, (h,w),  linear_num=linear_num)
    elif use_convnext:
        model_structure = ft_net_convnext(nclasses, linear_num=linear_num)
    elif use_efficient:
        model_structure = ft_net_efficient(nclasses, linear_num=linear_num)
    elif use_hr:
        model_structure = ft_net_hr(nclasses, linear_num=linear_num)
    else:
        model_structure = ft_net(nclasses, stride = stride, ibn = ibn, linear_num=linear_num)
    
    if PCB:
        model_structure = PCB(nclasses)
    #print('nclasses',nclasses)
    #if opt.fp16:
    #    model_structure = network_to_half(model_structure)
    
    
    model = load_network(model_structure)
    
    # Remove the final fc layer and classifier layer
    if PCB:
        #if opt.fp16:
        #    model = PCB_test(model[1])
        #else:
            model = PCB_test(model)
    else:
        #if opt.fp16:
            #model[1].model.fc = nn.Sequential()
            #model[1].classifier = nn.Sequential()
        #else:
            model.classifier.classifier = nn.Sequential()
    
    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    
    
    print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
    model = fuse_all_conv_bn(model)
    
    # We can optionally trace the forward method with PyTorch JIT so it runs faster.
    # To do so, we can call `.trace` on the reparamtrized module with dummy inputs
    # expected by the module.
    # Comment out this following line if you do not want to trace.
    '''
    dummy_forward_input = torch.rand(batchsize, 3, h, w).cuda()
    model = torch.jit.trace(model, dummy_forward_input)
    '''
    print(model)
    
    return model

def person_RID(data_dir, model):
    ########################################
    
                #DATA LOAD
    
    ########################################
    data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ############### Ten Crop        
            #transforms.TenCrop(224),
            #transforms.Lambda(lambda crops: torch.stack(
             #   [transforms.ToTensor()(crop) 
              #      for crop in crops]
               # )),
            #transforms.Lambda(lambda crops: torch.stack(
             #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
              #       for crop in crops]
              # ))
    ])
    
    if PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((384,192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
    
    
    
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                shuffle=False, num_workers=4) for x in ['gallery','query']}
    
    class_names = image_datasets['query'].classes
    
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs
    
    gallery_label = get_id(gallery_path)
    query_label = get_id(query_path)
    
    

    
    
    ########################################
    
                #SAVE FEATURES
    
    ########################################
    
    
    import scipy.io
    import time
    
    
    # Extract feature
    since = time.time()
    with torch.no_grad():
        gallery_feature = extract_feature(model,dataloaders['gallery'], linear_num)
        query_feature = extract_feature(model,dataloaders['query'], linear_num)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.2f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    
    
    
    
    del image_datasets
    del dataloaders
    
    '''
    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'query_f':query_feature.numpy(),'query_label':query_label}
    
    scipy.io.savemat(base_path+'pytorch_result_people.mat',result)
    
    
    
    
    
    
    
    

    
    ######################################################################
    result = scipy.io.loadmat(base_path+'pytorch_result_people.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]
    '''
    
    ########################################
    
                #INFERENCE
    
    ########################################
    
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}
    
    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()
    
    #######################################################################
    # sort the images
    def get_max_score(qf, ql, gf, gl):
        query = qf.view(-1,1)
        # print(query.shape)
        score = torch.mm(gf,query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        # predict index
        index = np.argsort(score)  #from small to large
        index = index[::-1]
        
        max_score = score.max()
        label =index[-1]
    
        return max_score
    
    i = query_index
    #print('query_feature[i]',query_feature[i])
    #print('query_label[i]',query_label[i])
    #print('gallery_feature',gallery_feature)
    #print('gallery_label',gallery_label)
    
    max_score = get_max_score(query_feature[i],query_label[i],gallery_feature,gallery_label)
    
    del query_feature
    del gallery_feature
    del query_label
    del gallery_label
    
    return max_score