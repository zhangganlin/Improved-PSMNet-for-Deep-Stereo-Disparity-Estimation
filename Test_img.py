from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='ImprovedPSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help="select model 'stackhourglass' or 'dilated'")
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables no CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--seg', action='store_true', default=False,
                    help='Whether add segmentation')
parser.add_argument('--gwc', action='store_true', default=False,
                    help='Whether use group wise cost volume')
parser.add_argument('--leftimg', default= './VO04_L.png',
                    help='left image')
parser.add_argument('--rightimg', default= './VO04_R.png',
                    help='right image')  
parser.add_argument('--segimg', default= './VO04_L.png',
                    help='segmentation image')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.gwc:
    num_groups = 40
    concat_channels=12
else:
    num_groups = 0
    concat_channels = 32


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp,args.cuda, num_groups, concat_channels, seg=args.seg)
    model_name = 'psm'
elif args.model == 'dilated':
    model = dilated(args.maxdisp,args.cuda, num_groups, concat_channels, seg=args.seg)
    model_name = 'dilated'
else:
    print('no model')

if args.gwc:
    model_name = model_name+"_gwc"
if args.seg:
    model_name = model_name+"_seg"

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    if args.no_cuda:
        pretrain_dict = torch.load(args.loadmodel,map_location=torch.device('cpu'))['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        pretrain_dict = torch.load(args.loadmodel)
        model.load_state_dict(pretrain_dict['state_dict'])


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR,imgSeg):
        model.eval()

        if args.cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda()     
           imgSeg = imgSeg.cuda() 

        with torch.no_grad():
            disp = model(imgL,imgR,imgSeg)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp


def main():

        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])    

        imgL_o = Image.open(args.leftimg).convert('RGB')
        imgR_o = Image.open(args.rightimg).convert('RGB')
        imgSeg_o = Image.open(args.segimg)


        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o) 
        imgSeg = transforms.ToTensor()(imgSeg_o)
       

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgSeg = F.pad(imgSeg,(0,right_pad, top_pad,0)).unsqueeze(0)

        start_time = time.time()
        pred_disp = test(imgL,imgR,imgSeg)
        print('time = %.2f' %(time.time() - start_time))

        
        if top_pad !=0 and right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            img = pred_disp[:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            img = pred_disp[top_pad:,:]
        else:
            img = pred_disp
        
        img = (img*256).astype('uint16')
        img = Image.fromarray(img)
        img.save('Test_disparity.png')

if __name__ == '__main__':
   main()






