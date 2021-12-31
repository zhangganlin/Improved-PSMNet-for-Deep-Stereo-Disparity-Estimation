from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from dataloader import KITTIloader2015 as kitti2015
from dataloader import KITTILoader as kittiDA

from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
                    help='datapath')
parser.add_argument('--kittidatapath', default='dataset/data_scene_flow_2015/training/',
                    help='kitti datapath')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables no CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--batchsize', type=int, default=2,
                    help='batch size')
parser.add_argument('--numworker', type=int, default=0,
                    help='num_worker')
parser.add_argument('--seg', action='store_true', default=False,
                    help='Whether add segmentation')
parser.add_argument('--gwc', action='store_true', default=False,
                    help='Whether use group wise cost volume')
parser.add_argument('--startepoch', type=int, default=0,
                    help='start from epoch')
parser.add_argument('--test', action='store_true', default=False,
                    help='enables testing')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp, all_left_seg, test_left_seg = kitti2015.dataloader(
    args.kittidatapath)

TrainImgLoader = torch.utils.data.DataLoader(
    kittiDA.myImageFloder(all_left_img, all_right_img, all_left_disp, all_left_seg, True),
    batch_size=args.batchsize, shuffle=True, num_workers=args.numworker, drop_last=False)
TestImgLoader = torch.utils.data.DataLoader(
    kittiDA.myImageFloder(test_left_img, test_right_img, test_left_disp, test_left_seg, False),
    batch_size=2, shuffle=False, num_workers=0, drop_last=False)

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
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))




def train(imgL, imgR, disp_L, seg_L):
    model.train()

    disp_true = disp_L
    if args.cuda:
        imgL, imgR, disp_true, seg_L = imgL.cuda(), imgR.cuda(), disp_L.cuda(), seg_L.cuda()

   # ---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    # ----
    optimizer.zero_grad()

    if args.model == 'stackhourglass' or args.model == 'dilated':
        output1, output2, output3 = model(imgL, imgR, seg_L)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(
            output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
    
    loss.backward()
    optimizer.step()

    return loss.data


def test(imgL, imgR, disp_true, seg_L):

    model.eval()

    if args.cuda:
        imgL, imgR, disp_true, seg_L = imgL.cuda(), imgR.cuda(), disp_true.cuda(), seg_L.cuda()
    # ---------
    mask = disp_true < 192
    # ----

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16
        top_pad = (times+1)*16 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3]//16
        right_pad = (times+1)*16-imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))
    seg_L = F.pad(seg_L, (0, right_pad, top_pad, 0) )

    with torch.no_grad():
        output3 = model(imgL, imgR, seg_L)
        output3 = torch.squeeze(output3)

    if top_pad != 0:
        img = output3[:, top_pad:, :]
    else:
        img = output3

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        # torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error
        loss = F.l1_loss(img[mask], disp_true[mask])

    return loss.data.cpu()


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main_train():
    
    if args.startepoch != 0:
        loss_to_write_file_name = args.savemodel+"/"+str(args.startepoch)+model_name+"_kittiloss.txt"
    else:   
        loss_to_write_file_name =args.savemodel+"/"+str(args.startepoch)+model_name+"_kittiloss.txt"
    loss_to_write = open(loss_to_write_file_name,"w")

    start_full_time = time.time()
    for epoch in range(args.startepoch, args.startepoch+args.epochs):
        print('This is %d-th epoch' % (epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)
        
        if epoch%20 == 0:
            loss_to_write.close()
            loss_to_write = open(loss_to_write_file_name,"a")

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, seg_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L, seg_L)
            print('Iter %d training loss = %.3f , time = %.2f' %
                  (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %
              (epoch, total_train_loss/len(TrainImgLoader)))

        loss_to_write.write("{}\n".format(total_train_loss/len(TrainImgLoader)))

        # SAVE
    savefilename = args.savemodel+'/kitticheckpoint_'+str(epoch)+model_name+'.tar'
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'train_loss': total_train_loss/len(TrainImgLoader),
    }, savefilename)

    print('full training time = %.2f HR' %
          ((time.time() - start_full_time)/3600))
    loss_to_write.close()
    return

def main_test():
    # ------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L, seg_L) in enumerate(TestImgLoader):
        test_loss = test(imgL, imgR, disp_L, seg_L)
        print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
        total_test_loss += test_loss

    print('total test loss = %.3f' % (total_test_loss/len(TestImgLoader)))
    # ----------------------------------------------------------------------------------
    # SAVE test information
    loss_to_write_file_name =args.savemodel+"/kittiloss.txt"
    loss_to_write = open(loss_to_write_file_name,"a")
    loss_to_write.write("test_loss: {}\n".format(total_test_loss/len(TestImgLoader)))
if __name__ == '__main__':
        main_train()
        main_test()
