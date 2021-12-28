from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *

class dilated_sub(nn.Module):
    def __init__(self, inplanes):
        super(dilated_sub, self).__init__()

        self.conv0 = nn.Sequential(convbn_3d(inplanes, inplanes, 3, 2, 1),        
                                   nn.ReLU(inplace=True))                             #39

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes, 3, 1, 1),
                                   nn.ReLU(inplace=True))                             #40

        self.conv2_1 = nn.Sequential(convbn_3d(inplanes, inplanes, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2_2 = nn.Sequential(convbn_3d(inplanes, inplanes, kernel_size=3, stride=1, pad=2, dil = 2),
                                   nn.ReLU(inplace=True))

        self.conv2_3 = nn.Sequential(convbn_3d(inplanes, inplanes, kernel_size=3, stride=1, pad=4, dil = 4),
                                   nn.ReLU(inplace=True))

        self.conv3_cat = nn.Sequential(convbn_3d(inplanes*3, inplanes, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))
        
        self.deconv = nn.Sequential(nn.ConvTranspose3d(inplanes, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(32))
        self.pred_conv = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

    def forward(self, x, last):
        out_1 = self.conv0(x)
        out_2 = self.conv3_cat(torch.cat((self.conv2_1(out_1),self.conv2_2(out_1),self.conv2_3(out_1)),1))
        out_1 = self.conv1(out_1)+last
        
        out_2 = self.deconv(out_2)
        out_3 = self.pred_conv(out_2)
        return out_1,out_2,out_3

class PSMNet(nn.Module):
    def __init__(self, maxdisp, gpu=True, num_groups = 40, concat_channels=12, seg=False):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.seg = seg

        self.gpu = gpu

        self.num_groups = num_groups

        self.concat_channels = concat_channels   #cost volume channels: num_groups + 2*concat_channels

        self.feature_extraction = feature_extraction(self.concat_channels)

        self.layer37_38 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.layer39_46 = dilated_sub(32)
        self.layer47_54 = dilated_sub(32)
        self.layer55_62 = dilated_sub(32)


        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.final_conv = nn.Sequential(convbn(2,16,3,1,1,1),
                                        nn.ReLU(inplace=True),
                                        convbn(16,1,3,1,1,1),
                                        nn.ReLU(inplace=True))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right, seg):

        refimg_fea, refimg_fea_gwc = self.feature_extraction(left)
        targetimg_fea, targetimg_fea_gwc = self.feature_extraction(right)

        if self.num_groups == 0:
            concat_volume = build_concat_volume(refimg_fea, targetimg_fea, self.maxdisp // 4)
            volume = concat_volume
        else:
            gwc_volume = build_gwc_volume(refimg_fea_gwc, targetimg_fea_gwc, self.maxdisp // 4, self.num_groups)
            concat_volume = build_concat_volume(refimg_fea, targetimg_fea, self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)

        layer38_out = self.layer37_38(volume)
        B,C,H,W,D = layer38_out.shape
        layer40_out, layer45_out, cost1 = self.layer39_46(layer38_out,layer38_out.new_zeros([B,C,H//2,W//2,D//2]))
        cost1 = cost1+layer38_out

        layer48_out, layer53_out, cost2 = self.layer47_54(layer45_out,layer40_out)
        cost2 = cost2+layer38_out

        _,_,cost3 = self.layer55_62(layer53_out,layer48_out)
        cost3 = cost3+layer38_out

        cost1 = self.classif1(cost1)
        cost2 = self.classif2(cost2) + cost1
        cost3 = self.classif3(cost3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp,self.gpu)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp,self.gpu)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp,self.gpu)(pred3)

        if self.training:
            if self.seg==False:
                return pred1, pred2, pred3
            else:
                pred1 = self.final_conv(torch.cat((pred1, seg), 1))
                pred2 = self.final_conv(torch.cat((pred2, seg), 1))
                pred3 = self.final_conv(torch.cat((pred3, seg), 1))
                return pred1, pred2, pred3
        else:
            if self.seg == False:
                return pred3
            else:
                pred3 = self.final_conv(torch.cat((pred3, seg), 1))
                return pred3
