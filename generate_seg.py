# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
import sys
sys.path
sys.path.append('semantic_segmentation')
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg



parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
parser.add_argument(
    "--kitti15",
    default="dataset/data_scene_flow_2015/training",
    type=str
)
parser.add_argument(
    "--kitti12",
    default="dataset/data_stereo_flow_2012/training",
    type=str,
)

parser.add_argument(
    "--driving",
    default="/cluster/scratch/zhangga/dataset",
    type=str,
)

parser.add_argument(
    "--gpu",
    default=0,
    type=int,
    help="gpu id for evaluation"
)

args = parser.parse_args()




def visualize_result(data, pred, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = pred.astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(pred).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))


gpu=args.gpu


cfg.merge_from_file("semantic_segmentation/config/ade20k-resnet50dilated-ppm_deepsup.yaml")
# cfg.freeze()

cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()
cfg.DIR = "semantic_segmentation/" + cfg.DIR
cfg.TEST.result = "./seg_result"
# absolute paths of model weights
cfg.MODEL.weights_encoder = os.path.join(
    cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
cfg.MODEL.weights_decoder = os.path.join(
    cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

assert os.path.exists(cfg.MODEL.weights_encoder) and \
    os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

if gpu != -1:
    torch.cuda.set_device(gpu)
net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder,
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_encoder)
net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder,
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=cfg.MODEL.weights_decoder,
    use_softmax=True)

crit = nn.NLLLoss(ignore_index=-1)

segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)


def generate_seg(src_folder, save_folder):
    
    global segmentation_module
    
    imgs = find_recursive(src_folder,".png")
    cfg.list_test = [{'fpath_img': x} for x in imgs]
    cfg.TEST.result = save_folder
    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=4,
        drop_last=True)
    if gpu != -1:
        segmentation_module = nn.DataParallel(segmentation_module)
        segmentation_module.cuda()
        
    segmentation_module.eval()


    pbar = tqdm(total=len(loader_test))

    for batch_data in loader_test:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])

            if gpu != -1:
                scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                if gpu != -1:
                    feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )
        pbar.update(1)



src_folder_kitti15 = args.kitti15+"/image_2"
src_folder_kitti12 = args.kitti12+"/colored_0"

result_folder_kitti15 = args.kitti15+"/seg"
result_folder_kitti12 = args.kitti12+"/seg"

generate_seg(src_folder_kitti15, result_folder_kitti15)
generate_seg(src_folder_kitti12, result_folder_kitti12)