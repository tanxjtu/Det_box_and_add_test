import _init_paths

import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
from PIL import Image
import glob
import json

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/rfcn/cfgs/rfcn_coco_demo.yaml')
sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
import gc


def parse_args():
    parser = argparse.ArgumentParser(description='Show Deformable ConvNets demo')
    # general
    parser.add_argument('--rfcn_only', help='whether use R-FCN only (w/o Deformable ConvNets)', default=False, action='store_true')

    args = parser.parse_args()
    return args

args = parse_args()

num_classes = 81
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
           'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


# pprint.pprint(config)
config.symbol = 'resnet_v1_101_rfcn_dcn' if not args.rfcn_only else 'resnet_v1_101_rfcn'
sym_instance = eval(config.symbol + '.' + config.symbol)()
sym = sym_instance.get_symbol(config, is_train=False)

data_names = ['data', 'im_info']
label_names = []
max_data_shape = [[('data', (1, 3, 600, 1000))]]
arg_params, aux_params = load_param(cur_path + '/../model/' + ('rfcn_dcn_coco' if not args.rfcn_only else 'rfcn_coco'),
                                    0, process=True)
nms = gpu_nms_wrapper(config.TEST.NMS, 0)


Charades_frame = '/data1/THL_Dataset/charades/rgb/Charades_v1_rgb'
bbox_dir = '/home/thl/Desktop/BBox_Charades/BBOX'
# each_classes_dir = glob.glob(Charades_frame + '/*')  # dir name  long 9848== 9848(total video)
# all the video name is 5 long  type str
target_size = config.SCALES[0][0]
max_size = config.SCALES[0][1]


def save_each_video_det(video_det, save_dir, video_name):
    json_name = save_dir+'/'+video_name + '.json'
    with open(json_name, "w") as file:
        json.dump(video_det, file)
        file.close()


def Pcs_each_vdo(vid_list,sv_dir, GPU):
    # process each_video
    for order, ec_vid in enumerate(vid_list):
        ec_vid = str(ec_vid)
        tic()
        fm_lis = glob.glob(ec_vid + '/*')
        fm_num = len(fm_lis) # OK count right
        vd_name = ec_vid.split('/')[-1]
        sv_ph = sv_dir + '/' + vd_name
        # judge whether the json exist
        json_fl =sv_ph + '.json'  # if exist and the key length is frame then PASS
        if os.path.exists(json_fl):
            with open(json_fl,'r') as load_f:
                load_file = json.load(load_f)
                det_frm_nm = len(list(load_file.keys()))
                if det_frm_nm == fm_num:
                    print 'the file name  ', order + 1, ' ',  vd_name, '  Exist OK  Passed'
                    continue
   

if __name__ == '__main__':
    list_file = 'videos_list.json'
    with open(list_file, 'r') as load_f:
        load_file = json.load(load_f)

    pcs_list = load_file
	
    GPU = 2  # 0,1,2,3
    
