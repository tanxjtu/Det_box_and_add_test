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
        print 'The next vid name:  ', vd_name, '  Frm num: ', fm_num
        jpg_bs_name = fm_lis[0][:-10]
        frm_list = [jpg_bs_name+'%06d.jpg' % No for No in range(1, fm_num+1)]
        ft_im = cv2.imread(frm_list[0], 1 | 128)
        ft_im, im_scale = resize(ft_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tenssp = transform(ft_im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tenssp.shape[2], im_tenssp.shape[3], im_scale]], dtype=np.float32)
        data = []
        # for fm_dir in frm_list:
        # to change back to change back to change back to change back to change back to change back to change back

        # ====================  get each video frames and generate data ====================
        for fm_dir in frm_list:
            fm = cv2.imread(fm_dir, 1 | 128)
            #target_size  #max_size
            # im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            fm, _ = resize(fm, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            fm_sptensor = transform(fm, config.network.PIXEL_MEANS)
            # im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
            data.append({'data': fm_sptensor, 'im_info': im_info})

        for x in locals().keys():
            del locals()[x]
        Memery = gc.collect()
        del fm_sptensor, fm
        Memery = gc.collect()


        # process
        data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
        provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
        provide_label = [None for i in xrange(len(data))]

        predictor = Predictor(sym, data_names, label_names,
                              context=[mx.gpu(GPU)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
        # nms  discleared  PASSED
        # test

        # ====================  process each to get bbox  ====================
        each_video_det = {}
        for idx, im_name in enumerate(frm_list):
            data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                         provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                         provide_label=[None])
            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
            scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
            boxes = boxes[0].astype('f')
            scores = scores[0].astype('f')
            dets_nms = []
            for j in range(1, scores.shape[1]):
                cls_scores = scores[:, j, np.newaxis]
                cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                cls_dets = cls_dets[keep, :]
                cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
                if len(cls_dets) > 0:
                    dets_nms.append(np.insert(cls_dets, 5, values=j, axis=1).tolist())

            each_video_det[im_name.split('/')[-1][-10:-4]] = dets_nms
            # print 'testing {} {:.4f}s'.format(im_name, toc())
            # # visualize
            # im = cv2.imread(im_name)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # show_boxes(im, dets_nms, classes, 1)
        save_each_video_det(video_det=each_video_det, save_dir=sv_dir, video_name=vd_name)
        print order+1, '/', len(vid_list), 'time {:.4f}s'.format(toc()),vd_name






if __name__ == '__main__':
    list_file = 'videos_list.json'
    with open(list_file, 'r') as load_f:
        load_file = json.load(load_f)

    pcs_list = load_file[9500:]
    GPU = 2  # 0,1,2,3
    Pcs_each_vdo(pcs_list, bbox_dir, GPU)
