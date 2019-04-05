#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging

import sys
import time
import json
import numpy as np

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        # help='./configs/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml',
        default='../configs/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml',
        # default='../configs/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        # default=None,
        default='../trained_model/FasterR-50-FPNX2.pkl',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        # default='/tmp/infer_simple',
        default='../out_test',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--box_per_thre',
        dest='box_per_thre',
        help='Threshold for box human score ',
        default=0.5,
        type=float
    )

    parser.add_argument(
        '--box_obj_thre',
        dest='box_obj_thre',
        help='Threshold for box human score ',
        default=0.4,
        type=float
    )


    parser.add_argument(
        '--im_or_folder', help='image or folder of images',
        # default=None,
        default='../demo/HICO_test2015_00000002.jpg'
    )

    parser.add_argument(
        '--datatype',
        dest='datatype',
        help='test or train data',
        default='train',
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(1)
    return parser.parse_args()


def save_each_video_det(video_det, save_dir):
    json_name = save_dir
    with open(json_name, "w") as file:
        json.dump(video_det, file)
        file.close()

def main(args):
    # logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if args.datatype=='test':
        img_dir = '/HICO_DET/hico_20160224_det/images/test2015'
        save_father_dir = '/HICO_DET/box/Faster_RCNN_50/test'
    elif args.datatype=='train':
        img_dir = '/HICO_DET/hico_20160224_det/images/train2015'
        save_father_dir = '/HICO_DET/box/Faster_RCNN_50/train'
    else:
        assert 0, "img type must be test or train"

    im_list = glob.glob(img_dir+'/*')
    im_list.sort()
    img_num = len(im_list)

    for i, im_name in enumerate(im_list):
        save_name = im_name.split('/')[-1].split('.')[0]+'.json'
        im_det_sv_pth = os.path.join(save_father_dir,save_name)
        if os.path.exists(im_det_sv_pth):
            continue
        print('Process', i+1, ' / ', img_num,'   img name  ', save_name.split('.')[0] )
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        # t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )

        # select box
        each_img_out = []
        box_per_thre = args.box_per_thre   # 0.7
        box_obj_thre = args.box_obj_thre   # 0.4
        human_hum = 0
        object_num = 0
        for class_idx, each_box in enumerate(cls_boxes[1:]):
            class_name = dummy_coco_dataset['classes'][class_idx+1]
            if class_name == 'person':
                threthold = box_per_thre
            else:
                threthold = box_obj_thre
            for each_ins in each_box:
                if each_ins[-1] > threthold:
                    if class_name == 'person':
                        human_hum += 1
                    object_num += 1
                    each_ins = np.around(each_ins, 2)
                    ins = each_ins.tolist()
                    ins.append(class_name)
                    each_img_out.append(ins)

        each_img_out.append({'per_num': human_hum, 'total_num': object_num})
        with open(im_det_sv_pth, "w") as file:
            json.dump(each_img_out, file)
            file.close()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
