import sys
import os
import cv2
import json
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.image as mping
import time
import argparse
import multiprocessing
import numpy as np
from PIL import Image
from random import random as rand


num_classes = 81

classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
           'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']  # COCO class list

def Prc_video(vid_list, frm_num, sv_base_dir, BBox_dir):
    for No_vid, Vd_Dir in enumerate(vid_list):
        vd_name = Vd_Dir.split('/')[-1]
        vd_sv_floder = os.path.join(sv_base_dir, vd_name)
        if not os.path.exists(vd_sv_floder):
            os.mkdir(vd_sv_floder)
        elif len(glob.glob(vd_sv_floder + '/*')) == frm_num:
            continue

        # process each vid
        vd_fm_num = len(glob.glob(Vd_Dir + '/*'))
        fms_dir = [Vd_Dir+'/'+ vd_name +'-%06d.jpg' %random.randint(1, vd_fm_num) for No in range(frm_num)]
        IMg_W,IMg_H = Image.open(fms_dir[0]).size
        vd_bbox = json.load(open(os.path.join(BBox_dir, vd_name + '.json'), 'r'))

        # process each frame and save image
        for No, frame_dir in enumerate(fms_dir):
            # frame = Image.open(frame_dir)
            # frm_bbox = vd_bbox[frame_dir[-10:-4]]

            frame = mping.imread(frame_dir)
            frm_bbox = vd_bbox[frame_dir[-10:-4]]
            plt.axis("on")
            plt.imshow(frame)

            for j, each_class_tube in enumerate(frm_bbox):
                for single_bbox in each_class_tube:
                    color = (rand(), rand(), rand())
                    rect = plt.Rectangle((single_bbox[0], single_bbox[1]), single_bbox[2] - single_bbox[0],
                                         single_bbox[3] - single_bbox[1], fill=False, edgecolor=color, linewidth=2.5)
                    plt.gca().add_patch(rect)
                    score = single_bbox[-2]
                    plt.gca().text(single_bbox[0], single_bbox[1],
                                   '{:s} {:.3f}'.format(classes[int(single_bbox[-1]) - 1], score),
                                   bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
            saved_fame_dir = vd_sv_floder + '/' + '%06d.jpg'% int(frame_dir[-9:-4])
            plt.savefig(saved_fame_dir)
            plt.close()
        print('The ', No_vid+1, ' / ', len(vid_list))


if __name__ == '__main__':

    # bacause each video contain to much of frames so select some of them
    fl_num_2_add = 10

    # basic info
    sv_added_pth = './Charades_ADD_tube'
    video_info_fl = './videos_list.json'
    BBox_dir = '/data1/THL_Dataset/charades/BBox_Charades/BBOX'  # absolute path

    # Load the video info to pre process
    with open(video_info_fl, 'r') as load_f:
        vido_Dir = json.load(load_f)

    #
    vido_Dir = vido_Dir[0:1500]
    Prc_video(vido_Dir, fl_num_2_add, sv_added_pth, BBox_dir)
