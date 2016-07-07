#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
print ("starting detection")

import rospkg
from rospkg import rospack
import struct
import pickle
import time

import rospy
from std_msgs.msg import String
from lsd_slam_viewer.msg import keyframeMsg
from lsd_slam_viewer.msg import keyframeGraphMsg
from PIL import Image

import os,sys
ros_slam_path = "/home/meppe/Coding/semslam"
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn")
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn/caffe-fast-rcnn/python")
sys.path.insert(0, ros_slam_path+"/src/frcnn/src/py-faster-rcnn/lib")


from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
# from numpy import uint8
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

print("imports done!")
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
#
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
#

# DETECTION_RUNNING = False
NEW_DETECTION = False
VIS_RUNNING = False
# DETECT_RUNNING = False
current_scores = []
current_boxes = []
current_kf = None
current_kf_id = None
min_c = 255
max_c = 0
CONF_THRESH = 0.2
NMS_THRESH = 0.1

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    print("Visualizing detection of class " + str(class_name))

    # switch red and blue
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    # plt.draw()
    plt.savefig("kf_"+str(current_kf_id) + "_" + str(class_name) + ".png")
    print("image drawn")



def frame_detect(net, im=None):
    """Detect object classes in an image using pre-computed object proposals."""
    global NEW_DETECTION, VIS_RUNNING, current_scores, current_boxes, current_kf

    if im is None:
        im = current_kf

    # DETECT_RUNNING = True
    print("starting object detection")
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    current_scores, current_boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, current_boxes.shape[0])

    NEW_DETECTION = True

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

def cb_keyframe_received(msg, net=None):
    global current_kf, current_kf_id, min_c, max_c, VIS_RUNNING, NEW_DETECTION
    print("Keyframe {} received".format(msg.id))
    if VIS_RUNNING:
        print ("Visualization of last KF still running, can not detect any more objects and will ignore this keyframe")
        return
    if NEW_DETECTION:
        print ("Object detection for last KF still running, can not detect any more objects and will ignore this keyframe")
        return
    # print("time: " + str(msg.time))
    # print("isKeyframe: " + str(msg.isKeyframe))
    # print("camToWorld: " + str(msg.camToWorld))
    # print("fx, fy, cx, cy: " + str(msg.fx) + ", " + str(msg.fy) + ", " + str(msg.cx) + ", " + str(msg.cy))
    # print("height, width: " + str(msg.height) + ", " + str(msg.width))
    # print("pointcloud: ...")

    structSizeInputPointDense = 12
    height = 480
    width = 640
    channels = 3  # 4 channels should mean that the data comes in CMYK format from LSD_SLAM
    im_shape = (height, width, channels)
    num_points = width * height
    timer = Timer()
    timer.tic()
    im = np.zeros(shape=im_shape, dtype=np.uint8)

    fmt = "<ffBBBB"
    for p in range(num_points):
        (d, v, c, m, y, k) = struct.unpack_from(fmt, msg.pointcloud, p*structSizeInputPointDense)
        row = int(p % width)
        line = int(p / width)
        # r,g,b = cmyk_to_rgb(c,m,y,255)
        if c != m or c != k:
            print "c != m or c != k"

        # blue
        im[line, row, 0] = int(c)
        # green
        im[line, row, 1] = int(c)
        # red
        im[line, row, 2] = int(c)

    timer.toc()
    print("It took {} sec. to deserialize binary data from lsd_slam_keyframe_msg.".format(timer.total_time))
    current_kf = im
    current_kf_id = msg.id


    if net is not None:
        frame_detect(net)
        # plt.show()

    cv2.imwrite("kf_"+str(current_kf_id)+".png", current_kf)

    print("Waiting for next KF.")

def fake_detect(fname="kf_20542.png",net=None):
    global NEW_DETECTION, current_kf, current_kf_id
    NEW_DETECTION = True
    current_kf = cv2.imread(fname)
    current_kf_id = "fake_detect_frame"
    if net is not None:
        frame_detect(net)



if __name__ == '__main__':

    rospy.init_node("frcnn")

    print("node initialized")

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Not sure if this is necesary. Leaving it for now, bu should test later what the effect of this warmup is...
    # Warmup on a dummy image
    # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _= im_detect(net, im)


    sub_keyframes = rospy.Subscriber("/lsd_slam/keyframes", keyframeMsg , cb_keyframe_received,
                                     queue_size=1,callback_args=net)

    fake_detect(net=net)

    ctr = 0
    if True:
    # while True:
        # Visualize detections for each class
        time.sleep(0.5)
        if NEW_DETECTION:
            VIS_RUNNING = True
            for cls_ind, cls in enumerate(CLASSES[1:]):
                cls_ind += 1  # because we skipped background
                cls_boxes = current_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = current_scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                vis_detections(current_kf, cls, dets, thresh=CONF_THRESH)
                # break
            ctr += 1
            NEW_DETECTION = False
            VIS_RUNNING = False

    # plt.show()
