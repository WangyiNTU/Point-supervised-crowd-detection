from __future__ import division
import os
import time
import cPickle
import cv2
import numpy as np

from keras_csp import config
# from keras_csp.utilsfunc import *
from keras.utils import generic_utils

## t threshold and epoch
thr = 0.4
epoch = '197'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
C = config.Config()
C.offset = True
C.scale = 'h'
C.num_scale = 1
cache_path = 'data/cache/widerface/val'
# cache_path = os.path.join('output/valmodels/wider/%s/off' % (C.scale), 'exp/revise_w_R6_197ep/train_updated')
with open(cache_path, 'rb') as fid:
    val_data = cPickle.load(fid)
num_imgs = len(val_data)
print 'num of val samples: {}'.format(num_imgs)

C.size_test = [0, 0]
detections_path = os.path.join('output/valresults/wider/%s/off' % (C.scale), str(epoch))

out_path = os.path.join('output/valviews/wider/%s/off' % (C.scale), str(epoch))
if not os.path.exists(out_path):
    os.makedirs(out_path)
print out_path

progbar = generic_utils.Progbar(num_imgs)
start_time = time.time()
for f in range(num_imgs):
    filepath = val_data[f]['filepath']
    event = filepath.split('/')[-2]
    event_path = os.path.join(out_path, event)
    if not os.path.exists(event_path):
        os.mkdir(event_path)
    filename = filepath.split('/')[-1].split('.')[0]
    jpgpath = os.path.join(event_path, filename + '.jpg')
    if os.path.exists(jpgpath):
        continue

    img = cv2.imread(filepath)

    gt_bboxes = val_data[f]['bboxes'].astype(np.int)
    for i in range(gt_bboxes.shape[0]):
        x1, y1 = gt_bboxes[i, 0], gt_bboxes[i, 1]
        x2, y2 = gt_bboxes[i, 2], gt_bboxes[i, 3]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    file_detections_path = os.path.join(detections_path, event)
    txtpath = os.path.join(file_detections_path, filename + '.txt')
    with open(txtpath, 'rb') as fid:
        lines = fid.readlines()

    bboxes = []
    for line in lines[2:]:
        bboxes.append([float(i) for i in line.split()])
    bboxes = np.array(bboxes, dtype='float32')
    bboxes = bboxes[bboxes[:,4] > thr]
    for i in range(bboxes.shape[0]):
        x1, y1 = int(bboxes[i,0]), int(bboxes[i,1])
        x2, y2 = x1 + int(bboxes[i,2]), y1 + int(bboxes[i,3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, '{:0.2f}'.format(bboxes[i,4]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(jpgpath, img)

    if f % 50 == 0:
        progbar.update(f)
print time.time() - start_time