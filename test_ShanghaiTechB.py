from __future__ import division

import os
import time
import cPickle
from keras.layers import Input, MaxPooling2D
from keras.models import Model, Sequential
from keras_csp import config, bbox_process
from keras_csp.utilsfunc import *
from keras.utils import generic_utils
from keras_csp.nms_wrapper import nms

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
C = config.Config()
C.offset = False
C.scale = 'h'
C.num_scale = 1
C.down = 1
cache_path = 'data/cache/ShanghaiTechB/test'
with open(cache_path, 'rb') as fid:
    val_data = cPickle.load(fid)
num_imgs = len(val_data)
print 'num of val samples: {}'.format(num_imgs)

C.size_test = [0, 0]
input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)

score = 0.2
soft_box_thre = 0.3

limit = 3.5
# define the base network
from keras_csp.resnet50_full_res import nn_p3p4p5_fpn as network

# define the network prediction
preds = network(img_input, offset=C.offset, num_scale=C.num_scale, trainable=False)
model = Model(img_input, preds)

nms_max = Sequential()
nms_max.add(MaxPooling2D(pool_size=(3, 3), strides=1,padding='same',input_shape=(None, None, 1)))

if C.offset:
    w_path = 'output/valmodels/ShanghaiTechB/%s/off' % (C.scale)
    out_path = 'output/valresults/ShanghaiTechB/%s/off' % (C.scale)
else:
    w_path = 'output/valmodels/ShanghaiTechB/%s/nooff' % (C.scale)
    out_path = 'output/valresults/ShanghaiTechB/%s/nooff' % (C.scale)
if not os.path.exists(out_path):
    os.makedirs(out_path)

history = load_json(w_path + '/history.json')
path_best_model = history["path_best_model"]
print 'load weights from {}'.format(path_best_model)
model.load_weights(path_best_model, by_name=True)
res_path = os.path.join(out_path, str(history["best_epoch"]))
if not os.path.exists(res_path):
    os.makedirs(res_path)
print res_path

progbar = generic_utils.Progbar(num_imgs)
start_time = time.time()
for f in range(num_imgs):
    filepath = val_data[f]['filepath']
    filename = filepath.split('/')[-1].split('.')[0]
    txtpath = os.path.join(res_path, filename + '.txt')
    if os.path.exists(txtpath):
        continue

    img = cv2.imread(filepath)

    def detect_face(img, scale=1, flip=False):
        img_h, img_w = img.shape[:2]
        img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(np.ceil(scale * img_w / 16) * 16)
        scale_h, scale_w = img_h_new / img_h, img_w_new / img_w

        img_s = cv2.resize(img, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
        # img_h, img_w = img_s.shape[:2]
        # print frame_number
        C.size_test[0] = img_h_new
        C.size_test[1] = img_w_new

        if flip:
            img_sf = cv2.flip(img_s, 1)
            # x_rcnn = format_img_pad(img_sf, C)
            x_rcnn = format_img(img_sf, C)
        else:
            # x_rcnn = format_img_pad(img_s, C)
            x_rcnn = format_img(img_s, C)
        Y = model.predict(x_rcnn)
        Y_max = nms_max.predict(Y[0])
        keep = (Y_max == Y[0])
        Y[0] = Y[0] * keep
        if C.offset:
            boxes = bbox_process.parse_shanghai_h_offset(Y, C, score=score, down=C.down, nmsthre=0.4)
        else:
            boxes = bbox_process.parse_shanghai_h_nooff(Y, C, score=score, down=C.down, nmsthre=0.4)
        if len(boxes) > 0:
            keep_index = np.where(np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= limit)[0]
            boxes = boxes[keep_index, :]
        if len(boxes) > 0:
            if flip:
                boxes[:, [0, 2]] = img_s.shape[1] - boxes[:, [2, 0]]
            boxes[:, 0:4:2] = boxes[:, 0:4:2] / scale_w
            boxes[:, 1:4:2] = boxes[:, 1:4:2] / scale_h
        else:
            boxes = np.empty(shape=[0, 5], dtype=np.float32)
        return boxes

    def im_det_ms_pyramid(image, max_im_shrink):
        # shrink detecting and shrink only detect big face
        det_s = np.row_stack((detect_face(image, 0.5), detect_face(image, 0.5, flip=True)))
        index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 64)[0]
        det_s = det_s[index, :]

        det_temp = np.row_stack((detect_face(image, 0.75), detect_face(image, 0.75, flip=True)))
        index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 42)[0]
        det_temp = det_temp[index, :]
        det_s = np.row_stack((det_s, det_temp))
        #
        det_temp = np.row_stack((detect_face(image, 0.25), detect_face(image, 0.25, flip=True)))
        index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 128)[0]
        det_temp = det_temp[index, :]
        det_s = np.row_stack((det_s, det_temp))

        # st = [1.25, 1.5, 1.75, 2.0, 2.25]
        st = [1.5, 2.0]
        for i in range(len(st)):
            if (st[i] <= max_im_shrink):
                det_temp = np.row_stack((detect_face(image, st[i]), detect_face(image, st[i], flip=True)))
                # Enlarged images are only used to detect small faces.
                # if st[i] == 1.25:
                #     index = np.where(
                #         np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 128)[0]
                #     det_temp = det_temp[index, :]
                if st[i] == 1.5:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 12)[0]
                    det_temp = det_temp[index, :]
                # elif st[i] == 1.75:
                #     index = np.where(
                #         np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 64)[0]
                #     det_temp = det_temp[index, :]
                elif st[i] == 2.0:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 8)[0]
                    det_temp = det_temp[index, :]
                # elif st[i] == 3.0:
                #     index = np.where(
                #         np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 16)[0]
                #     det_temp = det_temp[index, :]
                det_s = np.row_stack((det_s, det_temp))
        return det_s

    max_im_shrink = (0x7fffffff / 577.0 / (img.shape[0] * img.shape[1])) ** 0.5  # the max size of input image
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    # print('max_im_shrink:', max_im_shrink)
    det0 = detect_face(img)
    det1 = detect_face(img, flip=True)
    # det = np.row_stack((det0, det1))
    det2 = im_det_ms_pyramid(img, max_im_shrink)
    # merge all test results via bounding box voting
    det = np.row_stack((det0, det1, det2))
    keep_index = np.where(np.minimum(det[:, 2] - det[:, 0], det[:, 3] - det[:, 1]) >= 2)[0] # >= 3
    dets = det[keep_index, :]
    dets = bbox_process.soft_bbox_vote(dets, thre=soft_box_thre, score=score)
    # keep = nms(dets, 0.4, usegpu=True, gpu_id=0)
    # dets = det[keep, :]

    keep_index = np.where((dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1) >= 2 ** 2)[0]
    dets = dets[keep_index, :]

    with open(txtpath, 'w') as file:
        file.write('{:s}\n'.format(filename))
        file.write('{:d}\n'.format(len(dets)))
        for line in dets:
            file.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.3f}\n'.
                    format(line[0], line[1], line[2] - line[0] + 1, line[3] - line[1] + 1, line[4]))

    if f % 10 == 0:
        progbar.update(f)
print time.time() - start_time