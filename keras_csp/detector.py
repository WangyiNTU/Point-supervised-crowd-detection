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

score = 0.2
soft_box_thre = 0.3

limit = 3.5
nms_max = Sequential()
nms_max.add(MaxPooling2D(pool_size=(3, 3), strides=1,padding='same',input_shape=(None, None, 1)))

def detect_main(model, val_data, C):
    C.size_test = [0, 0]
    num_imgs = len(val_data)
    print 'num of val samples: {}'.format(num_imgs)
    meta = {}
    meta['filepath'] = []
    meta['true_box'] = []
    meta['pred_box'] = []
    progbar = generic_utils.Progbar(num_imgs)
    start_time = time.time()
    for f in range(num_imgs):
        filepath = val_data[f]['filepath']
        # filename = filepath.split('/')[-1].split('.')[0]
        img = cv2.imread(filepath)
        max_im_shrink = (0x7fffffff / 577.0 / (img.shape[0] * img.shape[1])) ** 0.5  # the max size of input image
        det0 = detect_face(model, img, C)
        det1 = detect_face(model, img, C, flip=True)
        det2 = im_det_ms_pyramid(model, img, C, max_im_shrink)
        det = np.row_stack((det0, det1, det2))
        keep_index = np.where(np.minimum(det[:, 2] - det[:, 0], det[:, 3] - det[:, 1]) >= 2)[0]  # >= 3
        dets = det[keep_index, :]
        dets = bbox_process.soft_bbox_vote(dets, thre=soft_box_thre, score=score)

        keep_index = np.where((dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1) >= 2 ** 2)[0]
        dets = dets[keep_index, :]

        gt_bboxes = val_data[f]['bboxes']

        meta['filepath'].append(filepath)
        meta['true_box'].append(dets)
        meta['pred_box'].append(gt_bboxes)

        if f % 10 == 0:
            progbar.update(f)

    best_mae = 1e8
    best_mse = 1e8
    best_thr = 0.4
    mae_dict = {}
    mse_dict = {}
    for i in range(30,51):
        thr_i = i * 0.01
        pred_counts = []
        true_counts = []
        for true_box, pred_box in zip(meta['true_box'],meta['pred_box']):
            true_box = true_box[true_box[:, 4] > thr_i]
            pred_count = true_box.shape[0]
            true_count = pred_box.shape[0]
            pred_counts.append(pred_count)
            true_counts.append(true_count)

        pred_counts = np.array(pred_counts, dtype='float')
        true_counts = np.array(true_counts, dtype='float')

        mae = np.mean(np.abs(true_counts - pred_counts))
        mse = np.sqrt(np.mean((true_counts - pred_counts)**2))

        mae_dict['{:0.2f}'.format(thr_i)] = float(mae)
        mse_dict['{:0.2f}'.format(thr_i)] = float(mse)

        if mae <= best_mae:
            best_mae = float(mae)
            best_mse = float(mse)
            best_thr = float(thr_i)

    print time.time() - start_time

    return best_mae, best_mse, best_thr, mae_dict, mse_dict

def detect_face(model, img, C, scale=1, flip=False):
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
    Y = model.predict_on_batch(x_rcnn)
    Y_max = nms_max.predict_on_batch(Y[0])
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


def im_det_ms_pyramid(model, image, C, max_im_shrink):
    # shrink detecting and shrink only detect big face
    det_s = np.row_stack((detect_face(model, image, C, 0.5), detect_face(model, image, C, 0.5, flip=True)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 64)[0]
    det_s = det_s[index, :]

    det_temp = np.row_stack((detect_face(model, image, C, 0.75), detect_face(model, image, C, 0.75, flip=True)))
    index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 42)[0]
    det_temp = det_temp[index, :]
    det_s = np.row_stack((det_s, det_temp))
    #
    det_temp = np.row_stack((detect_face(model, image, C, 0.25), detect_face(model, image, C, 0.25, flip=True)))
    index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 128)[0]
    det_temp = det_temp[index, :]
    det_s = np.row_stack((det_s, det_temp))

    # st = [1.25, 1.5, 1.75, 2.0, 2.25]
    st = [1.5, 2.0]
    for i in range(len(st)):
        if (st[i] <= max_im_shrink):
            det_temp = np.row_stack((detect_face(model, image, C, st[i]), detect_face(model, image, C, st[i], flip=True)))
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