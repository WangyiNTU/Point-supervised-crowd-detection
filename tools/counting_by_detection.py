from __future__ import division
import os
import time
import cPickle

import sys
sys.path.insert(0,'.')
from keras_csp import config, bbox_process
from keras_csp.utilsfunc import *
from keras.utils import generic_utils

# def find_class_threshold(f, iters, test_funcs, network, splits=10, beg=0.0, end=0.3):
#     for li_idx in range(iters):
#         avg_errors = []
#         threshold = list(np.arange(beg, end, (end - beg) / splits))
#         log(f, 'threshold:'+str(threshold))
#         for class_threshold in threshold:
#             avg_error = test_lsccnn(test_funcs, dataset, 'test_valid', network, True, thresh=class_threshold)
#             avg_errors.append(avg_error[0]['new_mae'])
#             log(f, "class threshold: %f, avg_error: %f" % (class_threshold, avg_error[0]['new_mae']))
#
#         mid = np.asarray(avg_errors).argmin()
#         beg = threshold[max(mid - 2, 0)]
#         end = threshold[min(mid + 2, splits - 1)]
#     log(f, "Best threshold: %f" % threshold[mid])
#     optimal_threshold = threshold[mid]
#     return optimal_threshold

def test(val_data, detections_path, out_path, thr):
    metrics_test = {}
    meta = {}
    meta['filepath'] = []
    meta['true_count'] = []
    meta['pred_count'] = []
    progbar = generic_utils.Progbar(len(val_data))
    txtpath = os.path.join(out_path, str(thr) + '_count.txt')
    start_time = time.time()
    for f in range(len(val_data)):
        filepath = val_data[f]['filepath']

        if 'WIDER' in filepath:
            event = filepath.split('/')[-2]
        else:
            event = ''
        filename = filepath.split('/')[-1].split('.')[0]

        gt_bboxes = val_data[f]['bboxes']
        true_count = gt_bboxes.shape[0]

        file_detections_path = os.path.join(detections_path, event)
        pred_txtpath = os.path.join(file_detections_path, filename + '.txt')
        with open(pred_txtpath, 'rb') as fid:
            lines = fid.readlines()
        bboxes = []
        for line in lines[2:]:
            bboxes.append([float(i) for i in line.split()])
        bboxes = np.array(bboxes, dtype='float32')
        if bboxes.shape[0] == 0:
            meta['filepath'].append(filepath)
            meta['true_count'].append(true_count)
            meta['pred_count'].append(bboxes.shape[0])
            continue

        bboxes = bboxes[bboxes[:, 4] > thr]
        pred_count = bboxes.shape[0]

        meta['filepath'].append(filepath)
        meta['true_count'].append(true_count)
        meta['pred_count'].append(pred_count)
        if f % 50 == 0:
            progbar.update(f)

    with open(txtpath, 'w') as file:
        for filename, true_c, pred_c in zip(meta['filepath'], meta['true_count'], meta['pred_count']):
            file.write('{:s}\n'.format(filename))
            file.write('True_count: {:d}, Pred_count: {:d}\n'.format(true_c, pred_c))

    true_counts = np.array(meta['true_count'], dtype='float')
    pred_counts = np.array(meta['pred_count'], dtype='float')

    metrics_test['mae'] = np.mean(np.abs(true_counts - pred_counts))
    metrics_test['mse'] = np.sqrt(np.mean((true_counts - pred_counts)**2))
    mask = true_counts > 0
    metrics_test['nae'] = np.mean(np.abs(true_counts[mask] - pred_counts[mask])/true_counts[mask])
    print '\nMAE: {:.3f}, MSE: {:.3f}, NAE: {:.3f}'.format(metrics_test['mae'], metrics_test['mse'], metrics_test['nae'])

    max_ind = int(np.argmax(np.abs(true_counts - pred_counts)))

    print('File with Max MAE: {}'.format(meta['filepath'][max_ind]))

    with open(txtpath, 'a+') as file:
        file.write('MAE: {:.3f}, MSE: {:.3f}, NAE: {:.3f}\n'.format(metrics_test['mae'], metrics_test['mse'], metrics_test['nae']))
        file.write('File with Max MAE: {}\n'.format(meta['filepath'][max_ind]))
        file.write('Threshold: {:.3f}'.format(thr))

    print time.time() - start_time

if __name__ == '__main__':
    C = config.Config()

    part = 'part_A'  # part_A, part_B, wider

    C.offset = True
    if part == 'part_A':
        res_dataset = 'ShanghaiTechA'
        C.offset = False
        threshold = 0.41 # according to the best val's thr
        epoch = '32' # according to the best val's epoch
        split = 'test'
    elif part == 'part_B':
        res_dataset = 'ShanghaiTechB'
        C.offset = False
        threshold = 0.42 # according to the best val's thr
        epoch = '32' # according to the best val's epocH
        split = 'test'
    elif part == 'wider':
        res_dataset = 'wider'
        threshold = 0.4
        epoch = '197'
        split = 'val'

    C.scale = 'h'
    C.num_scale = 1
    cache_path = os.path.join('data/cache', res_dataset, split)
    if part == 'wider':
        cache_path = os.path.join('data/cache', 'widerface', split)
    # cache_path = os.path.join('output/valmodels/wider/%s/off' % (C.scale), 'exp/revise_w_R6_197ep/train_updated')
    with open(cache_path, 'rb') as fid:
        val_data = cPickle.load(fid)
    num_imgs = len(val_data)
    print 'num of val samples: {}'.format(num_imgs)

    if C.offset:
        detections_path = os.path.join('output/valresults', res_dataset, '%s/off' % (C.scale), str(epoch))
        out_path = os.path.join('output/valcount', res_dataset, '%s/off' % (C.scale), str(epoch))
    else:
        detections_path = os.path.join('output/valresults', res_dataset, '%s/nooff' % (C.scale), str(epoch))
        out_path = os.path.join('output/valcount', res_dataset, '%s/nooff' % (C.scale), str(epoch))

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print out_path

    test(val_data, detections_path, out_path, threshold)