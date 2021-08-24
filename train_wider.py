from __future__ import division
import random
import sys, os
import time
import numpy as np
import cPickle
from keras.utils import generic_utils
from keras.utils import GeneratorEnqueuer
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_csp import config, data_generators
from keras_csp import losses as losses
from keras_csp.utilsfunc import load_json, save_json

# get the config parameters
C = config.Config()
C.gpu_ids = '0,1,2'
num_gpu = len(C.gpu_ids.split(','))
C.onegpu = 4
C.size_train = (704,704)
C.init_lr = 2e-5 * (num_gpu/8.)
C.offset = True
C.scale = 'h'
C.num_scale = 1
C.num_epochs = 200
C.iter_per_epoch = 12000#4000
batchsize = C.onegpu * num_gpu
os.environ["CUDA_VISIBLE_DEVICES"] = C.gpu_ids

C.path_history = None #'output/valmodels/wider/%s/off' % (C.scale)
C.update_thr = 0.6 #0.6 for ours, 1.0 for no-update
C.update_weights_thr = 10
# get the training data
cache_path = 'data/cache/widerface/train'
cache_path_updated = 'data/cache/widerface/train_updated'

# define the base network (resnet here, can be MobileNet, etc)
if C.network=='resnet50':
    from keras_csp import resnet50 as nn
    weight_path = 'data/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

input_shape_img = (C.size_train[0], C.size_train[1], 3)
img_input = Input(shape=input_shape_img)
# define the network prediction
preds = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)

model = Model(img_input, preds)
if num_gpu>1:
    from keras_csp.parallel_model import ParallelModel
    model = ParallelModel(model, int(num_gpu))

if C.path_history is not None:
    history = load_json(C.path_history + '/history.json')
    path_model = C.path_history + '/net_last.hdf5' #history["path_last_model"]
    model.load_weights(path_model, by_name=True)
    print 'load weights from {}'.format(path_model)
    # model_tea.load_weights(path_model, by_name=True)
    s_epoch = history["train"][-1]["epoch"]
    best_loss = history["best_loss"]
    history["Config"] = {k:v for k, v in C.__dict__.items() if not (k.startswith('__') and k.endswith('__'))}
    out_path = C.path_history
    history["cache_path"] = cache_path_updated
    cache_path = cache_path_updated+str(s_epoch)
    print("Resuming epoch...{}".format(s_epoch))

else:
    history = {"train": [],
               "Config": {k:v for k, v in C.__dict__.items() if not (k.startswith('__') and k.endswith('__'))},
               "cache_path": cache_path_updated,
               "path_last_model": None,
               "path_best_model": None,
               "best_epoch": 1,
               "best_loss": np.Inf}
    s_epoch = 0
    best_loss = np.Inf
    model.load_weights(weight_path, by_name=True)
    # model_tea.load_weights(weight_path, by_name=True)
    print 'load weights from {}'.format(weight_path)
    print("Starting from scratch...")

    if C.offset:
        out_path = 'output/valmodels/wider/%s/off' % (C.scale)
    else:
        out_path = 'output/valmodels/wider/%s/nooff' % (C.scale)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    C.path_history = out_path
    cache_path = 'data/cache/widerface/train'

with open(cache_path, 'rb') as fid:
    train_data = cPickle.load(fid)
print 'load cache_path from {}'.format(cache_path)
num_imgs_train = len(train_data)
print 'num of training samples: {}'.format(num_imgs_train)
data_gen_train = data_generators.get_data_wider_class(train_data, C, batchsize=batchsize)
data_generators_thread = data_generators.ThreadingBatches(data_gen_train)

optimizer = Adam(lr=C.init_lr)
if C.offset:
    model.compile(optimizer=optimizer, loss=[losses.cls_center, losses.regr_h, losses.regr_offset])
else:
    model.compile(optimizer=optimizer, loss=[losses.cls_center, losses.regr_h])
model.metrics_tensors += model.outputs
model.metrics_names += ['predictions']
epoch_length = int(C.iter_per_epoch/batchsize)
iter_num = 0
add_epoch = 0
losses = np.zeros((epoch_length, 3))

print('Starting training with lr {} and alpha {}'.format(C.init_lr, C.alpha))
start_time = time.time()
total_loss_r, cls_loss_r1, regr_loss_r1, offset_loss_r1 = [], [], [], []
updata_cache = [2, 50, 100, 150]
for epoch_num in range(s_epoch, C.num_epochs):
    # updata cache at epoch 2, 50, 100, 150, may be not necessary
    if epoch_num in updata_cache:
        with open(cache_path_updated + str(epoch_num), 'rb') as fid:
            train_data = cPickle.load(fid)
        data_gen_train = data_generators.get_data_wider_class(train_data, C, batchsize=batchsize)
        data_generators_thread = data_generators.ThreadingBatches(data_gen_train)
        print('updating the data_generators_thread')
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1 + add_epoch, C.num_epochs + C.add_epoch))
    while True:
        try:
            train_dict = {}
            # X, Y = next(data_gen_train)
            X, Y, gt_ind_map, meta = data_generators_thread.popNextBatch()
            loss_s1 = model.train_on_batch(X, Y)

            gt_ct_ind = np.where(Y[0][:, :, :, -1] == 1)
            hms = loss_s1[4][..., 0]
            hms = hms[gt_ct_ind]
            gt_weights_map = Y[1][:, :, :, -1]
            gt_weights_map = gt_weights_map[gt_ct_ind]
            # ind = np.where(np.logical_and(hms > C.update_thr, gt_weights_map < C.update_weights_thr))
            ind = np.where(hms > C.update_thr)
            if ind[0].shape[0] > 0:
                height_map = np.exp(loss_s1[5][:, :, :, 0]) * C.down
                height_map = height_map[gt_ct_ind][ind]
                gt_ind_map = gt_ind_map[gt_ct_ind][ind]
                hms = hms[ind]

                file_and_ratios = [meta[x] for x in gt_ct_ind[0][ind]]
                modify_img_data = [[file_and_ratio[0], gt_box_ind, height / file_and_ratio[1], file_and_ratio[2], hm] for
                                   file_and_ratio, gt_box_ind, height, hm in zip(file_and_ratios, gt_ind_map, height_map, hms)]
                data_generators_thread.revise_ped_data(modify_img_data)

            losses[iter_num, 0] = loss_s1[1]
            losses[iter_num, 1] = loss_s1[2]
            if C.offset:
                losses[iter_num, 2] = loss_s1[3]
            else:
                losses[iter_num, 2] = 0

            iter_num += 1
            if iter_num % 20 == 0:
                progbar.update(iter_num,
                               [('cls', np.mean(losses[:iter_num, 0])), ('regr_h', np.mean(losses[:iter_num, 1])), ('offset', np.mean(losses[:iter_num, 2]))])
            if iter_num == epoch_length:
                cls_loss1 = np.mean(losses[:, 0])
                regr_loss1 = np.mean(losses[:, 1])
                offset_loss1 = np.mean(losses[:, 2])
                total_loss = cls_loss1+regr_loss1+offset_loss1

                total_loss_r.append(total_loss)
                cls_loss_r1.append(cls_loss1)
                regr_loss_r1.append(regr_loss1)
                offset_loss_r1.append(offset_loss1)
                print('Total loss: {}'.format(total_loss))
                print('Elapsed time: {}'.format(time.time() - start_time))

                iter_num = 0
                start_time = time.time()

                train_dict["total_loss"] = total_loss
                train_dict["cls_loss"] = cls_loss1
                train_dict["regr_loss"] = regr_loss1
                train_dict["offset_loss"] = offset_loss1
                train_dict["epoch"] = epoch_num + 1 + add_epoch
                history["train"] += [train_dict]

                if total_loss < best_loss:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss, total_loss))
                    best_loss = total_loss
                    history["best_epoch"] = epoch_num + 1 + add_epoch
                    history["best_loss"] = best_loss
                    history["path_best_model"] = os.path.join(out_path,'net_e{}_l{}.hdf5'.format(epoch_num + 1 + add_epoch,total_loss))
                    if epoch_num + 1 > 10:
                        model.save_weights(os.path.join(out_path, 'net_e{}_l{}.hdf5'.format(epoch_num + 1 + add_epoch, total_loss)))
                model.save_weights(os.path.join(out_path, 'net_last.hdf5'))
                history["path_last_model"] = os.path.join(out_path, 'net_e{}_l{}.hdf5'.format(epoch_num + 1 + add_epoch, total_loss))
                save_json(C.path_history + '/history.json', history)
                data_generators_thread.save_ped_data(cache_path_updated + str(epoch_num + 1 + add_epoch))
                break
        except Exception as e:
            print ('Exception: {}'.format(e))
            continue
print('Training complete, exiting.')
