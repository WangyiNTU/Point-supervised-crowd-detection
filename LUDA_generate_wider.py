import os
import cv2
import cPickle
import numpy as np
import scipy
import scipy.spatial

############ locally-uniform distribution assumption (LUDA)##########
# change to your WIDER Face dataset folder ##########################
root_dir = '/media/wangyi/Data/BK_STLab_19-20/data/WIDER'
split = 'val' # train or val
########################################################


res_folder = 'data/cache/widerface'
if not os.path.exists(res_folder):
    os.makedirs(res_folder)
if split == 'train':
    img_path = os.path.join(root_dir, 'WIDER_train/images')
    anno_path = os.path.join(root_dir, 'wider_face_split','wider_face_train_bbx_gt.txt')
    res_path = os.path.join(res_folder, split)
else:
    img_path = os.path.join(root_dir, 'WIDER_val/images')
    anno_path = os.path.join(root_dir, 'wider_face_split','wider_face_val_bbx_gt.txt')
    res_path = os.path.join(res_folder, split)

image_data = []
valid_count = 0
img_count = 0
centers_count = 0
with open(anno_path, 'rb') as fid:
    lines = fid.readlines()
num_lines = len(lines)

index = 0
while index<num_lines:
    filename = lines[index].strip()
    img_count += 1
    if img_count%1000 == 0:
        print 'Processed' + str(img_count)
    num_obj = int(lines[index+1])
    filepath = os.path.join(img_path, filename)
    img = cv2.imread(filepath)
    img_height, img_width = img.shape[:2]
    centers = []
    if num_obj>0:
        for i in range(num_obj):
            info = lines[index+2+i].strip().split(' ')
            x1, y1 = max(int(info[0]), 0), max(int(info[1]), 0)
            w, h = min(int(info[2]), img_width - x1 - 1), min(int(info[3]), img_height - y1 - 1)
            c_x, c_y = x1 + w / 2., y1 + h / 2.
            if w>=5 and h>=5:
                # only save the central points of objects
                centers.append(np.array([c_x, c_y]))
    centers = np.array(centers)
    centers_count += len(centers)
    if len(centers)>0:
        valid_count += 1
        annotation = {}
        annotation['filepath'] = filepath
        # add scale in bboxes height
        tree = scipy.spatial.KDTree(centers.copy(), leafsize=1024)
        k = min(len(centers),3)
        if k <= 2:
            scale = max(img_height / (4. + k), 12)
            scale = np.ones(k)* scale
            scale_weight = 0.1 if k==1 else 1.
            scale_weight = np.ones(k)* scale_weight
        else:
            crowd_range = np.max(centers[:,1]) - np.min(centers[:,1]) # range: y_max - y_min
            circle_scale = crowd_range / 6.
            distances, distances_idx = tree.query(centers, k=len(centers))
            distances_mean = 0.5 * np.mean(distances[:,1:k],axis=1)
            places = np.where(distances <= circle_scale)
            unique, counts = np.unique(places[0], return_counts=True) # places[0]: row index
            take_d_places = dict(zip(unique, counts))
            scale = []
            scale_weight = []
            for key,value in take_d_places.iteritems():
                # value = 1
                idx_in_circle = distances_idx[key, :value]
                # distances_mean_filtered = removeOutliers(distances_mean[idx_in_circle])
                # s_p = (1 - (1./len(boxes))**0.5) * np.mean(distances_mean_filtered) + (1./len(boxes))**0.5 * max(crowd_range / 8., 12)
                s_p = np.mean(distances_mean[idx_in_circle])
                s_p = np.clip(s_p,8,None) #limit min size
                scale.append(s_p)
                scale_weight.append(value)
            scale = np.array(scale)
            scale_weight = np.array(scale_weight)
            # scale = repelce_Outliers(scale, distances_idx)

        boxes_with_scale = np.zeros((len(centers), 4))
        boxes_with_scale[:, 0], boxes_with_scale[:, 2] = centers[:, 0] - scale / 2., centers[:, 0] + scale / 2.
        boxes_with_scale[:, 1], boxes_with_scale[:, 3] = centers[:,1] - scale/2., centers[:,1] + scale/2.
        boxes_with_scale[:, 0:4:2] = np.clip(boxes_with_scale[:, 0:4:2], 0, img_width - 1)
        boxes_with_scale[:, 1:4:2] = np.clip(boxes_with_scale[:, 1:4:2], 0, img_height - 1)
        annotation['bboxes'] = boxes_with_scale
        annotation['confs'] = 0.6 * np.ones((boxes_with_scale.shape[0]))
        annotation['w_bboxes'] = scale_weight
        image_data.append(annotation)

    if len(centers)==0 and split=='train':
        num_obj += 1
    index += (2+num_obj)

print '{} images and {} valid images and {} boxes'.format(img_count, valid_count,centers_count)
with open(res_path, 'wb') as fid:
    cPickle.dump(image_data, fid, cPickle.HIGHEST_PROTOCOL)
