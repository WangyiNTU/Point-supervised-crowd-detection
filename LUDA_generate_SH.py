import os
import cv2
import cPickle
import numpy as np
import scipy
import scipy.spatial
#used to import .mat file
import scipy.io

############ locally-uniform distribution assumption (LUDA)##########
# change to your ShanghaiTech dataset folder ##########################
### set input dataset path for training or val--part A or B
part = 'part_A' # part_A or part_B
split = 'train' # train or train_val or test
root_dir = os.path.join('/home/wangyi/data', part)

if part == 'part_A':
    res_dataset = 'ShanghaiTechA'
    range_split = 10.
    ratio = 1.
    max_sp = None
elif part == 'part_B':
    res_dataset = 'ShanghaiTechB'
    range_split = 8.
    ratio = 0.5
    max_sp = 50

#train:
if split == 'train':
    #directory of images for training
    img_path = os.path.join(root_dir, 'train_data/images')
    #directory of annotation info          -use .mat
    anno_path = os.path.join(root_dir, 'train_data/ground-truth','')
    # anno_path = os.path.join(root_dir, 'wider_face_split','wider_face_test_filelist.txt')
#train_val:
if split == 'train_val':
    #directory of images for training
    img_path = os.path.join(root_dir, 'val_data/images')
    #directory of annotation info          -use .mat
    anno_path = os.path.join(root_dir, 'val_data/ground-truth','')

elif split == 'test':
    img_path = os.path.join(root_dir, 'test_data/images')
    anno_path = os.path.join(root_dir, 'test_data/ground-truth','')

if not os.path.exists(os.path.join('data/cache', res_dataset)):
    os.makedirs(os.path.join('data/cache', res_dataset))

res_path = os.path.join('data/cache', res_dataset, split)


img_names=[x for x in os.listdir(img_path)]    #read all filenames in image folder
image_data = []               #will be used at the end--to write files
weights_list = []
scales = []
valid_count = 0          #number of valid images--with at least 1 person
img_count = 0           #number of all images
box_count = 0            #number of all heads in one image
for img_name in img_names:      #read all images
    img_count += 1              #record the total number of imgs
    if img_count % 20 == 0:
        print img_count
    full_img_path = os.path.join(img_path,img_name)   #full image path
    img=cv2.imread(full_img_path)                   #read imgs
    img_height, img_width = img.shape[:2]           #img-height=row of the img;img_width=colums of the img
    mat_path = img_name.replace('jpg','mat')
    data=scipy.io.loadmat(os.path.join(anno_path,'GT_'+mat_path))  #find path of .mat file
    centers=data['image_info'][0][0][0][0][0].astype(np.float32)   #read all the centers in the image

    box_count_image = centers.shape[0]     #number of boxes(targets) in each img
    box_count += box_count_image           #number of boxes in the whole dataset

    if box_count_image>0:                #judge wether it is a valid img:if no less than 1 box->valid
        valid_count += 1
        annotation = {}                  #to restore annotation info
        annotation['filepath'] = full_img_path             #to restore full imgpath into the annotation[]
        # add scale in bboxes height
        #KD tree
        tree = scipy.spatial.KDTree(centers.copy(), leafsize=1024) #intialization of the tree
        k = min(box_count_image,3)              #for each box, find the nearest k-1 boxes around it(it itself is included in k->so k-1)
        if k <= 2:                                          #need parameters-img_height,centers,len(boxes)
            scale = max(img_height / (4. + k), 12)
            scale = np.ones(k)* scale
            scale_weight = 0.1 if k==1 else 1.
            scale_weight = np.ones(k)* scale_weight
        else:
            crowd_range = np.max(centers[:,1]) - np.min(centers[:,1]) # range: y_max - y_min;for the whole img,find the gap of y between the highest box and the lowest box
            circle_scale = crowd_range / range_split                          #initialization of scale of local window
            distances, distances_idx = tree.query(centers, k=box_count_image//2)  #distances:Euclinear distance between box X and others(by order from near to far)
            #distances_idx is the coresponding index of these distances
            distances_mean = ratio * np.mean(distances[:,1:k],axis=1)      #mean of distances
            places = np.where(distances <= circle_scale)     #how many boxes within circle_scale
            unique, counts = np.unique(places[0], return_counts=True) # places[0]: row index
            #counts:how many boxes within circle_scale
            take_d_places = dict(zip(unique, counts))    #zip:make (unique,counts)
            scale = []
            scale_weight = []
            for key,value in take_d_places.iteritems():
                idx_in_circle = distances_idx[key, :value]
                s_p = np.mean(distances_mean[idx_in_circle])
                s_p = np.clip(s_p,2,max_sp)         #set the number of s_P to be 2-infinite
                scale.append(s_p)                 #scale->s_P
                scale_weight.append(value)
            scale = np.array(scale)
            scale_weight = np.array(scale_weight)

        weights_list.extend(list(scale_weight))
        scales.extend(list(scale))
        boxes_with_scale = np.zeros((box_count_image,4),dtype=np.float32)         #initialization of boxes_with_scale,datatype:int64
        boxes_with_scale[:, 0], boxes_with_scale[:, 2] = centers[:, 0] - scale / 2., centers[:, 0] + scale / 2. #x1, x2
        boxes_with_scale[:, 1], boxes_with_scale[:, 3] = centers[:,1] - scale/2., centers[:,1] + scale/2. #y1, y2
        boxes_with_scale[:, 0:4:2] = np.clip(boxes_with_scale[:, 0:4:2], 0, img_width - 1)
        boxes_with_scale[:, 1:4:2] = np.clip(boxes_with_scale[:, 1:4:2], 0, img_height - 1)
        annotation['bboxes'] = boxes_with_scale                                          #store results in annotation[]
        annotation['confs'] = 0.6 * np.ones((boxes_with_scale.shape[0]))
        annotation['w_bboxes'] = scale_weight
        image_data.append(annotation)

weights = np.array(weights_list,dtype='float')        #for testing:record weights_max,weights_mean
print('weights_max: {}'.format(weights.max()))
print('weights_mean: {}'.format(weights.mean()))
print('weights_std: {}'.format(weights.std()))

scales = np.array(scales,dtype='float')
print('scales_max: {}'.format(scales.max()))
print('scales_min: {}'.format(scales.min()))
print('scales_mean: {}'.format(scales.mean()))
print('scales_std: {}'.format(scales.std()))

for image_data_i in image_data:
    image_data_i['w_bboxes'] = np.clip(image_data_i['w_bboxes'], None, 50)

print '{} images and {} valid images and {} boxes'.format(img_count, valid_count,box_count)
with open(res_path, 'wb') as fid:                                                    #to write a file in the res_path
    cPickle.dump(image_data, fid, cPickle.HIGHEST_PROTOCOL)