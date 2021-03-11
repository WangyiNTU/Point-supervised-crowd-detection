from __future__ import absolute_import
from __future__ import division
# import numpy as np
# import cv2
import random
from . import data_augment
from .bbox_transform import *
import threading
import time
import cPickle
from keras_csp.utilsfunc import AverageMeter

def calc_gt_center(C, img_data,r=2, down=4,scale='h',offset=True):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 2))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			gau_map = np.multiply(dy, np.transpose(dx))
			seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)
			seman_map[y1:y2, x1:x2,1] = 1
			seman_map[c_y, c_x, 2] = 1

			if scale == 'h':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
			if offset:
				offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, 2] = 1

	if offset:
		return seman_map,scale_map,offset_map
	else:
		return seman_map, scale_map

def calc_gt_top(C, img_data,r=2):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 2))
	seman_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/4
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/4
		for ind in range(len(gts)):
			x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			w = x2-x1
			c_x = int((gts[ind, 0] + gts[ind, 2]) / 2)

			dx = gaussian(w)
			dy = gaussian(w)
			gau_map = np.multiply(dy, np.transpose(dx))

			ty = np.maximum(0,int(round(y1-w/2)))
			ot = ty-int(round(y1-w/2))
			seman_map[ty:ty+w-ot, x1:x2,0] = np.maximum(seman_map[ty:ty+w-ot, x1:x2,0], gau_map[ot:,:])
			seman_map[ty:ty+w-ot, x1:x2,1] = 1
			seman_map[y1, c_x, 2] = 1

			scale_map[y1-r:y1+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind,3]-gts[ind,1])
			scale_map[y1-r:y1+r+1, c_x-r:c_x+r+1, 1] = 1
	return seman_map,scale_map

def calc_gt_bottom(C, img_data, r=2):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	scale_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 2))
	seman_map = np.zeros((int(C.size_train[0]/4), int(C.size_train[1]/4), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/4
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/4
		for ind in range(len(gts)):
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			y2 = np.minimum(int(C.random_crop[0] / 4) - 1, y2)
			w = x2 - x1
			c_x = int((gts[ind, 0] + gts[ind, 2]) / 2)
			dx = gaussian(w)
			dy = gaussian(w)
			gau_map = np.multiply(dy, np.transpose(dx))

			by = np.minimum(int(C.random_crop[0]/4)-1, int(round(y2+w/2)))
			ob = int(round(y2+w/2))-by
			seman_map[by-w+ob:by, x1:x2, 0] = np.maximum(seman_map[by-w+ob:by, x1:x2, 0], gau_map[:w-ob, :])
			seman_map[by-w+ob:by, x1:x2, 1] = 1
			seman_map[y2, c_x, 2] = 1

			scale_map[y2-r:y2+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind,3]-gts[ind,1])
			scale_map[y2-r:y2+r+1, c_x-r:c_x+r+1, 1] = 1

	return seman_map,scale_map

def get_data(ped_data, C, batchsize = 8):
	current_ped = 0
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize:
			random.shuffle(ped_data)
			current_ped = 0
		for img_data in ped_data[current_ped:current_ped + batchsize]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=True)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_hybrid(ped_data, emp_data, C, batchsize = 8,hyratio=0.5):
	current_ped = 0
	current_emp = 0
	batchsize_ped = int(batchsize * hyratio)
	batchsize_emp = batchsize - batchsize_ped
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		if current_ped>len(ped_data)-batchsize_ped:
			random.shuffle(ped_data)
			current_ped = 0
		if current_emp>len(emp_data)-batchsize_emp:
			random.shuffle(emp_data)
			current_emp = 0
		for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))

			except Exception as e:
				print ('get_batch_gt:',e)
		for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
			try:
				img_data, x_img = data_augment.augment(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)
				else:
					if C.point == 'top':
						y_seman, y_height = calc_gt_top(C, img_data)
					elif C.point == 'bottom':
						y_seman, y_height = calc_gt_bottom(C, img_data)
					else:
						y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt_emp:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize_ped
		current_emp += batchsize_emp
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_wider(ped_data, C, batchsize = 8):
	current_ped = 0
	while True:
		x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
		meta = []
		if current_ped>len(ped_data)-batchsize:
			random.shuffle(ped_data)
			current_ped = 0
		for img_data in ped_data[current_ped:current_ped + batchsize]:
			try:
				img_data, x_img = data_augment.augment_wider_with_weights(img_data, C)
				if C.offset:
					y_seman, y_height, y_offset = calc_gt_center_weights(C, img_data, down=C.down, scale=C.scale, offset=True)
				else:
					y_seman, y_height = calc_gt_center(C, img_data,down=C.down, scale=C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))
				meta.append({'filepath': img_data['filepath'], 'keep_inds': img_data['keep_inds'], 'ratio': img_data['ratio']})
				if C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
			except Exception as e:
				print ('get_batch_gt:',e)
		x_img_batch = np.concatenate(x_img_batch,axis=0)
		y_seman_batch = np.concatenate(y_seman_batch, axis=0)
		y_height_batch = np.concatenate(y_height_batch, axis=0)
		if C.offset:
			y_offset_batch = np.concatenate(y_offset_batch, axis=0)
		current_ped += batchsize
		if C.offset:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)], meta
		else:
			yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)], meta

class get_data_wider_class(object):
	def __init__(self, ped_data, C, batchsize = 8):
		self.ped_data = ped_data
		self.C = C
		self.batchsize = batchsize
		self.indexes = range(len(self.ped_data))
		random.shuffle(self.indexes)
		self.current_ped = 0
		self.correct_proportion = AverageMeter()

	def run(self):
		#self.current_ped = 0
		while True:
			x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
			meta = {}
			ind_map_batch = []
			if self.current_ped>len(self.ped_data)-self.batchsize:
				random.shuffle(self.indexes)
				self.current_ped = 0
			for i in range(self.batchsize):
				img_data = self.ped_data[self.indexes[self.current_ped]]
				try:
					img_data, x_img = data_augment.augment_wider_with_weights(img_data, self.C)
					if self.C.offset:
						y_seman, y_height, y_offset, ind_map = calc_gt_center_weights(self.C, img_data, down=self.C.down, scale=self.C.scale, offset=True)
					else:
						y_seman, y_height, ind_map = calc_gt_center(self.C, img_data,down=self.C.down, scale=self.C.scale, offset=False)

					x_img = x_img.astype(np.float32)
					x_img[:, :, 0] -= self.C.img_channel_mean[0]
					x_img[:, :, 1] -= self.C.img_channel_mean[1]
					x_img[:, :, 2] -= self.C.img_channel_mean[2]

					x_img_batch.append(np.expand_dims(x_img, axis=0))
					y_seman_batch.append(np.expand_dims(y_seman, axis=0))
					y_height_batch.append(np.expand_dims(y_height, axis=0))

					ind_map_batch.append(np.expand_dims(ind_map, axis=0))
					meta[i] = [self.indexes[self.current_ped], img_data['ratio'], img_data['hw']]
					if self.C.offset:
						y_offset_batch.append(np.expand_dims(y_offset, axis=0))
				except Exception as e:
					print ('get_batch_gt:',e)
				self.current_ped += 1
			x_img_batch = np.concatenate(x_img_batch,axis=0)
			y_seman_batch = np.concatenate(y_seman_batch, axis=0)
			y_height_batch = np.concatenate(y_height_batch, axis=0)
			ind_map_batch = np.concatenate(ind_map_batch, axis=0)
			if self.C.offset:
				y_offset_batch = np.concatenate(y_offset_batch, axis=0)
			if self.C.offset:
				yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)], np.copy(ind_map_batch), meta
			else:
				yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)], np.copy(ind_map_batch), meta

	def revise_ped_data(self, modify_img_data): #modify_img_data=[file_ind, box_ind, new_h, hw, hm]
		for modify_img_datum in modify_img_data:
			gt_data = self.ped_data[modify_img_datum[0]]
			x1, y1, x2, y2 = gt_data['bboxes'][modify_img_datum[1]]
			conf = gt_data['confs'][modify_img_datum[1]]
			#c_x, c_y = x1 + (x2-x1) / 2., y1 + (y2-y1) / 2.
			if conf > modify_img_datum[4]:
				continue
			new_h = modify_img_datum[2]
			new_h = max(new_h, 6)
			step_x = int(((x2 - x1 + 1) - new_h) / 2.)
			step_y = int(((y2 - y1 +1) - new_h)/2.)

			img_height, img_width = modify_img_datum[3]

			x1, y1 = max(x1+step_x, 0), max(y1+step_y, 0 )
			x2, y2 = min(x2-step_x, img_width - 1), min(y2-step_y, img_height - 1)
			self.ped_data[modify_img_datum[0]]['bboxes'][modify_img_datum[1]] = np.array([x1, y1, x2, y2])
			self.ped_data[modify_img_datum[0]]['confs'][modify_img_datum[1]] = modify_img_datum[4]

	def save_ped_data(self, res_path):
		with open(res_path, 'wb') as fid:
			cPickle.dump(self.ped_data, fid, cPickle.HIGHEST_PROTOCOL)


def calc_gt_center_weights(C, img_data,r=2, down=4,scale='h',offset=True):
	def gaussian(kernel):
		sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
		s = 2*(sigma**2)
		dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
		return np.reshape(dx,(-1,1))
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	w_igs = np.copy(img_data['w_bboxes'])
	ind_gts = np.copy(img_data['keep_inds'])
	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 4))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	ind_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down)),dtype='int')
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	seman_map[:,:,1] = 1
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
			dx = gaussian(x2-x1)
			dy = gaussian(y2-y1)
			gau_map = np.multiply(dy, np.transpose(dx))
			seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)
			seman_map[y1:y2, x1:x2,1] = 1
			seman_map[c_y, c_x, 2] = 1

			ind_map[c_y, c_x] = ind_gts[ind]

			if scale == 'h':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
				scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = w_igs[ind]
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
				scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = w_igs[ind]
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
				scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 3] = w_igs[ind]
			if offset:
				offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, 2] = 1

	if offset:
		return seman_map,scale_map,offset_map, ind_map
	else:
		return seman_map, scale_map, ind_map

def gaussian_dense(kernel):
	sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
	s = 2*(sigma**2)
	dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
	return np.reshape(dx,(-1,1))

kernel_size = 3
kernel_size_half = kernel_size // 2
gaussian_9 = gaussian_dense(kernel_size)

def calc_gt_center_weights_dense(C, img_data,r=2, down=4,scale='h',offset=True):
	gts = np.copy(img_data['bboxes'])
	igs = np.copy(img_data['ignoreareas'])
	w_igs = np.copy(img_data['w_bboxes'])
	ind_gts = np.copy(img_data['keep_inds'])
	scale_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	if scale=='hw':
		scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 4))
	if offset:
		offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
	ind_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down)),dtype='int')
	seman_map = np.zeros((int(C.size_train[0]/down), int(C.size_train[1]/down), 3))
	seman_map[:,:,1] = 1
	# pre_ct = []
	if len(igs) > 0:
		igs = igs/down
		for ind in range(len(igs)):
			x1,y1,x2,y2 = int(igs[ind,0]), int(igs[ind,1]), int(np.ceil(igs[ind,2])), int(np.ceil(igs[ind,3]))
			seman_map[y1:y2, x1:x2,1] = 0
	if len(gts)>0:
		gts = gts/down
		for ind in range(len(gts)):
			# x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
			x1, y1, x2, y2 = int(gts[ind, 0]), int(gts[ind, 1]), int(gts[ind, 2]), int(gts[ind, 3])
			c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)

			# if len(pre_ct) > 0:
			# 	distances = np.sqrt(np.sum((np.array(pre_ct) - np.array([c_x, c_y]))**2, axis=1))
			# 	if np.any(distances < 2):
			# 		continue
			# pre_ct.append([c_x, c_y])

			if x1 == x2:
				if x1 == C.size_train[1]/down - 1:
					x1 = x2 - 1
				else:
					x2 = x1 + 1
			if y1 == y2:
				if y1 == C.size_train[0]/down - 1:
					y1 = y2 - 1
				else:
					y2 = y1 + 1

			if x2-x1>kernel_size:
				dx = gaussian_dense(x2 - x1)
				flag_dx = 0
			elif c_x >= kernel_size_half and c_x <= C.size_train[1]/down-(kernel_size_half+1):
				dx = gaussian_9
				flag_dx = 1
			else:
				dx = gaussian_dense(x2 - x1)
				flag_dx = 0

			if y2-y1>kernel_size:
				dy = gaussian_dense(y2 - y1)
				flag_dy = 0
			elif c_y >= kernel_size_half and c_y <= C.size_train[0]/down-(kernel_size_half+1):
				dy = gaussian_9
				flag_dy = 1
			else:
				dy = gaussian_dense(y2 - y1)
				flag_dy = 0
			gau_map = np.multiply(dy, np.transpose(dx))

			if flag_dx == 1 and flag_dy == 1:
				try:
					seman_map[c_y-kernel_size_half:c_y+(kernel_size_half+1), c_x-kernel_size_half:c_x+(kernel_size_half+1), 0] = np.maximum(seman_map[c_y-kernel_size_half:c_y+(kernel_size_half+1), c_x-kernel_size_half:c_x+(kernel_size_half+1), 0], gau_map)
					seman_map[c_y-kernel_size_half:c_y+(kernel_size_half+1), c_x-kernel_size_half:c_x+(kernel_size_half+1), 1] = 1
				except Exception as e:
					print ('case_1:', seman_map[c_y-kernel_size_half:c_y+(kernel_size_half+1), c_x-kernel_size_half:c_x+(kernel_size_half+1), 0].shape, gau_map.shape)
			elif flag_dx == 1:
				try:
					seman_map[y1:y2, c_x-kernel_size_half:c_x+(kernel_size_half+1), 0] = np.maximum(seman_map[y1:y2, c_x-kernel_size_half:c_x+(kernel_size_half+1), 0], gau_map)
					seman_map[y1:y2, c_x-kernel_size_half:c_x+(kernel_size_half+1), 1] = 1
				except Exception as e:
					print ('case_2:', seman_map[y1:y2, c_x-kernel_size_half:c_x+(kernel_size_half+1), 0].shape, gau_map.shape)
			elif flag_dy == 1:
				try:
					seman_map[c_y-kernel_size_half:c_y+(kernel_size_half+1), x1:x2, 0] = np.maximum(seman_map[c_y-kernel_size_half:c_y+(kernel_size_half+1), x1:x2, 0], gau_map)
					seman_map[c_y-kernel_size_half:c_y+(kernel_size_half+1), x1:x2, 1] = 1
				except Exception as e:
					print ('case_3:', seman_map[c_y-kernel_size_half:c_y+(kernel_size_half+1), x1:x2, 0].shape, gau_map.shape)
			else:
				try:
					seman_map[y1:y2, x1:x2,0] = np.maximum(seman_map[y1:y2, x1:x2,0], gau_map)
					seman_map[y1:y2, x1:x2,1] = 1
				except Exception as e:
					print ('case_4:', seman_map[y1:y2, x1:x2,0].shape, gau_map.shape)

			seman_map[c_y, c_x, 2] = 1

			ind_map[c_y, c_x] = ind_gts[ind]

			if scale == 'h':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
				scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = w_igs[ind]
			elif scale=='w':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = 1
				scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = w_igs[ind]
			elif scale=='hw':
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
				scale_map[c_y-r:c_y+r+1, c_x-r:c_x+r+1, 2] = 1
				scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 3] = w_igs[ind]
			if offset:
				offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
				offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
				offset_map[c_y, c_x, 2] = 1

	if offset:
		return seman_map,scale_map,offset_map, ind_map
	else:
		return seman_map, scale_map, ind_map


class get_data_ShanghaiTechA_class(object):
	def __init__(self, ped_data, C, batchsize=8):
		self.ped_data = ped_data
		self.C = C
		self.batchsize = batchsize
		self.indexes = range(len(self.ped_data))
		random.shuffle(self.indexes)
		self.current_ped = 0
		self.correct_proportion = AverageMeter()

	def run(self):
		# self.current_ped = 0
		while True:
			x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
			meta = {}
			ind_map_batch = []
			if self.current_ped > len(self.ped_data) - self.batchsize:
				random.shuffle(self.indexes)
				self.current_ped = 0
			for i in range(self.batchsize):
				img_data = self.ped_data[self.indexes[self.current_ped]]
				img_data, x_img = data_augment.augment_ShanghaiTech_with_weights(img_data, self.C)
				# self.C.size_train = img_data['hw_n'] #TODO delete for multi batch
				if self.C.offset:
					y_seman, y_height, y_offset, ind_map = calc_gt_center_weights(self.C, img_data,
																				  down=self.C.down,
																				  scale=self.C.scale, offset=True) ##r=0
				else:
					y_seman, y_height, ind_map = calc_gt_center_weights(self.C, img_data, down=self.C.down,
																		scale=self.C.scale, offset=False)

				x_img = x_img.astype(np.float32)
				# import cv2
				# cv2.imwrite('1.png', x_img)
				# cv2.imwrite('2.png', y_seman[...,0]*255)
				# cv2.imwrite('3.png', y_height[..., 0] * 50)

				x_img[:, :, 0] -= self.C.img_channel_mean[0]
				x_img[:, :, 1] -= self.C.img_channel_mean[1]
				x_img[:, :, 2] -= self.C.img_channel_mean[2]

				x_img_batch.append(np.expand_dims(x_img, axis=0))
				y_seman_batch.append(np.expand_dims(y_seman, axis=0))
				y_height_batch.append(np.expand_dims(y_height, axis=0))

				ind_map_batch.append(np.expand_dims(ind_map, axis=0))
				meta[i] = [self.indexes[self.current_ped], img_data['ratio'], img_data['hw']]
				if self.C.offset:
					y_offset_batch.append(np.expand_dims(y_offset, axis=0))
				self.current_ped += 1

			x_img_batch = np.concatenate(x_img_batch, axis=0)
			y_seman_batch = np.concatenate(y_seman_batch, axis=0)
			y_height_batch = np.concatenate(y_height_batch, axis=0)
			ind_map_batch = np.concatenate(ind_map_batch, axis=0)
			if self.C.offset:
				y_offset_batch = np.concatenate(y_offset_batch, axis=0)
			if self.C.offset:
				yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch),
											 np.copy(y_offset_batch)], np.copy(ind_map_batch), meta
			else:
				yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)], np.copy(
					ind_map_batch), meta

	def revise_ped_data(self, modify_img_data):  # modify_img_data=[file_ind, box_ind, new_h, hw, hm]
		for modify_img_datum in modify_img_data:
			gt_data = self.ped_data[modify_img_datum[0]]
			x1, y1, x2, y2 = gt_data['bboxes'][modify_img_datum[1]]
			conf = gt_data['confs'][modify_img_datum[1]]
			# c_x, c_y = x1 + (x2-x1) / 2., y1 + (y2-y1) / 2.
			if conf > modify_img_datum[4]:
				continue
			new_h = modify_img_datum[2]
			new_h = max(new_h, 2)
			step_x = ((x2 - x1) - new_h) / 2. #TODO
			step_y = ((y2 - y1) - new_h) / 2.

			img_height, img_width = modify_img_datum[3]

			x1, y1 = max(x1 + step_x, 0), max(y1 + step_y, 0)
			x2, y2 = min(x2 - step_x, img_width - 1), min(y2 - step_y, img_height - 1)
			self.ped_data[modify_img_datum[0]]['bboxes'][modify_img_datum[1]] = np.array([x1, y1, x2, y2])
			self.ped_data[modify_img_datum[0]]['confs'][modify_img_datum[1]] = modify_img_datum[4]


	def save_ped_data(self, res_path):
		with open(res_path, 'wb') as fid:
			cPickle.dump(self.ped_data, fid, cPickle.HIGHEST_PROTOCOL)

class ThreadingBatches(object):
	""" Threading example class
	The run() method will be started and it will run in the background
	until the application exits.
	"""

	def __init__(self, generator=None):
		""" Constructor
		:type interval: int
		:param interval: Check interval, in seconds
		"""
		self.generator = generator
		self.stack = []
		self.queue_size = 16  # This number should be experimented with based on how much RAM you have
		thread = threading.Thread(target=self.run, args=())
		thread.daemon = True  # Daemonize thread
		thread.start()  # Start the execution

	def run(self):
		""" Method that runs forever """
		while True:
			# Do something
			if (len(self.stack) < self.queue_size):
				try:
					batch = next(self.generator.run())
					self.stack.append(batch)
				except ValueError:
					None

	def popNextBatch(self):
		while len(self.stack) == 0:
			time.sleep(0.1)

		return self.stack.pop(0)

	def revise_ped_data(self, modify_img_data):
		return self.generator.revise_ped_data(modify_img_data)

	def save_ped_data(self, res_path):
		self.generator.save_ped_data(res_path)
