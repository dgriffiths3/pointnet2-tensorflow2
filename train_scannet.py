import os
import sys
import datetime

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from models.sem_seg_model import SEM_SEG_Model
from data.dataset import TFDataset

tf.random.set_seed(42)


def train_step(optimizer, model, loss_object, acc_object, train_pts, train_labels):

	with tf.GradientTape() as tape:

		pred = model(train_pts)
		loss = loss_object(train_labels, pred)
		acc = acc_object(train_labels, pred)

	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return loss, acc


def test_step(optimizer, model, loss_object, acc_object, test_pts, test_labels):

	with tf.GradientTape() as tape:

		pred = model(test_pts)
		loss = loss_object(test_labels, pred)
		acc = acc_object(test_labels, pred)

	return loss, acc


def train(config, params):

	model = SEM_SEG_Model(params['batch_size'], params['num_points'], params['num_classes'], params['bn'])

	optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
	acc_object = tf.keras.metrics.SparseCategoricalAccuracy()

	train_ds = TFDataset(os.path.join(config['dataset_dir'], 'train.tfrecord'), params['batch_size'], 'scannet')
	val_ds = TFDataset(os.path.join(config['dataset_dir'], 'val.tfrecord'), params['batch_size'], 'scannet')

	train_summary_writer = tf.summary.create_file_writer(
		os.path.join(config['log_dir'], config['log_code'])
	)

	with train_summary_writer.as_default():

		while True:

			train_pts, train_labels = train_ds.get_batch()

			loss, acc = train_step(
				optimizer,
				model,
				loss_object,
				acc_object,
				train_pts,
				train_labels
			)

			if optimizer.iterations % config['log_freq'] == 0:
				tf.summary.scalar('train loss', loss, step=optimizer.iterations)
				tf.summary.scalar('train accuracy', acc, step=optimizer.iterations)

			if optimizer.iterations % config['test_freq'] == 0:

				test_pts, test_labels = val_ds.get_batch()

				test_loss, test_acc = test_step(
					optimizer,
					model,
					loss_object,
					acc_object,
					test_pts,
					test_labels
				)

				tf.summary.scalar('test loss', test_loss, step=optimizer.iterations)
				tf.summary.scalar('test accuracy', test_acc, step=optimizer.iterations)


if __name__ == '__main__':

	config = {
		'dataset_dir' : 'data/scannet',
		'log_dir' : 'logs',
		'log_code' : 'scannet_1',
		'log_freq' : 10,
		'test_freq' : 100
	}

	params = {
		'batch_size' : 4,
		'num_points' : 8192,
		'num_classes' : 21,
		'lr' : 0.001,
		'bn' : True
	}

	train(config, params)
