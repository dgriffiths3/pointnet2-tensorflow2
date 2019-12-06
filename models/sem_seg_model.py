import os
import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout

from pnet2_layers.layers import Pointnet_SA, Pointnet_FP


class SEM_SEG_Model(Model):

	def __init__(self, batch_size, num_points, num_classes, activation=tf.nn.relu):
		super(SEM_SEG_Model, self).__init__()

		self.activation = tf.nn.leaky_relu
		self.batch_size = batch_size
		self.num_points = num_points
		self.keep_prob = 0.5
		self.num_classes = num_classes

		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.init_network()


	def init_network(self):

		self.sa_1 = Pointnet_SA(
			npoint=1024,
			radius=0.1,
			nsample=32,
			mlp=[32, 32, 64],
			group_all=False,
			activation=self.activation
		)

		self.sa_2 = Pointnet_SA(
			npoint=256,
			radius=0.2,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation
		)

		self.sa_3 = Pointnet_SA(
			npoint=64,
			radius=0.4,
			nsample=32,
			mlp=[128, 128, 256],
			group_all=False,
			activation=self.activation
		)

		self.sa_4 = Pointnet_SA(
			npoint=16,
			radius=0.8,
			nsample=32,
			mlp=[256, 256, 512],
			group_all=False,
			activation=self.activation
		)

		self.fp_1 = Pointnet_FP(
			mlp = [256, 256],
			activation = self.activation
		)

		self.fp_2 = Pointnet_FP(
			mlp = [256, 256],
			activation = self.activation
		)

		self.fp_3 = Pointnet_FP(
			mlp = [256, 128],
			activation = self.activation
		)

		self.fp_4 = Pointnet_FP(
			mlp = [128, 128, 128],
			activation = self.activation
		)


		self.dense1 = Dense(128, activation=self.activation)

		self.dense2 = Dense(self.num_classes, activation=tf.nn.softmax)

		self.dropout = Dropout(self.keep_prob)


	def call(self, input):

		l0_xyz = input
		l0_points = None

		l1_xyz, l1_points = self.sa_1(l0_xyz, l0_points)
		l2_xyz, l2_points = self.sa_2(l1_xyz, l1_points)
		l3_xyz, l3_points = self.sa_3(l2_xyz, l2_points)
		l4_xyz, l4_points = self.sa_4(l3_xyz, l3_points)

		l3_points = self.fp_1(l3_xyz, l4_xyz, l3_points, l4_points)
		l2_points = self.fp_2(l2_xyz, l3_xyz, l2_points, l3_points)
		l1_points = self.fp_3(l1_xyz, l2_xyz, l1_points, l2_points)
		l0_points = self.fp_4(l0_xyz, l1_xyz, l0_points, l1_points)

		net = self.dense1(l0_points)
		net = self.dropout(net)
		net = self.dense2(net)

		return net
