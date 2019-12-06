import os
import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout

from pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG


class CLS_SSG_Model(Model):

	def __init__(self, batch_size, num_points, num_classes, activation=tf.nn.relu):
		super(CLS_SSG_Model, self).__init__()

		self.activation = tf.nn.leaky_relu
		self.batch_size = batch_size
		self.num_points = num_points
		self.num_classes = num_classes
		self.keep_prob = 0.5

		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.init_network()


	def init_network(self):

		self.layer1 = Pointnet_SA(
			npoint=512,
			radius=0.2,
			nsample=32,
			mlp=[64, 64, 128],
			group_all=False,
			activation=self.activation
		)

		self.layer2 = Pointnet_SA(
			npoint=128,
			radius=0.4,
			nsample=64,
			mlp=[128, 128, 256],
			group_all=False,
			activation=self.activation
		)

		self.layer3 = Pointnet_SA(
			npoint=None,
			radius=None,
			nsample=None,
			mlp=[256, 512, 1024],
			group_all=True,
			activation=self.activation
		)

		self.dense1 = Dense(512, activation=self.activation)

		self.dense2 = Dense(128, activation=self.activation)

		self.dense3 = Dense(self.num_classes, activation=tf.nn.softmax)

		self.dropout = Dropout(self.keep_prob)


	def call(self, input):

		xyz, points = self.layer1(input, None)
		xyz, points = self.layer2(xyz, points)
		xyz, points = self.layer3(xyz, points)

		net = tf.reshape(points, (self.batch_size, -1))

		net = self.dense1(net)
		net = self.dropout(net)
		net = self.dense2(net)
		net = self.dropout(net)
		net = self.dense3(net)

		return net
