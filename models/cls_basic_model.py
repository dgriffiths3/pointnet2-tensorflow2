# Pointnet 1 model: https://github.com/charlesq34/pointnet

import os
import sys

import pylab as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
	Dense,
	Flatten,
	Conv1D,
	MaxPool1D,
	BatchNormalization,
	Dropout
)


class Pointnet_Model(Model):

	def __init__(self, batch_size, num_points, num_classes, bn=False, activation=tf.nn.leaky_relu):
		super(Pointnet_Model, self).__init__()

		self.activation = activation
		self.batch_size = batch_size
		self.num_points = num_points
		self.num_classes = num_classes
		self.keep_prob = 0.5
		self.bn = bn

		self.kernel_initializer = 'glorot_normal'
		self.kernel_regularizer = None

		self.init_network()


	def init_network(self):

		self.mlp1 = Conv1D(
			filters=64,
			kernel_size=[1],
			strides=[1],
			padding='same',
			activation=self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		if self.bn: self.bn1 = BatchNormalization()

		self.mlp2 = Conv1D(
			filters=64,
			kernel_size=[1],
			strides=[1],
			padding='same',
			activation=self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		if self.bn: self.bn2 = BatchNormalization()

		self.mlp3 = Conv1D(
			filters=64,
			kernel_size=[1],
			strides=[1],
			padding='same',
			activation=self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		if self.bn: self.bn3 = BatchNormalization()

		self.mlp4 = Conv1D(
			filters=128,
			kernel_size=[1],
			strides=[1],
			padding='same',
			activation=self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		if self.bn: self.bn4 = BatchNormalization()

		self.mlp5 = Conv1D(
			filters=1024,
			kernel_size=[1],
			strides=[1],
			padding='same',
			activation=self.activation,
			kernel_initializer=self.kernel_initializer,
			kernel_regularizer=self.kernel_regularizer
		)

		if self.bn: self.bn5 = BatchNormalization()

		self.maxpool = MaxPool1D(
			pool_size = (self.num_points)
		)

		self.dense1 = Dense(512, activation=self.activation)

		if self.bn: self.bn6 = BatchNormalization()

		self.dense2 = Dense(256, activation=self.activation)

		if self.bn: self.bn7 = BatchNormalization()

		self.dropout1 = Dropout(self.keep_prob)

		self.dense3 = Dense(self.num_classes, activation=tf.nn.softmax)


	def call(self, input, training=True):

		net = self.mlp1(input)
		if training == True and self.bn == True: net = self.bn1(net, training=training)

		net = self.mlp2(net)
		if training == True and self.bn == True: net = self.bn2(net, training=training)

		net = self.mlp3(net)
		if training == True and self.bn == True: net = self.bn3(net, training=training)

		net = self.mlp4(net)
		if training == True and self.bn == True: net = self.bn4(net, training=training)

		net = self.mlp5(net)
		if training == True and self.bn == True: net = self.bn5(net, training=training)

		net = self.maxpool(net)
		net = tf.reshape(net, (self.batch_size, -1))

		net = self.dense1(net)
		if training == True and self.bn == True: net = self.bn6(net, training=training)

		net = self.dense2(net)
		if training == True and self.bn == True: net = self.bn7(net, training=training)

		net = self.dropout1(net)

		pred = self.dense3(net)

		return pred
