import tensorflow as tf

class TFDataset():

	def __init__(self, path, batch_size, dataset='modelnet', shuffle_buffer=10000):

		self.path = path
		self.batch_size = batch_size
		self.iterator = None
		self.dataset_type = dataset
		self.shuffle_buffer = shuffle_buffer

		self.dataset = self.read_tfrecord(self.path, self.batch_size)

		self.get_iterator()


	def read_tfrecord(self, path, batch_size):

		dataset = tf.data.TFRecordDataset(path).batch(batch_size)
		if self.dataset_type == 'modelnet':
			dataset = dataset.map(self.extract_modelnet_fn).shuffle(self.shuffle_buffer)
		elif self.dataset_type == 'scannet':
			dataset = dataset.map(self.extract_scannet_fn).shuffle(self.shuffle_buffer)

		return dataset


	def extract_modelnet_fn(self, data_record):

		features = {
			'points' : tf.io.VarLenFeature(tf.float32),
			'label' : tf.io.FixedLenFeature([],tf.int64)
		}

		sample = tf.io.parse_example(data_record, features)

		return sample['points'].values, sample['label']


	def extract_scannet_fn(self, data_record):

		features = {
			'points' : tf.io.VarLenFeature(tf.float32),
			'labels' : tf.io.VarLenFeature(tf.int64)
		}

		sample = tf.io.parse_example(data_record, features)

		return sample['points'].values, sample['labels'].values


	def get_iterator(self):

		self.iterator = self.dataset.__iter__()


	def reset_iterator(self):

		self.dataset.shuffle(self.shuffle_buffer)
		self.get_iterator()


	def get_batch(self):

		while True:
			try:
				batch = self.iterator.next()
				if self.dataset_type == 'modelnet':
					pts = tf.reshape(batch[0], (self.batch_size, -1, 3))
					label = tf.reshape(batch[1], (self.batch_size, 1))
				elif self.dataset_type == 'scannet':
					pts = tf.reshape(batch[0], (self.batch_size, -1, 3))
					label = tf.reshape(batch[1], (self.batch_size, -1))
				break
			except:
				self.reset_iterator()

		return pts, label
