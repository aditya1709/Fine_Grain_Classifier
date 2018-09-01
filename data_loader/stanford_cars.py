import glob
import scipy.io
import json
import cv2
import os
import base64
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/adi/Documents/Fine_grain/FG_classifier/config')

from config import model_config
from random import shuffle

class GenerateTFRecords:
	"""
		Convert images and labels to TFRecords format
	"""
	def __init__(self,config):
		self.config = config

	def gen_TFR(self):
		config = self.config

		# Load the image paths 
		images_path = sorted(glob.glob(config.DATA_PATH + 'cars_' + config.MODE + '/*.jpg'))

		# Load corresponding labels (fine labels)
		labels = scipy.io.loadmat(config.DATA_PATH + 'devkit/cars_' + config.MODE + '_annos.mat')

		return images_path, labels

	# Load images and resize them appropriately
	def load_image(self,im_address):
		image = cv2.imread(im_address)
		orig_h , orig_w,_  = [float(v) for v in image.shape]
		image = cv2.resize(image,(self.config.IMAGE_WIDTH,self.config.IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype(np.float32)
		image_string = cv2.imencode('.jpg', image)[1].tostring()
		return image_string

	# Convert data to suitable format
	def _int64_feature(self,value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	def _bytes_feature(self,value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class StanfordCarsTFRecords():
	"""
		Load TFRecords from the disk
	"""
	def __init__(self, config):
		self.config = config

	def Parse_Data(self):
		feature = {
			self.config.MODE+'/label_coarse': tf.FixedLenFeature([], tf.int64),
			self.config.MODE+'/label_fine': tf.FixedLenFeature([], tf.int64),
			self.config.MODE+'/image': tf.FixedLenFeature([], tf.string)}

		# Create a list of filenames and pass it to a queue
		filename_queue = tf.train.string_input_producer([self.config.TFR_PATH], num_epochs = self.config.NUM_EPOCHS)
		# Define a reader and read the next record
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		# Decode the record read by the reader
		features = tf.parse_single_example(serialized_example, features=feature)
		# Convert the image data from string back to the numbers
		image = tf.image.decode_jpeg(features[self.config.MODE+'/image'], channels = 3)
		image = tf.reshape(image, [self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3])
		image = tf.cast(image, tf.float32)
		
		# Cast label_coarse and label_fine data into int32
		label_f = tf.cast(features[self.config.MODE+'/label_fine'], tf.int32)
		label_c = tf.cast(features[self.config.MODE+'/label_coarse'], tf.int32)

		# Creates batches from tensors
		self.images, self.labels_f, self.labels_c = tf.train.batch([image, label_f, label_c], batch_size=self.config.BATCH_SIZE, capacity=30, num_threads=1)

def main():
	cfg = model_config()

	# ########## To generate TFRecords ###########
	if cfg.TFR_generate :
		generator = GenerateTFRecords(cfg)
		im_add, labels_fine = generator.gen_TFR()
		label_coarse = []
		label_fine = []
		for i in labels_fine['annotations'][0][:]:
			# Labels start from 0
			label_fine.append(i[4][0][0]-1) 
			# Convert fine label to coarse label
			label_coarse.append(cfg.CLASS_COARSE.index(cfg.CLASS_LIST[(i[4][0][0])-1][1]))	

		#Shuffle the data 
		if cfg.SHUFFLE:
			c = list(zip(im_add, label_coarse, label_fine))
			shuffle(c)
			im_add, labels_coarse, labels_fine = zip(*c)

		# Address to save the TFRecords file 
		filename = '/home/adi/Documents/Fine_grain/dataset/' + cfg.MODE + '.tfrecords'  
		# open the TFRecords file to write
		writer = tf.python_io.TFRecordWriter(filename)
		for i in range(len(im_add)):
			# print how many images are saved every 1000 images
			if not i % 1000:
				print(cfg.MODE+' data: {}/{}'.format(i, len(im_add)))
				sys.stdout.flush()
			# Load the image
			img = generator.load_image(im_add[i])
			label_c = labels_coarse[i]
			label_f = labels_fine[i]
			# Create a feature
			feature = {cfg.MODE+'/label_coarse': generator._int64_feature(label_c),
					   cfg.MODE+'/label_fine': generator._int64_feature(label_f),
					   cfg.MODE+'/image': generator._bytes_feature(tf.compat.as_bytes(img))}

			# Create an example protocol buffer
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			
			# Serialize to string and write on the file
			writer.write(example.SerializeToString())
			
		writer.close()
		sys.stdout.flush()

		print(cfg.MODE+' TFRecords have been successfully generated')

	# ############## Read from TFRecords and prepare data #################
	else:
		tf.reset_default_graph()
		sess = tf.Session()
		# Build the graph
		data_loader = StanfordCarsTFRecords(cfg)
		data_loader.Parse_Data()
		# Initialize all global and local variables
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)
		# Create a coordinator and run all QueueRunner objects
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess= sess)
		for batch_index in range(5):
			img, lbl_f, lbl_c = sess.run([data_loader.images, data_loader.labels_f, data_loader.labels_c])
			img = img.astype(np.uint8)
			for j in range(5):
				plt.subplot(2, 3, j+1)
				plt.imshow(img[j, ...])
				plt.title(cfg.CLASS_FINE[lbl_f[j]])
			plt.show()
		# Stop the threads
		coord.request_stop()
		
		# Wait for threads to stop
		coord.join(threads)
		sess.close()


if __name__ == '__main__':
	main()