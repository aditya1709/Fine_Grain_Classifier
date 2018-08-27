import glob
import scipy.io
import json
import numpy as np
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
		images_path = glob.glob(config.DATA_PATH + 'cars_' + config.MODE + '/*.jpg')

		# Load corresponding labels
		labels = scipy.io.loadmat(config.DATA_PATH + 'devkit/cars_' + config.MODE + '_annos.mat')

		return images_path, labels

# class StanfordCarsTFRecords():
#     """
#     	Load TFRecords from the disk
#     """

def main():
	cfg = model_config()

	generator = GenerateTFRecords(cfg)
	im_add, labels = generator.gen_TFR()
	label = []
	for i in labels['annotations'][0][:]:
		label.append(i[4]) 
	print(np.unique(label))
	print(cfg.class_names)

if __name__ == '__main__':
    main()