import os
import numpy as np
import sys

sys.path.append('/home/adi/Documents/Fine_grain/dataset/devkit/')

import scipy.io
import json
import collections
from easydict import EasyDict as edict 
from clean_classes import class_corrected

def model_config():
	cfg = edict()

	# Class mapping from fine to coarse
	cfg.CLASS_LIST = class_corrected()

	# Class names
	cfg.CLASS_COARSE = ['Sedan','Hatchback','Convertible','Coupe','Wagon','SUV','Cab','Van','Minivan']
	cfg.CLASS_FINE = list(zip(*cfg.CLASS_LIST))[0]

	# Number of classes
	cfg.CLASSES = len(cfg.CLASS_COARSE)

	# Image dimensions
	cfg.IMAGE_WIDTH = 640
	cfg.IMAGE_HEIGHT = 480

	# Network paramters
	cfg.BATCH_SIZE = 5
	cfg.NUM_EPOCHS = 100
	cfg.LR = 0.01

	# Checkpoint paramters
	cfg.MAX_TO_KEEP = 5

	# Shuffle data
	cfg.SHUFFLE = True

	# Dataset location
	cfg.DATA_PATH = '/home/adi/Documents/Fine_grain/dataset/'

	# Mode of network operation - accepts 'train' or 'test'
	cfg.MODE = 'train'

	# Mode of TFRecord generation
	cfg.TFR_generate = False
	cfg.TFR_PATH = cfg.DATA_PATH + cfg.MODE + '.tfrecords' 

	return cfg