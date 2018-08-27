import os
import numpy as np
import sys
import scipy.io
from easydict import EasyDict as edict 

def model_config():
	cfg = edict()

	# Classes for classification
	cfg.class_names = scipy.io.loadmat('/home/adi/Documents/Fine_grain/dataset/devkit/cars_meta.mat')
	cfg.CLASS_NAMES_COARSE = ()
	cfg.CLASS_NAMES_FINE = ()

	# Number of classes
	cfg.CLASSES = len(cfg.CLASS_NAMES_COARSE)

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

	return cfg