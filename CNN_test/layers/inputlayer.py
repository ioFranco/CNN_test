import cv2
import numpy as np
from .basiclayer import BasicLayer

class InputLayer(BasicLayer):
	'''
		作用：用于输入数据
	'''
	def __init__(self, imgs, labels):
		super().__init__('inputLayer')

		# I = (np.arange(9*2)+1).reshape(2,3,3)
		self.output=imgs[0:5, :, 0::2, 0::2]
		# self.output = I
		self.labels = labels[0:5, :, :, :]
		# self.labels = np.array([ [[1]], [[0]], [[0]] ])