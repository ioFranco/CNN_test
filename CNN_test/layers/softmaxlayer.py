import cv2
import numpy as np
from .basiclayer import BasicLayer


import pdb
class SoftmaxLayer(BasicLayer):
	''' 定义softmax层 '''


	def __init__(self, W_size):
		''' 
			作用：
				初始化softmax层的权值 
			格式： 
					W_size = (2,2,6,6), 是一个tuple类型，第一个数字2代表卷积核的数量，
				第二个数字2代表每个卷积核与输入数据的相卷时的通道数，即输入数据的
				通道数，后面的两个6表示每个卷积核每个通道的大小6*6
		'''

		# ''' (2,2,(1,1))输出特征数量,前一层特征数,前一层特征大小'''
		# self.WSZ = (2, 2, 1, 1)
		super().__init__('softmaxLayer')
		self.WSZ = W_size
		# self.W = 2/(W_size[0]*W_size[1]*W_size[2]*W_size[3])*np.random.rand(*self.WSZ)
		self.numClass = self.WSZ[0]
		self.W = 0.1/(np.arange(W_size[0]*W_size[1]*W_size[2]*W_size[3])+1).reshape(*self.WSZ)



	def run(self, src):
		''' 
			作用：
				执行softmax层，exp(W*X)/sum(exp(W*X))
			参数：
				src: 
					输入数据
		'''

		self.input = src
		self.inputSize = self.input.shape
		self.outputSize = (self.inputSize[0], self.numClass, 1,1)
		self.output = np.zeros(self.outputSize)
		self.channelControl = np.ones((self.inputSize[-3], self.outputSize[-3]))

		# pdb.set_trace()
		for num in np.arange(self.outputSize[0]):
			for i in np.arange(self.outputSize[-3]):
				h = self.W[i]*self.input[num]
				# import pdb
				# pdb.set_trace()
				self.output[num, i, 0] = np.sum(h)

			# pdb.set_trace()
			self.output[num] = np.exp(self.output[num])/np.sum(np.exp(self.output[num]))

		self.delta = np.zeros(self.outputSize)		# 用于反向传播保留误差值
		self.grad = np.zeros(self.WSZ)		# 用于反向传播保留梯度值
		self.perGrad = np.zeros( (self.outputSize[0], *self.WSZ) )
		self.numGradient = np.zeros(self.WSZ)		# 用于计算数值梯度值
		self.perNumGradient = np.zeros( (self.outputSize[0], *self.WSZ) )