import numpy as np
from .basiclayer import BasicLayer
from .convlayer import conv


import pdb
class PoolLayer(BasicLayer):
	'''
		作用： 池化层，降维
		注： 这里的池化层目前仅能（默认）使用均匀池化
			并且由于该pooling层没有做边界检验，因此pooling的stepSize必须使采样块不能重叠，且必须整除输入数据
			例如，大小3*3的采样块不能用2*2的stepSize，且输入数据可以为9*9，但不能为10*10
	'''

	def __init__(self, W_size, stepSize):
		'''
			参数：
				W_size：均匀范围（由于均匀池化在某种程度上可以看成卷积层）
				stepSize: 降维步长（只降维二维图像，对通道不降维）
		'''
		super().__init__('poolLayer')

		self.WSZ = W_size
		self.W = np.ones(self.WSZ)
		self.stepSize = stepSize


	def run(self, src):
		'''
			作用: 执行pooling层的操作
		'''
		self.input = src
		self.inputSize = self.input.shape
		self.outputSize = (self.inputSize[0],
			int(np.ceil((self.inputSize[-2]-np.ceil(self.WSZ[-2]/2))/self.stepSize[-2])),
			int(np.ceil((self.inputSize[-1]-np.ceil(self.WSZ[-1]/2))/self.stepSize[-1])) )

		# pdb.set_trace()
		self.output = np.zeros(self.outputSize)
		for k in range(self.outputSize[0]):
			for i in range(self.outputSize[-2]):
				for j in range(self.outputSize[-1]):
					self.output[k,i,j] = np.mean(
						self.input[k, i*self.stepSize[-2]:min(i*self.stepSize[-2]+self.WSZ[-2], self.inputSize[-2]), 
						j*self.stepSize[-1]:min(j*self.stepSize[-1]+self.WSZ[-1], self.inputSize[-1]) ])

		self.delta = np.zeros(self.outputSize)
		self.paddingDeltaSize = self.inputSize

