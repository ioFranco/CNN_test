import numpy as np
from .basiclayer import BasicLayer


import pdb
class ConvLayer(BasicLayer):
	""" 
		作用：
			定义卷积层 
		类的成员:
				WSZ			卷积核大小
				W     		卷积核
				input 		待卷积数据
				inputSize 	待卷积数据大小
				output 		卷积的输出
				outputSize	卷积的输出大小				  
	"""

    
	def __init__(self, W_size, channelControl):
		''' 
			作用：
				初始化卷积核 
			格式：
					W_size = (2,1,5,5), 是一个tuple类型，第一个数字2代表卷积核的数量，
				第二个数字1代表每个卷积核与输入数据的相卷时的通道数，即输入数据的
				通道数，后面的两个5表示每个卷积核每个通道的大小5*5
					channelCOntrol 用于控制卷积核对被卷积层的卷积数据的通道，
				例如，被卷积层是3*5*5，卷积输出2*2*2。如果不进行控制，则每个卷积核与被卷积
				层所有通道相卷，即3个通道，相当于channelControl=ones(3,2)，如果令
				channelControl=[[1, 0],[1, 1], [0, 1]], 则卷积输出层第一层与被卷积层1、2
				通道相连，第二层与其2、3通道相连
		'''

		# self.WSZ = (2,1,3,3)    # 2个卷积核，每个大小为5*5
		super().__init__('convLayer')
		self.WSZ = W_size
		# self.W = 2/(W_size[0]*W_size[1]*W_size[2]*W_size[3])*np.random.rand(*self.WSZ)
		self.W = 0.1/((np.arange((W_size[0]*W_size[1]*W_size[2]*W_size[3]))+1).reshape(*self.WSZ))
		self.channelControl = channelControl
		for k in range(W_size[0]):
			for c in range(W_size[1]):
				if self.channelControl[c,k]==0:
					self.W[k,c,:,:] = 0


	def run(self, src):
		''' 
			作用：
				执行卷积层的卷积操作，没用padding功能，即不会使输入输出尺寸一样 
			参数：
				src: 
					输入的数据，即待卷积数据
		'''

		self.input = src
		self.inputSize = self.input.shape
		self.outputSize = (self.inputSize[0], 		# 输入的图像数目
			self.WSZ[0], 					# 卷积后的输出数量和卷积核数目相同
			self.inputSize[-2]-self.WSZ[-2]+1,		# 卷积后的输出行数 = 输入行数-卷积核行数+1
			self.inputSize[-1]-self.WSZ[-1]+1)		# 卷积后的输出列数 = 输入列数-卷积核列数+1
		self.output = np.zeros(self.outputSize)
		# pdb.set_trace()
		conv(self.output, self.input, self.W, self.WSZ, self.channelControl, self.outputSize)

		self.delta = np.zeros(self.outputSize)		# 用于反向传播保留误差值
		self.grad = np.zeros(self.WSZ)		# 用于反向传播保留梯度值
		self.perGrad = np.zeros( (self.outputSize[0], *self.WSZ) )
		self.numGradient = np.zeros(self.WSZ)		# 用于计算数值梯度值
		self.perNumGradient = np.zeros( (self.outputSize[0], *self.WSZ) )



def conv(output, input, W, WSZ, channelControl, outputSize):
	'''
		作用：实现卷积操作

	'''

	for num in np.arange(outputSize[0]):
		for k in np.arange(outputSize[-3]):
			for i in np.arange(outputSize[-2]):
				for j in np.arange(outputSize[-1]):
					for c in np.arange(WSZ[1]):    # 1 代表l+1层每个核与l层连接的通道
						if channelControl[c, k] == 0:
							continue
						output[num, k, i, j] = ( output[num, k, i, j]+
							np.sum(W[k, c] * 
								input[num, c, i:i+WSZ[-2], j:j+WSZ[-1]]) )

