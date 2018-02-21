from .layers import conv
from .forwards import Forwards
import numpy as np


import pdb
class Backwards:
	""" CNN中后向传播，W 参数梯度传递，用于更新权值 W, 从而调整网络 """


	def __init__(self, forward):
		self.delta = forward.delta


	def run(self, *args):
		''' 先计算各层的 Delta 值，然后计算各层的梯度值 '''


		# pdb.set_trace()
		inputData = args[0].output

		for i in -(np.arange(len(args)-1))-1:	# args 中包含input层，而实际input无需计算
			# pdb.set_trace()
			# print(str(args[i].layerCount)+'\t'+args[i].layerType)
			if args[i].layerType == 'softmaxLayer':
				softmaxLayerDelta(args[i], self.delta)
			else:
				if args[i+1].layerType == 'softmaxLayer':
					layerBeforeSoftmaxLayerDelta(args[i], args[i+1])
				elif args[i+1].layerType == 'convLayer':
					# pdb.set_trace()
					layerBeforeConvLayerDelta(args[i], args[i+1])
				elif args[i+1].layerType == 'poolLayer':
					layerBeforePoolLayerDelta(args[i], args[i+1])
					# print(args[i].delta)
				
			if args[i].layerType in Forwards.hasGrad_layerTypeSet:
				computeGrad(args[i-1], args[i])


def softmaxLayerDelta(softmaxLayer, cost):
	'''
		作用：
			计算softmax层的 Delta 值
		参数：
			cost：cost = sum(p_label*log(p_predict))
			softmaxLayer: 即将计算的softmax层
	'''

	softmaxLayer.delta = cost


def layerBeforeSoftmaxLayerDelta(layerBeforesoftmaxLayer, softmaxLayer):
	'''
		作用：
			计算softmax前的那一层的 Delta 值
		注： 因为softmax是全连接，因此也可以看成一个卷积层
	'''
	# pdb.set_trace()
	layerBeforeConvLayerDelta(layerBeforesoftmaxLayer, softmaxLayer)


def layerBeforePoolLayerDelta(layerBeforepoolLayer, poolLayer):
	'''
		作用：
			计算pooling层前的那一层的 Delta 值
	'''

	# pdb.set_trace()
	layerBeforepoolLayer.delta = np.kron(poolLayer.delta, poolLayer.W)


def layerBeforeConvLayerDelta(layerBeforeconvLayer, convLayer):
	'''
		作用：
			计算卷积层前的那一层的 Delta 值
		参数：
			convLayer: 
				CNN前馈过程中，位于该层卷积层之后的层，即在误差后向传播过程中先于该卷积层计算
			layerBeforeconvLayer:
				即将计算的卷积层
	'''

	delta_convLayer = np.zeros(layerBeforeconvLayer.outputSize)    # 以下计算 conv 层的 Delta 值
	# pdb.set_trace()
	delta_layerAfterConvlayer_padding_Size = ( convLayer.outputSize[0], 
		convLayer.outputSize[-2]+2*convLayer.WSZ[-2]-2, 
		convLayer.outputSize[-1]+2*convLayer.WSZ[-2]-2 )
	delta_layerAfterConvlayer_padding = np.zeros( delta_layerAfterConvlayer_padding_Size )
	offset = ( convLayer.WSZ[-2]-1, convLayer.WSZ[-1]-1 )
	# pdb.set_trace()
	delta_layerAfterConvlayer_padding[:, offset[0]:delta_layerAfterConvlayer_padding_Size[-2]-offset[0], offset[1]:delta_layerAfterConvlayer_padding_Size[-1]-offset[1]] =  convLayer.delta
	for k in np.arange(layerBeforeconvLayer.outputSize[0]):
		for i in np.arange(layerBeforeconvLayer.outputSize[-2]):
			for j in np.arange(layerBeforeconvLayer.outputSize[-1]):
				for c in np.arange(convLayer.outputSize[0]):    # 1 代表l+1层每个核与l层连接的通道
					# pdb.set_trace()
					# print(k, c, i, j)
					
					delta_convLayer[k, i, j] = delta_convLayer[k, i, j] + np.sum(
						delta_layerAfterConvlayer_padding[c, i:i+convLayer.WSZ[-2], j:j+convLayer.WSZ[-1] ] *
						np.rot90(convLayer.W[c, k], 2) )

	layerBeforeconvLayer.delta = delta_convLayer


def layerBeforePoolLayerDelta(layerBeforepoolLayer, poolLayer):
	'''
		作用：
			计算pooling层的 Delta 值
	'''
	paddingDelta = np.zeros(poolLayer.paddingDeltaSize)
	for k in range(poolLayer.paddingDeltaSize[0]):
		paddingDelta[k] = 1/(poolLayer.WSZ[-2]*
			poolLayer.WSZ[-1])*np.kron(poolLayer.delta[k], 
			np.ones(poolLayer.WSZ))

	layerBeforepoolLayer.delta = paddingDelta[:,0:poolLayer.paddingDeltaSize[-2], 0:poolLayer.paddingDeltaSize[-1]]


def computeGrad(beforeThisLayer, thisLayer):
	for k in np.arange(thisLayer.WSZ[0]):
		# pdb.set_trace()
		for c in np.arange(thisLayer.WSZ[1]):
			for i in np.arange(thisLayer.WSZ[-1]):
				for j in np.arange(thisLayer.WSZ[-2]):
					# print((k,c,i,j),thisLayer.layerType)
					if thisLayer.layerType == 'convLayer' and thisLayer.channelControl[c, k] == 0:
						continue
					thisLayer.grad[k, c, i, j] = np.sum(
						thisLayer.delta[k]*beforeThisLayer.output[c, i:i+thisLayer.outputSize[-1], j:j+thisLayer.outputSize[-2]])