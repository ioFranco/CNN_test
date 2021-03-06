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


def softmaxLayerDelta(softmaxLayer, perCost):
	'''
		作用：
			计算softmax层的 Delta 值
		参数：
			perCost：perCost = sum(p_label*log(p_predict))
			softmaxLayer: 即将计算的softmax层
	'''

	softmaxLayer.delta = perCost


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

	delta_layerBeforeconvLayer = np.zeros(layerBeforeconvLayer.outputSize)    # 以下计算 conv 层的 Delta 值
	# pdb.set_trace()
	delta_convlayer_afterPadding_Size = ( 
		convLayer.outputSize[0], 
		convLayer.outputSize[-3],
		convLayer.outputSize[-2]+2*convLayer.WSZ[-2]-2, 
		convLayer.outputSize[-1]+2*convLayer.WSZ[-2]-2 )
	delta_convlayer_afterPadding = np.zeros( delta_convlayer_afterPadding_Size )
	offset = ( convLayer.WSZ[-2]-1, convLayer.WSZ[-1]-1 )
	# pdb.set_trace()
	delta_convlayer_afterPadding[:, :, offset[0]:delta_convlayer_afterPadding_Size[-2]-offset[0], offset[1]:delta_convlayer_afterPadding_Size[-1]-offset[1]] =  convLayer.delta
	for num in np.arange(layerBeforeconvLayer.outputSize[0]):
		for k in np.arange(layerBeforeconvLayer.outputSize[-3]):
			for i in np.arange(layerBeforeconvLayer.outputSize[-2]):
				for j in np.arange(layerBeforeconvLayer.outputSize[-1]):
					for c in np.arange(convLayer.outputSize[-3]):    # 1 代表l+1层每个核与l层连接的通道
						# pdb.set_trace()
						# print(k, c, i, j)
						
						delta_layerBeforeconvLayer[num, k, i, j] = delta_layerBeforeconvLayer[num, k, i, j] + np.sum(
							delta_convlayer_afterPadding[num, c, i:i+convLayer.WSZ[-2], j:j+convLayer.WSZ[-1] ] *
							np.rot90(convLayer.W[c, k], 2) )

	layerBeforeconvLayer.delta = delta_layerBeforeconvLayer


def layerBeforePoolLayerDelta(layerBeforepoolLayer, poolLayer):
	'''
		作用：
			计算pooling层的 Delta 值
	'''
	paddingDelta = np.zeros(poolLayer.paddingDeltaSize)
	for num in np.arange(poolLayer.paddingDeltaSize[0]):
		paddingDelta[num] = 1/(poolLayer.WSZ[-2]*
			poolLayer.WSZ[-1])*np.kron(poolLayer.delta[num], 
			np.ones(poolLayer.WSZ))

	layerBeforepoolLayer.delta = paddingDelta[:,:,0:poolLayer.paddingDeltaSize[-2], 0:poolLayer.paddingDeltaSize[-1]]


def computeGrad(beforeThisLayer, thisLayer):
	thisLayer.perGrad = np.zeros( (thisLayer.inputSize[0],*(thisLayer.WSZ)) )
	for num in np.arange(thisLayer.inputSize[0]):
		for k in np.arange(thisLayer.WSZ[0]):
			# pdb.set_trace()
			for c in np.arange(thisLayer.WSZ[1]):
				for i in np.arange(thisLayer.WSZ[-1]):
					for j in np.arange(thisLayer.WSZ[-2]):
						# print((k,c,i,j),thisLayer.layerType, beforeThisLayer.layerType)
						if thisLayer.layerType == 'convLayer' and thisLayer.channelControl[c, k] == 0:
							continue
						thisLayer.perGrad[num, k, c, i, j] = np.sum(
							thisLayer.delta[num, k]*beforeThisLayer.output[num, c, i:i+thisLayer.outputSize[-1], j:j+thisLayer.outputSize[-2]])
	# pdb.set_trace()
	thisLayer.grad = np.mean(thisLayer.perGrad, 0)