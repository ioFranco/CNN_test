from .forwards import Forwards
import numpy as np
import copy


import pdb
epsilon = 1e-8
def computeGradient(function, dependentLayerNum, *args):
	''' 
		作用：
			计算W的数值导数 
	    方法：
	    	利用微积分定义，取一个极小的扰动函数epsilon，(f(x+epsilon)-f(x))/epsilon
	    参数：
		    function: 
		    	CNN前馈的泛函
		    dependentLayerNum: 
		    	CNN网络结构中网络序号，例如[input, conv, softmax]，conv序号为1， softmax序号为2，以此类推
		    args: 
		    	CNN网络的结构（按顺序写入），顺序如上述
		注：  这里的delta_dependentLayer与old_dependentLayer都是dependentLayer的浅拷贝
	'''

	inputImgs = args[0].output
	labels = args[0].labels

	_args = copy.deepcopy(args)
	# pdb.set_trace()
	dependentLayer = _args[dependentLayerNum]

	old_dependentLayer = dependentLayer
	delta_dependentLayer = dependentLayer

	deltaCost_1 = np.zeros(dependentLayer.WSZ)
	deltaCost_2 = np.zeros(dependentLayer.WSZ)
	# pdb.set_trace()
	for k in np.arange(dependentLayer.WSZ[0]):    # 0 代表数量
		for c in np.arange(dependentLayer.WSZ[1]):    # 1 代表连接前一层数量
			for i in np.arange(dependentLayer.WSZ[-2]):    # 倒数第二个代表行
			    for j in np.arange(dependentLayer.WSZ[-1]):    # 倒数第一个代表列
			        ''' 计算 + delta 的函数值 '''
			        # pdb.set_trace()
			        delta_dependentLayer.W[k, c, i, j] = old_dependentLayer.W[k, c, i ,j]+epsilon
			        function.run(*_args)
			        deltaCost_1[k, c, i, j] = function.cost
			        delta_dependentLayer.W[k, c, i ,j] = old_dependentLayer.W[k, c, i, j]-epsilon

			        ''' 计算 - delta 的函数值 '''
			        delta_dependentLayer.W[k, c, i, j] = old_dependentLayer.W[k, c, i ,j]-epsilon
			        function.run(*_args)
			        deltaCost_2[k, c, i, j] = function.cost
			        delta_dependentLayer.W[k, c, i ,j] = old_dependentLayer.W[k, c, i, j]+epsilon

	return (deltaCost_1-deltaCost_2)/(2*epsilon)



class NumericalGradient:
	''' 计算数值梯度，用于进行数值梯度校验 '''


	def __init__(self):
		pass


	def run(self, *args):
		_forward = Forwards()

		index = 1
		for arg in args[1:]:
			if arg.layerType in Forwards.hasGrad_layerTypeSet:
				arg.numGradient = computeGradient(_forward, index, *args)
			index += 1


