from copy import deepcopy


class BasicLayer:
	count = 0
	def __init__(self, layerType):
		'''
			作用：
				表明该层的类型，卷积层的类型是"softmaxLayer"
		'''
		
		BasicLayer.count += 1
		self.layerType = layerType
		self.layerCount = BasicLayer.count
		# print(str(self.layerCount)+'\t'+self.layerType)


	def copy(self):
		'''
			作用：
				深拷贝
		'''
		return deepcopy(self)

