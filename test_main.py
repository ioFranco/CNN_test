import cv2
import numpy as np
from CNN_test import InputLayer
from CNN_test import ConvLayer
from CNN_test import PoolLayer
from CNN_test import SoftmaxLayer
from CNN_test import Forwards
from CNN_test import Backwards
from CNN_test import NumericalGradient
from read_data import Load_MNIST

import pdb
# 准备数据
MNIST = Load_MNIST()

inputLayer = InputLayer(imgs=MNIST['trainImgs'], labels=MNIST['trainLabels'])

''' 定义网络，前向传播 '''
convLayer = ConvLayer( W_size=(2,1,3,3), channelControl=np.array([[1,1]]) )
poolLayer = PoolLayer( W_size=(2,2), stepSize=(2,2) )
convLayer2 = ConvLayer( W_size=(2,2,3,3), channelControl=np.array([[1,0],[0,1]]) )
# convLayer3 = ConvLayer( W_size=(2,2,4,4), channelControl=np.array([[1,0],[0,0]]) )
poolLayer2 = PoolLayer( W_size=(2,2), stepSize=(2,2) )
softmaxLayer = SoftmaxLayer( W_size=(10,2,2,2) )
net = (inputLayer, convLayer, poolLayer, convLayer2, poolLayer2, softmaxLayer)

forward = Forwards()
# pdb.set_trace()
forward.run(*net)
# pdb.set_trace()
# print(forward.delta)
# print('\n')
# cv2.imshow('img', cv2.resize(convLayer.output, (30,30)))
# cv2.waitKey(0)
backward = Backwards(forward)
backward.run(*net)
# pdb.set_trace()
np.set_printoptions(precision=5, linewidth=300, suppress=True)
print(convLayer.grad)
print('\n------------------------------------------------\n')
# print(poolLayer.output)
# print('\n------------------------------------------------\n')
# print(convLayer.output)
# print('\n------------------------------------------------\n')

numericalGradient = NumericalGradient()
numericalGradient.run(*net)
print(convLayer.numGradient)