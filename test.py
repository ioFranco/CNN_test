from CNN_test import conv
import numpy as np
delta = np.zeros((2,8,8))
delta[0,3:5,3:5] = np.array([[0.08375,0.04044],[0.02609,0.01898]])
W = 0.1/( (np.arange(1*2*4*4) +1).reshape((1,2,4,4)) )
output = np.zeros((1,5,5))
conv(output, delta, W, (1,2,4,4), np.ones((2,1)), (1,5,5))
before = np.array(
	[[0.,0.,0.,0.,0.],
 	[  0.,0.,0.,0.,0.],
 	[  0.,0.,0.,0.,2.8],
 	[  0.,0.,0.,2.64444,9.975],
 	[  0.,0.,1.97778,8.85833,23.11397]])
print(before*output[0])