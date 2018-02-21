import numpy as np
import struct
import cv2


import pdb
def readImg(path, num=100):
    with open(path, 'rb') as imgfile:
        buff = imgfile.read(16)
        imageTotalNum, width, height = struct.unpack('>4x3L' , buff)

        # 每张图像28*28, 共提取num张
        imgs = np.zeros((num, 1, width, height))
        for i in range(num):
            buff = imgfile.read(width*height)
            imgs[i, 0, :, :]=np.array(struct.unpack('>'+str(width*height)+'B',buff)).reshape(width, height)

    return imgs

def readLabel(path, num=100):
    with open(path, 'rb') as labelfile:
        buff = labelfile.read(8)
        imageTotalNum = struct.unpack('>4x1L' , buff)

        # 每张图像28*28, 共提取num张
        labels = np.zeros((num, 10, 1, 1))
        for i in range(num):
            buff = labelfile.read(1)
            index = struct.unpack('>1B',buff)
            labels[i, index[0], :, :] = 1

    return labels 


def Load_MNIST(trainNum=100, testNum=20):
    '''
        作用：
            读取手写体数据MNIST，默认训练/测试=100/20
    '''

    print("load MNIST set ... ...")

    MNIST = {}
    train_image_path = r'./test_data/train-images.idx3-ubyte'
    train_label_path = r'./test_data/train-labels.idx1-ubyte'
    test_image_path = r'./test_data/t10k-images.idx3-ubyte'
    test_label_path = r'./test_data/t10k-labels.idx1-ubyte'
    
    MNIST['trainImgs'] = readImg(train_image_path, trainNum)
    MNIST['trainLabels'] = readLabel(train_label_path, trainNum)
    # cv2.imshow('test', trainImgs[0, 0, :, :])
    # cv2.waitKey(0)
    # print(trainLabels[0,:,0,0])
    MNIST['testImgs'] = readImg(test_image_path, testNum)
    MNIST['testLabels'] = readLabel(test_label_path, testNum)
    
    print("load finished.")
    return MNIST


if __name__ == '__main__':
    MNIST = Load_MNIST()