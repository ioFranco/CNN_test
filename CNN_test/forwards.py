import numpy as np


import pdb
class Forwards:
    ''' 
        作用：用于CNN 网络的前向传播。
        cost：记录前向传播的差异，
        run()：用于启动前向传播
    '''

    def __init__(self):
        self.cost = 0
        Forwards.layerTypeSet = {'inputLayer', 'convLayer', 'poolLayer', 'softmaxLayer'}
        Forwards.hasGrad_layerTypeSet = {'convLayer', 'softmaxLayer'}

    def run(self, *args):
        
        # pdb.set_trace()
        inputImgs = args[0].output
        labels = args[0].labels

        index = 1
        for arg in args[1:]:
            # print(str(args[index].layerCount)+'\t'+args[index].layerType)
            if index == 1:
                arg.run(inputImgs)
            else:
                arg.run(args[index-1].output)
            index = index+1

        # convLayer = args[0] 
        # softmaxLayer = args[1]
        # convLayer.run(inputImgs)
        # softmaxLayer.run(convLayer.output)

        # pdb.set_trace()
        softmaxLayer = args[-1]

        self.predict = np.zeros(softmaxLayer.outputSize)
        self.delta = np.zeros(softmaxLayer.outputSize)
        self.perCost = np.zeros((softmaxLayer.outputSize[0],1))
        for num in np.arange(softmaxLayer.outputSize[0]):
            self.predict[num, np.argmax(softmaxLayer.output[num])] = 1
            # pdb.set_trace()
            self.delta[num] = labels[num] - softmaxLayer.output[num]
            # pdb.set_trace()
            self.perCost[num] = np.sum(labels[num]*np.log(softmaxLayer.output[num]))

        # pdb.set_trace()
        self.cost = np.mean(self.perCost)


