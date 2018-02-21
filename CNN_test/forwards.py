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

        softmaxLayer = args[-1]
        self.predict = np.zeros(softmaxLayer.output.shape)
        self.predict[np.argmax(softmaxLayer.output)] = 1

        # pdb.set_trace()
        self.delta = labels - softmaxLayer.output
        # pdb.set_trace()
        self.cost = np.sum(labels*np.log(softmaxLayer.output))


